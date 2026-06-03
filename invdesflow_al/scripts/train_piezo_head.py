"""Train the Plan C piezoelectric scoring head.

Loads `data_raw/mp_piezo.jsonl` (matminer-piezoelectric_tensor by default;
MP-API data drops in via the same JSONL schema), trains a small EGNN
regressor (`PiezoHead`) on log(|eij_max|), and reports val Spearman rho.

Why log target: eij_max ~ 0..46 C/m^2 with median 0.25 — heavy tail.
We regress log(target + 0.01), which makes the loss scale-stable and
the Spearman ranking meaningful.

Why stratified split: only ~20 Nb-containing and ~20 Ti-containing
entries exist. A naive random 80/20 split could put all Ti/Nb in train
or all in val, both bad. We stratify on point-group family so each
family appears in both splits.

Acceptance: val Spearman rho >= 0.5. We are training a RANKER for AL,
not a calibrated predictor — Spearman is the right metric.

Usage:
    python -m invdesflow_al.scripts.train_piezo_head \
        --data data_raw/mp_piezo.jsonl \
        --out checkpoints/piezo_head.ckpt \
        --epochs 200 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from ..data.batch import collate
from ..data.representation import Crystal
from ..models.piezo_head import PiezoHead


TARGET_OFFSET = 0.01   # log(target + 0.01) — handles target==0 cases


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def record_to_crystal_and_target(rec: dict) -> tuple[Crystal, float]:
    c = Crystal(
        atom_types=torch.tensor(rec["z"], dtype=torch.long),
        frac_coords=torch.tensor(rec["frac"], dtype=torch.float),
        lattice=torch.tensor(rec["lattice"], dtype=torch.float),
    )
    return c, float(rec["target"])


def stratified_split(
    recs: list[dict], val_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    """Round-robin split per point-group bucket; rare groups land mostly in train."""
    rng = torch.Generator().manual_seed(seed)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(recs):
        buckets[r.get("point_group", "unknown") or "unknown"].append(i)
    train_idx, val_idx = [], []
    for pg, idxs in buckets.items():
        perm = torch.randperm(len(idxs), generator=rng).tolist()
        idxs = [idxs[i] for i in perm]
        n_val = max(1, int(round(len(idxs) * val_frac))) if len(idxs) > 2 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return sorted(train_idx), sorted(val_idx)


def spearman(pred: Tensor, true: Tensor) -> float:
    """Plain Spearman rho (handles ties via average ranks via argsort.argsort)."""
    if pred.numel() < 2:
        return float("nan")
    rp = pred.argsort().argsort().float()
    rt = true.argsort().argsort().float()
    rp = rp - rp.mean()
    rt = rt - rt.mean()
    denom = (rp.norm() * rt.norm()).clamp_min(1e-12)
    return float((rp * rt).sum() / denom)


def iterate_batches(
    indices: list[int],
    crystals: list[Crystal],
    targets_log: Tensor,
    batch_size: int,
    shuffle: bool,
    rng: torch.Generator | None,
):
    order = list(range(len(indices)))
    if shuffle:
        order = torch.randperm(len(indices), generator=rng).tolist()
    for i in range(0, len(order), batch_size):
        chunk = order[i:i + batch_size]
        chunk_idx = [indices[j] for j in chunk]
        crysts = [crystals[j] for j in chunk_idx]
        y = targets_log[torch.tensor(chunk_idx, dtype=torch.long)]
        yield collate(crysts, mode="complete"), y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_raw/mp_piezo.jsonl")
    ap.add_argument("--out", default="checkpoints/piezo_head.ckpt")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--fourier-freqs", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=30,
                    help="early-stop if val Spearman doesn't improve for this many epochs")
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)

    print(f"loading {args.data} ...", flush=True)
    recs = load_jsonl(args.data)
    crystals = []
    raw_targets = []
    for r in recs:
        c, y = record_to_crystal_and_target(r)
        crystals.append(c)
        raw_targets.append(y)
    raw_targets_t = torch.tensor(raw_targets, dtype=torch.float)
    log_targets = torch.log(raw_targets_t + TARGET_OFFSET)
    print(f"  N={len(recs)}  target=log(eij_max+{TARGET_OFFSET})  "
          f"log-target range [{log_targets.min():.3f}, {log_targets.max():.3f}]",
          flush=True)

    train_idx, val_idx = stratified_split(recs, args.val_frac, args.seed)
    print(f"  train/val: {len(train_idx)}/{len(val_idx)}  "
          f"(stratified on point_group)", flush=True)

    # quick coverage check: how many Nb / Ti / Ba / Bi end up in val
    for Z, sym in [(41, "Nb"), (22, "Ti"), (56, "Ba"), (83, "Bi")]:
        n_tr = sum(1 for i in train_idx if Z in recs[i]["z"])
        n_va = sum(1 for i in val_idx if Z in recs[i]["z"])
        print(f"    {sym} (Z={Z}): train={n_tr}  val={n_va}", flush=True)

    model = PiezoHead(
        hidden=args.hidden,
        num_layers=args.num_layers,
        fourier_freqs=args.fourier_freqs,
        dropout=args.dropout,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  PiezoHead: hidden={args.hidden} layers={args.num_layers} "
          f"freqs={args.fourier_freqs} dropout={args.dropout}  "
          f"params={n_params}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    def lr_at(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return args.lr * (epoch + 1) / max(args.warmup_epochs, 1)
        prog = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    loss_fn = nn.SmoothL1Loss()

    best_val_rho = -1.0
    best_val_mse = float("inf")
    best_epoch = -1
    patience_left = args.patience
    history = []

    t0 = time.time()
    for epoch in range(args.epochs):
        for g in opt.param_groups:
            g["lr"] = lr_at(epoch)
        model.train()
        train_losses = []
        for batch, y in iterate_batches(
            train_idx, crystals, log_targets,
            args.batch_size, shuffle=True, rng=rng,
        ):
            batch = batch.to(args.device)
            y = y.to(args.device)
            pred = model(batch)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(loss.item())

        # eval
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for batch, y in iterate_batches(
                val_idx, crystals, log_targets,
                args.batch_size, shuffle=False, rng=None,
            ):
                batch = batch.to(args.device)
                pred = model(batch).cpu()
                preds.append(pred)
                trues.append(y)
        preds = torch.cat(preds)
        trues = torch.cat(trues)
        val_mse = float(((preds - trues) ** 2).mean())
        val_rho = spearman(preds, trues)
        train_mean = float(torch.tensor(train_losses).mean()) if train_losses else float("nan")

        history.append({
            "epoch": epoch,
            "lr": lr_at(epoch),
            "train_loss": train_mean,
            "val_mse": val_mse,
            "val_spearman": val_rho,
        })
        improved = val_rho > best_val_rho + 1e-4
        if improved:
            best_val_rho = val_rho
            best_val_mse = val_mse
            best_epoch = epoch
            patience_left = args.patience
            torch.save({
                "state_dict": model.state_dict(),
                "cfg": {
                    "hidden": args.hidden,
                    "num_layers": args.num_layers,
                    "fourier_freqs": args.fourier_freqs,
                    "dropout": args.dropout,
                },
                "epoch": epoch,
                "val_spearman": val_rho,
                "val_mse": val_mse,
                "target_name": "log(eij_max + 0.01)",
                "target_offset": TARGET_OFFSET,
                "source_data": args.data,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            }, args.out)
        else:
            patience_left -= 1

        if epoch % args.log_every == 0 or improved or epoch == args.epochs - 1:
            tag = "*" if improved else " "
            print(f"  ep {epoch:3d} {tag} lr={lr_at(epoch):.4g} "
                  f"train={train_mean:.4f} "
                  f"val_mse={val_mse:.4f} val_rho={val_rho:.4f} "
                  f"(best rho={best_val_rho:.4f} @ ep{best_epoch}; "
                  f"pat={patience_left}; t={time.time()-t0:.0f}s)",
                  flush=True)

        if patience_left <= 0:
            print(f"  early stop at epoch {epoch}", flush=True)
            break

    # save history alongside the checkpoint
    hist_path = Path(str(args.out).replace(".ckpt", "_history.json"))
    hist_path.write_text(json.dumps({
        "best_val_spearman": best_val_rho,
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "config": vars(args),
        "history": history,
    }, indent=2))
    print(f"\ndone. best val Spearman rho = {best_val_rho:.4f} @ epoch {best_epoch}  "
          f"(val MSE {best_val_mse:.4f})")
    print(f"checkpoint -> {args.out}")
    print(f"history    -> {hist_path}")
    print(f"acceptance (rho >= 0.5): {'PASS' if best_val_rho >= 0.5 else 'FAIL'}")


if __name__ == "__main__":
    main()
