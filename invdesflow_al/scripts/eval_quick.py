"""Fast generator eval: held-out per-channel loss + small-N sampling sanity.

Two proxies for the full Fig S.4 eval (which samples 1k+ crystals at T=1000):

  (1) Per-channel held-out denoising loss.  Forward pass only, no sampling.
      Mirrors the training objective on records the model never saw.  Splits
      total loss into A / L / F so collapses are visible per channel (Entry 3
      type-only collapse, Entry 7 lattice tail, etc.).

  (2) Small-N sampling reality check.  N=64 by default, full T=1000 reverse.
      Reports chemical-formula unique rate + volume-per-atom sanity.  Unbiased
      estimator of the Fig S.4 number, just noisier than N=1000.

Held-out set = `split_records(...)` val split with the same seed as training,
so we test on records that were excluded from the train DataLoader.

Usage:
  python -m invdesflow_al.scripts.eval_quick \
      --ckpt gen_10k_ax0.ckpt --manifest data_raw/pretrain.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from functools import reduce
from math import gcd

import torch

from ..data.datasets import filter_records, load_structures
from ..data.torch_dataset import CrystalDataset, make_collate, split_records
from ..models.generator import CrystalGenerator


def reduced_formula(z: list[int]) -> str:
    c = Counter(z)
    g = reduce(gcd, c.values()) if c else 1
    return "-".join(f"{e}{n // g}" for e, n in sorted(c.items()))


def heldout_loss(gen: CrystalGenerator, records, n_loss: int, batch_size: int,
                 cutoff: float, max_nbr: int, graph_mode: str, seed: int) -> dict:
    """Per-channel forward-pass loss on n_loss held-out records.

    Random t per record (the training distribution), no grad.  Returns
    weighted total + raw per-channel means matching DiffusionProcess.training_loss.
    """
    torch.manual_seed(seed)
    records = records[:n_loss]
    ds = CrystalDataset(records)
    coll = make_collate(cutoff, max_nbr, graph_mode)

    sums = {"total": 0.0, "coord": 0.0, "lattice": 0.0, "type": 0.0}
    n_batches = 0
    gen.eval()
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            crystals = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
            batch = coll(crystals)
            losses = gen.loss_on_batch(batch)
            for k in sums:
                sums[k] += float(losses[k])
            n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in sums.items()}


def sample_sanity(gen: CrystalGenerator, records, n_samples: int, gen_batch: int,
                  seed: int) -> dict:
    """Small-N sampling check: unique rate + vpa sanity.

    Atom counts drawn from the held-out records' histogram (matches the
    eval_unique_rate.py convention).
    """
    g = torch.Generator().manual_seed(seed)
    hist_t = torch.tensor([r.num_atoms for r in records])

    formulas: list[str] = []
    vpa: list[float] = []
    nan = 0
    while len(formulas) < n_samples:
        bs = min(gen_batch, n_samples - len(formulas))
        idx = torch.randint(0, len(hist_t), (bs,), generator=g)
        n_atoms = hist_t[idx].tolist()
        for c in gen.sample(num_atoms=n_atoms):
            formulas.append(reduced_formula(c.atom_types.tolist()))
            if torch.isfinite(c.lattice).all():
                vpa.append(float(torch.det(c.lattice).abs()) / c.num_atoms)
            else:
                nan += 1

    vt = torch.tensor(vpa) if vpa else torch.zeros(1)
    sane = float(((vt > 0) & (vt <= 500)).float().mean())
    return {
        "n": n_samples,
        "unique_rate": len(set(formulas)) / n_samples,
        "vpa_median": round(float(vt.median()), 2),
        "vpa_max": round(float(vt.max()), 2),
        "sane_fraction": round(sane, 3),
        "nan": nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n-loss", type=int, default=256,
                    help="held-out records for per-channel loss (forward only)")
    ap.add_argument("--n-samples", type=int, default=64,
                    help="N for the sampling reality check (full T=1000 reverse)")
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--loss-batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval_quick.json")
    args = ap.parse_args()

    t0 = time.time()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    gen = CrystalGenerator(ckpt["cfg"], device=args.device)
    gen.load_state_dict(ckpt["state_dict"])
    gen.eval()
    epoch = ckpt.get("epoch", "?")
    ckpt_val = ckpt.get("val_loss", float("nan"))
    print(f"loaded {args.ckpt} (epoch {epoch}, ckpt val_loss {ckpt_val})",
          flush=True)

    # held-out val split — same seed as training, so these records were never
    # seen by the train DataLoader.
    cfg_data = ckpt["cfg"]["data"]
    max_atoms = int(cfg_data["max_atoms_per_structure"])
    cutoff = float(cfg_data["neighbor_cutoff_radius"])
    max_nbr = int(cfg_data["max_neighbors_per_atom"])
    graph_mode = str(cfg_data.get("graph_construction", "complete"))

    records = list(filter_records(load_structures(args.manifest),
                                  max_atoms=max_atoms))
    _, va, _ = split_records(records)
    print(f"records={len(records)} held_out_val={len(va)}", flush=True)

    # (1) per-channel held-out loss
    t1 = time.time()
    loss = heldout_loss(gen, va, args.n_loss, args.loss_batch,
                        cutoff, max_nbr, graph_mode, args.seed)
    t_loss = time.time() - t1
    print(f"loss[{args.n_loss}] total={loss['total']:.4f} "
          f"A={loss['type']:.4f} L={loss['lattice']:.4f} F={loss['coord']:.4f} "
          f"({t_loss:.1f}s)", flush=True)

    # (2) small-N sampling sanity
    t2 = time.time()
    samp = sample_sanity(gen, va, args.n_samples, args.gen_batch, args.seed)
    t_samp = time.time() - t2
    print(f"sample[{samp['n']}] unique={samp['unique_rate']:.3f} "
          f"sane={samp['sane_fraction']:.3f} vpa_med={samp['vpa_median']} "
          f"vpa_max={samp['vpa_max']} nan={samp['nan']} ({t_samp:.0f}s)",
          flush=True)

    wall = time.time() - t0
    out = {
        "ckpt": args.ckpt, "epoch": epoch, "ckpt_val_loss": ckpt_val,
        "heldout_loss": loss, "sample": samp,
        "wall_seconds": round(wall, 1),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}  (total {wall:.0f}s)", flush=True)
    print(f"QUICK ckpt={args.ckpt} epoch={epoch} val={ckpt_val:.4f} | "
          f"loss tot/A/L/F={loss['total']:.3f}/{loss['type']:.3f}/"
          f"{loss['lattice']:.3f}/{loss['coord']:.3f} | "
          f"N={samp['n']} unique={samp['unique_rate']:.3f} "
          f"sane={samp['sane_fraction']:.3f} | wall={wall:.0f}s")


if __name__ == "__main__":
    main()
