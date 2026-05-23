"""Pretraining entrypoint for the crystal generation model.

Table S.2 protocol: Adam(1e-4) + ReduceLROnPlateau(0.6,30), batch 96/64/64,
up to 1000 epochs. On a small GPU we instead run a fixed WALLCLOCK budget
(--max-hours) with periodic + best-val checkpointing, so an overnight run
always leaves a usable checkpoint even if interrupted.

Usage:
  python -m invdesflow_al.scripts.train_generator \
      --manifest pretrain.jsonl --device cuda \
      --max-hours 9 --batch 64 --workers 0 --ckpt-every 800
"""

from __future__ import annotations

import argparse
import time

import torch

from ..data.datasets import compute_lattice_stats, filter_records, load_structures
from ..data.torch_dataset import make_dataloaders
from ..models.generator import CrystalGenerator, load_config


def _save(path, gen, cfg, epoch, val_loss, step_global, opt=None, sched=None):
    payload = {"cfg": cfg, "state_dict": gen.state_dict(),
               "epoch": epoch, "val_loss": val_loss, "global_step": step_global}
    if opt is not None:
        payload["optimizer"] = opt.state_dict()
    if sched is not None:
        payload["scheduler"] = sched.state_dict()
    torch.save(payload, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop after this wallclock; always checkpoints on exit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt", default="generator.ckpt")
    ap.add_argument("--batch", type=int, default=None, help="override train batch")
    ap.add_argument("--auto-batch", action="store_true",
                    help="probe the largest batch that fits VRAM before training")
    ap.add_argument("--workers", type=int, default=None,
                    help="override dataloader workers (use 0 on low-RAM hosts)")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=1000,
                    help="also checkpoint every N global steps (latest.ckpt)")
    ap.add_argument("--resume", default=None,
                    help="resume from this checkpoint (loads state_dict, "
                         "optimizer + scheduler if present, picks up epoch counter)")
    args = ap.parse_args()

    cfg = load_config()
    if args.batch is not None:
        cfg["data"]["batch_size_train"] = args.batch
    if args.workers is not None:
        cfg["data"]["num_preprocess_workers"] = args.workers
    epochs = args.epochs or int(cfg["optim"]["epochs"])
    deadline = (time.time() + args.max_hours * 3600) if args.max_hours else None

    records = list(filter_records(
        load_structures(args.manifest),
        max_atoms=int(cfg["data"]["max_atoms_per_structure"]),
    ))
    train_dl, val_dl, _ = make_dataloaders(records, cfg)
    # lattice channel: dataset mean/std for the normalized x0-diffusion
    lm, ls = compute_lattice_stats(records)
    cfg["diffusion"]["lattice_mean"] = lm.tolist()
    cfg["diffusion"]["lattice_std"] = ls.tolist()
    print(f"records={len(records)} train_batches={len(train_dl)} "
          f"val_batches={len(val_dl)} batch={cfg['data']['batch_size_train']} "
          f"device={args.device} budget={args.max_hours}h", flush=True)

    gen = CrystalGenerator(cfg, device=args.device)

    if args.auto_batch and args.device.startswith("cuda"):
        from ..data.batch import collate
        from ..data.representation import Crystal
        chosen = cfg["data"]["batch_size_train"]
        for cand in (96, 80, 64, 48, 32, 24, 16):
            if cand > len(records):
                continue
            try:
                cs = [records[i].to_crystal() for i in range(cand)]
                b = collate(cs, gen.cutoff, gen.max_nbr)
                o = gen.loss_on_batch(b)
                o["total"].backward()
                gen.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                chosen = cand
                break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
        cfg["data"]["batch_size_train"] = chosen
        print(f"auto-batch -> {chosen}", flush=True)
        train_dl, val_dl, _ = make_dataloaders(records, cfg)

    opt, sched = gen.configure_optimizer()
    best = float("inf")
    gstep = 0
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        gen.load_state_dict(ck["state_dict"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            sched.load_state_dict(ck["scheduler"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        gstep = int(ck.get("global_step", 0))
        best = float(ck.get("val_loss", float("inf")))
        print(f"resumed from {args.resume}: epoch={start_epoch} gstep={gstep} "
              f"prior_val={best:.4f}", flush=True)
    latest = args.ckpt.replace(".ckpt", "_latest.ckpt")
    t0 = time.time()
    stop = False

    for epoch in range(start_epoch, start_epoch + epochs):
        gen.train()
        run = n = 0.0
        for step, batch in enumerate(train_dl):
            out = gen.loss_on_batch(batch)
            opt.zero_grad()
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
            opt.step()
            run += float(out["total"]); n += 1; gstep += 1
            if step % args.log_every == 0:
                el = (time.time() - t0) / 3600
                print(f"  e{epoch} s{step} g{gstep} loss {float(out['total']):.4f} "
                      f"(c/l/t {out['coord']:.3f}/{out['lattice']:.3f}/"
                      f"{out['type']:.3f}) {el:.2f}h", flush=True)
            if gstep % args.ckpt_every == 0:
                _save(latest, gen, cfg, epoch, best, gstep, opt, sched)
            if deadline and time.time() > deadline:
                stop = True
                break

        gen.eval()
        with torch.no_grad():
            vrun = vn = 0.0
            for batch in val_dl:
                vrun += float(gen.loss_on_batch(batch)["total"]); vn += 1
        val_loss = vrun / max(vn, 1)
        sched.step(val_loss)
        print(f"epoch {epoch:4d} train {run / max(n,1):.4f} val {val_loss:.4f} "
              f"lr {opt.param_groups[0]['lr']:.2e} "
              f"{(time.time()-t0)/3600:.2f}h", flush=True)

        if val_loss < best:
            best = val_loss
            _save(args.ckpt, gen, cfg, epoch, val_loss, gstep, opt, sched)
        _save(latest, gen, cfg, epoch, val_loss, gstep, opt, sched)
        if stop:
            print(f"wallclock budget reached at epoch {epoch}", flush=True)
            break

    print(f"done. best val {best:.4f} -> {args.ckpt} (latest -> {latest})",
          flush=True)


if __name__ == "__main__":
    main()
