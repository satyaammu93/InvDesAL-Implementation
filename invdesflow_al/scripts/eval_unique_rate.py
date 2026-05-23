"""Step 4: chemical-formula unique rate vs Supplementary Fig. S.4.

Fig S.4 reports the ratio of unique chemical formulas as the pretrained
generator produces 1k ... 256k samples:

    1000:0.992 2000:0.989 4000:0.984 8000:0.973 16000:0.959
    32000:0.935 64000:0.899 128000:0.851 256000:0.789

We sample N crystals (atom counts drawn from the training-set histogram),
compute distinct reduced-formula count / N at each checkpoint, and print a
side-by-side comparison with the paper.

Usage:
  python -m invdesflow_al.scripts.eval_unique_rate \
      --ckpt generator.ckpt --manifest data_raw/pretrain.jsonl \
      --max-samples 16000 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from functools import reduce
from math import gcd

import torch

from ..data.datasets import load_structures
from ..models.generator import CrystalGenerator

PAPER = {
    1000: 0.992, 2000: 0.989, 4000: 0.984, 8000: 0.973, 16000: 0.959,
    32000: 0.935, 64000: 0.899, 128000: 0.851, 256000: 0.789,
}


def reduced_formula(z: list[int]) -> str:
    c = Counter(z)
    g = reduce(gcd, c.values()) if c else 1
    return "-".join(f"{e}{n // g}" for e, n in sorted(c.items()))


def atom_count_hist(manifest: str, cap: int = 200000) -> list[int]:
    counts = []
    for i, r in enumerate(load_structures(manifest)):
        counts.append(len(r.z))
        if i + 1 >= cap:
            break
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max-samples", type=int, default=16000)
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="unique_rate.json")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    gen = CrystalGenerator(ckpt["cfg"], device=args.device)
    gen.load_state_dict(ckpt["state_dict"])
    gen.eval()
    print(f"loaded {args.ckpt} (epoch {ckpt.get('epoch','?')}, "
          f"val {ckpt.get('val_loss','?')})")

    g = torch.Generator().manual_seed(args.seed)
    hist = atom_count_hist(args.manifest)
    hist_t = torch.tensor(hist)

    checkpoints = [n for n in sorted(PAPER) if n <= args.max_samples]
    if not checkpoints:
        checkpoints = [args.max_samples]

    formulas: list[str] = []
    vpa: list[float] = []          # volume per atom of each sampled crystal
    nan = 0
    t0 = time.time()
    target = max(checkpoints)
    while len(formulas) < target:
        bs = min(args.gen_batch, target - len(formulas))
        idx = torch.randint(0, len(hist_t), (bs,), generator=g)
        n_atoms = hist_t[idx].tolist()
        for c in gen.sample(num_atoms=n_atoms):
            formulas.append(reduced_formula(c.atom_types.tolist()))
            if torch.isfinite(c.lattice).all():
                vpa.append(float(torch.det(c.lattice).abs()) / c.num_atoms)
            else:
                nan += 1
        if len(formulas) % 1000 < args.gen_batch:
            print(f"  generated {len(formulas)}/{target} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'N':>8} {'ours':>8} {'paper':>8} {'delta':>8}")
    rows = []
    for n in checkpoints:
        uniq = len(set(formulas[:n])) / n
        p = PAPER.get(n)
        delta = f"{uniq - p:+.3f}" if p is not None else "  n/a"
        print(f"{n:>8} {uniq:>8.3f} {('%.3f'%p) if p else '   n/a':>8} {delta:>8}")
        rows.append({"n": n, "ours": uniq, "paper": p})

    # lattice sanity over all sampled crystals
    vt = torch.tensor(vpa) if vpa else torch.zeros(1)
    q = lambda f: float(vt.sort().values[min(len(vt) - 1, int(f * len(vt)))])
    sane_frac = float(((vt > 0) & (vt <= 500)).float().mean())
    lat = {"vpa_min": round(float(vt.min()), 2), "vpa_p5": round(q(0.05), 2),
           "vpa_median": round(float(vt.median()), 2), "vpa_p95": round(q(0.95), 2),
           "vpa_max": round(float(vt.max()), 2), "nan": nan,
           "sane_fraction": round(sane_frac, 3)}
    print(f"\nlattice volume/atom (A^3): min {lat['vpa_min']}  p5 {lat['vpa_p5']}  "
          f"median {lat['vpa_median']}  p95 {lat['vpa_p95']}  max {lat['vpa_max']}")
    print(f"sane fraction (0<vpa<=500): {lat['sane_fraction']}   nan/inf: {nan}")

    with open(args.out, "w") as f:
        json.dump({"unique_rate": rows, "lattice": lat}, f, indent=2)
    print(f"\nwrote {args.out}")
    print("RESULT_JSON " + json.dumps({
        "unique_rate_at_max": rows[-1]["ours"], "n_max": rows[-1]["n"],
        **lat}))


if __name__ == "__main__":
    main()
