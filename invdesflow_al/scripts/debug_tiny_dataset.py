"""Gate 9 (PROGRESS.md Entry 4): does tiny-dataset sampling retain diversity?

Overfits the generator on a small set of diverse real crystals, samples many,
and checks that the samples are diverse (not collapsed) and physically sane.

PASS: nonzero formula diversity, generated elements roughly within the
training element support, finite/plausible lattices.

Usage:
  python -m invdesflow_al.scripts.debug_tiny_dataset \
      --manifest data_raw/pretrain.jsonl --k 32 --steps 2500 \
      --n-sample 256 --device cuda
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch

from ..data.datasets import load_structures
from ..models.generator import CrystalGenerator, config_with_lattice_stats


def reduced(z):
    from functools import reduce
    from math import gcd
    c = Counter(z)
    g = reduce(gcd, c.values()) if c else 1
    return "-".join(f"{e}:{n // g}" for e, n in sorted(c.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_raw/pretrain.jsonl")
    ap.add_argument("--k", type=int, default=32, help="# diverse crystals to overfit")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--n-sample", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    crystals = []
    for r in load_structures(args.manifest):
        crystals.append(r.to_crystal())
        if len(crystals) >= args.k:
            break
    train_elems = {z for c in crystals for z in c.atom_types.tolist()}
    train_forms = {reduced(c.atom_types.tolist()) for c in crystals}
    n_atoms = [c.num_atoms for c in crystals]
    print(f"{len(crystals)} crystals  | {len(train_forms)} distinct formulas | "
          f"{len(train_elems)} elements | N range {min(n_atoms)}-{max(n_atoms)}\n")

    gen = CrystalGenerator(config_with_lattice_stats(args.manifest), device=args.device)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-4)
    for step in range(args.steps):
        out = gen.train_step(crystals)
        opt.zero_grad()
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
        opt.step()
        if step % max(args.steps // 6, 1) == 0 or step == args.steps - 1:
            print(f"step {step:5d}  c/l/t = {out['coord']:.4f}/"
                  f"{out['lattice']:.4f}/{out['type']:.4f}")

    print(f"\nsampling {args.n_sample} crystals ...")
    g = torch.Generator().manual_seed(0)
    sample_N = [n_atoms[int(torch.randint(0, len(n_atoms), (1,), generator=g))]
                for _ in range(args.n_sample)]
    samples = gen.sample(num_atoms=sample_N)

    forms = Counter(reduced(s.atom_types.tolist()) for s in samples)
    elems = Counter(z for s in samples for z in s.atom_types.tolist())
    dets = torch.tensor([float(torch.det(s.lattice).abs()) for s in samples])
    vpa = torch.tensor([float(torch.det(s.lattice).abs()) / s.num_atoms
                        for s in samples])
    nan = sum(int(not torch.isfinite(s.lattice).all()) for s in samples)
    uniq = len(forms) / len(samples)
    in_support = sum(elems[z] for z in elems if z in train_elems) / max(sum(elems.values()), 1)

    print(f"\ndistinct formulas : {len(forms)}/{len(samples)}  "
          f"(unique rate {uniq:.3f})")
    print(f"top formulas      : {forms.most_common(4)}")
    print(f"distinct elements : {len(elems)}  | fraction within training "
          f"support: {in_support:.2f}")
    print(f"lattice |det|     : min/med/max = "
          f"{dets.min():.1f}/{dets.median():.1f}/{dets.max():.1f}")
    print(f"volume/atom (A^3) : min/med/max = "
          f"{vpa.min():.1f}/{vpa.median():.1f}/{vpa.max():.1f}")
    print(f"nan/inf lattices  : {nan}")

    fails = []
    if uniq < 0.2:
        fails.append(f"formula unique rate {uniq:.3f} too low (collapse)")
    if nan:
        fails.append(f"{nan} nan/inf lattices")
    if vpa.min() <= 0 or vpa.max() > 500:
        fails.append(f"volume/atom out of (0,500]: {vpa.min():.1f}-{vpa.max():.1f}")
    if in_support < 0.5:
        fails.append(f"only {in_support:.2f} of atoms within training elements")
    top_frac = forms.most_common(1)[0][1] / len(samples)
    if top_frac > 0.95:
        fails.append(f"one formula dominates {top_frac:.0%}")

    print()
    if fails:
        print("GATE 9 FAIL:")
        for f in fails:
            print("  -", f)
    else:
        print("GATE 9 PASS - tiny-dataset sampling stays diverse and sane")

    import json
    print("RESULT_JSON " + json.dumps({
        "gate": 9, "pass": not fails,
        "unique_rate": round(uniq, 4),
        "distinct_formulas": len(forms),
        "vpa_min": round(float(vpa.min()), 3),
        "vpa_median": round(float(vpa.median()), 3),
        "vpa_max": round(float(vpa.max()), 3),
        "nan": nan, "in_support": round(in_support, 3),
    }))


if __name__ == "__main__":
    main()
