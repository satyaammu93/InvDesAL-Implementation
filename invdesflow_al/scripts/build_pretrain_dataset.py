"""Assemble the pretraining manifest (REBUILD_PLAN.md sec. 6).

Pipeline (paper, Methods):
  Alex-MP-20 + GNoME  ->  filter (<=20 atoms)  ->  de-dup  ->
  [optional] exclude CSP test sets (Perov-5/MP-20/MPTS-52)  ->
  diversity sampling  ->  manifest.jsonl

The manifest is the dependency-free JSONL consumed by train_generator.py.

Example:
  python -m invdesflow_al.scripts.build_pretrain_dataset \
      --inputs alex_mp_20.jsonl:alex-mp-20  gnome/cifs:gnome \
      --exclude perov5.jsonl mp20.jsonl mpts52.jsonl \
      --target-size 800000 --out pretrain_manifest.jsonl

Fine-tune variants (Fig S.7):
  low formation : --e-form-max -0.5
  low E_hull    : --e-hull-max 0.05
"""

from __future__ import annotations

import argparse
import time

from ..data.datasets import (
    dedup_records,
    filter_records,
    load_structures,
    write_manifest,
)
from ..data.diversity import DiversitySampler
from ..data.torch_dataset import apply_exclusion, exclusion_keyset


def _parse_input(spec: str):
    if ":" in spec and not spec.split(":")[0].endswith((".jsonl", ".json")):
        path, src = spec.rsplit(":", 1)
    elif spec.count(":") and spec.rsplit(":", 1)[1].isalpha():
        path, src = spec.rsplit(":", 1)
    else:
        path, src = spec, None
    return path, src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs", nargs="+", required=True,
        help="dataset specs PATH or PATH:SOURCE (e.g. alex.jsonl:alex-mp-20)",
    )
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="CSP test-set paths to hold out (Perov-5/MP-20/MPTS-52)")
    ap.add_argument("--max-atoms", type=int, default=20)          # Table S.2
    ap.add_argument("--e-form-max", type=float, default=None)     # Fig S.7a
    ap.add_argument("--e-hull-max", type=float, default=None)     # Fig S.7b
    ap.add_argument("--target-size", type=int, default=None)
    ap.add_argument("--sampler", default="round_robin",
                    choices=["round_robin", "inverse_freq"])
    ap.add_argument("--exact-dedup", action="store_true",
                    help="StructureMatcher inside collision buckets (needs pymatgen)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    raw = []
    for spec in args.inputs:
        path, src = _parse_input(spec)
        c0 = len(raw)
        raw.extend(load_structures(path, src))
        print(f"  loaded {len(raw) - c0:>8d} from {path} (source={src})")
    print(f"total ingested: {len(raw)}")

    filt = list(
        filter_records(
            raw,
            max_atoms=args.max_atoms,
            e_form_max=args.e_form_max,
            e_hull_max=args.e_hull_max,
        )
    )
    print(f"after filter (<= {args.max_atoms} atoms"
          f"{', e_form<=%s' % args.e_form_max if args.e_form_max is not None else ''}"
          f"{', e_hull<=%s' % args.e_hull_max if args.e_hull_max is not None else ''}"
          f"): {len(filt)}")

    dedup = list(dedup_records(filt, exact=args.exact_dedup))
    print(f"after de-dup: {len(dedup)}")

    if args.exclude:
        excl = set()
        for ep in args.exclude:
            ks = exclusion_keyset(list(load_structures(ep)))
            excl |= ks
        before = len(dedup)
        dedup = apply_exclusion(dedup, excl)
        print(f"after CSP-testset exclusion: {len(dedup)} (-{before - len(dedup)})")

    sampler = DiversitySampler(mode=args.sampler, seed=args.seed)
    print("coverage (pre-sampling):", sampler.coverage_report(dedup))
    selected = sampler.select(dedup, target_size=args.target_size)
    print("coverage (post-sampling):", sampler.coverage_report(selected))

    n = write_manifest(selected, args.out)
    print(f"wrote {n} records -> {args.out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
