"""Gate 1 (PROGRESS.md Entry 4): is the data sane?

Inspects N records from a JSONL manifest and reports atom-count histogram,
element frequencies, reduced-formula diversity, lattice volume/atom, and
invalid lattices. Prints PASS/FAIL against the Entry-4 hard-fail criteria.

Usage:
  python -m invdesflow_al.scripts.debug_data_sanity \
      --manifest data_raw/pretrain.jsonl --n 1000
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from functools import reduce
from math import gcd

import torch

from ..data.datasets import load_structures


def reduced_formula(z: list[int]) -> str:
    c = Counter(z)
    g = reduce(gcd, c.values()) if c else 1
    return "-".join(f"{e}:{n // g}" for e, n in sorted(c.items()))


def quants(xs):
    s = sorted(xs)
    n = len(s)
    return {q: s[min(n - 1, int(q * n))] for q in (0.0, 0.05, 0.5, 0.95, 1.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_raw/pretrain.jsonl")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    recs = []
    for r in load_structures(args.manifest):
        recs.append(r)
        if len(recs) >= args.n:
            break
    print(f"inspected {len(recs)} records from {args.manifest}\n")

    n_atoms, elements, formulas = [], Counter(), Counter()
    vpa, bad_lat, bad_z, nan_inf, big_entry = [], 0, 0, 0, 0
    for r in recs:
        n_atoms.append(len(r.z))
        elements.update(r.z)
        formulas[reduced_formula(r.z)] += 1
        if any((z < 1 or z > 100) for z in r.z):
            bad_z += 1
        lat = torch.tensor(r.lattice, dtype=torch.float)
        frac = torch.tensor(r.frac, dtype=torch.float)
        if not (torch.isfinite(lat).all() and torch.isfinite(frac).all()):
            nan_inf += 1
            continue
        if lat.abs().max() > 100:
            big_entry += 1
        vol = float(torch.det(lat).abs())
        if vol <= 1e-6 or not math.isfinite(vol):
            bad_lat += 1
            continue
        vpa.append(vol / max(len(r.z), 1))

    uniq_rate = len(formulas) / len(recs)
    top_el, top_el_n = elements.most_common(1)[0]
    top_el_frac = top_el_n / sum(elements.values())
    qa = quants(n_atoms)
    qv = quants(vpa) if vpa else {}

    print(f"atom count   : min/med/max = {qa[0.0]}/{qa[0.5]}/{qa[1.0]}  "
          f"(5%/95% = {qa[0.05]}/{qa[0.95]})")
    print(f"distinct elements: {len(elements)}   top: {elements.most_common(6)}")
    print(f"reduced formulas : {len(formulas)} distinct / {len(recs)} "
          f"-> unique rate {uniq_rate:.3f}")
    print(f"most common element: Z={top_el} at {top_el_frac:.1%}")
    if qv:
        print(f"volume/atom (A^3): min/med/max = "
              f"{qv[0.0]:.1f}/{qv[0.5]:.1f}/{qv[1.0]:.1f}  "
              f"(5%/95% = {qv[0.05]:.1f}/{qv[0.95]:.1f})")
    print(f"invalid: nan/inf={nan_inf}  bad_lattice(det<=0)={bad_lat}  "
          f"bad_Z={bad_z}  |lat entry|>100={big_entry}")

    fails = []
    if nan_inf:
        fails.append(f"{nan_inf} records with NaN/Inf")
    if bad_lat:
        fails.append(f"{bad_lat} non-positive/again-finite lattice volumes")
    if bad_z:
        fails.append(f"{bad_z} records with Z outside 1..100")
    if vpa and (qv[0.05] <= 0 or qv[0.95] > 500):
        fails.append(f"volume/atom out of (0,500] at 5/95% quantiles")
    if uniq_rate < 0.2:
        fails.append(f"reduced-formula unique rate {uniq_rate:.3f} too low")
    if top_el_frac > 0.95:
        fails.append(f"element Z={top_el} dominates {top_el_frac:.1%}")

    print()
    if fails:
        print("GATE 1 FAIL:")
        for f in fails:
            print("  -", f)
    else:
        print("GATE 1 PASS - data is sane")


if __name__ == "__main__":
    main()
