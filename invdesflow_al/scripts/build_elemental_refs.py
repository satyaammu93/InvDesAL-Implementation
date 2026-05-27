"""Pre-warm the Stage-1 elemental reference cache.

Iterates a list of Z values (or scans a manifest for the elements it contains)
and computes E_elem_per_atom for each via ASE bulk + CHGNet relax. Cached to
JSON; both successes and failures are kept (negative cache).

Usage:
  # cover everything present in the pretrain manifest
  python -m invdesflow_al.scripts.build_elemental_refs \
      --manifest data_raw/pretrain.jsonl \
      --out data_raw/chgnet_elemental_refs.json --device cuda

  # or an explicit list
  python -m invdesflow_al.scripts.build_elemental_refs \
      --elements 1 6 8 14 23 24 26 28 38 56 \
      --out data_raw/chgnet_elemental_refs.json --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ..al import CHGNetOracle, ElementalRefs


def _elements_from_manifest(manifest: str, cap: int = 50000) -> list[int]:
    from ..data.datasets import load_structures

    seen: set[int] = set()
    for i, r in enumerate(load_structures(manifest)):
        seen.update(int(z) for z in r.z)
        if i + 1 >= cap:
            break
    return sorted(seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_raw/chgnet_elemental_refs.json",
                    help="cache path")
    ap.add_argument("--manifest",
                    help="scan this JSONL for elements to warm")
    ap.add_argument("--elements", type=int, nargs="*",
                    help="explicit Z list (e.g. 1 6 8 14 23 24 26)")
    ap.add_argument("--exclude", type=int, nargs="*", default=[],
                    help="Z values to skip (e.g. banned in the loop)")
    ap.add_argument("--lbfgs-steps", type=int, default=200,
                    help="elemental relaxation usually wants more steps")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.elements:
        zlist = sorted(set(args.elements))
    elif args.manifest:
        zlist = _elements_from_manifest(args.manifest)
        print(f"manifest scan: {len(zlist)} distinct elements")
    else:
        ap.error("provide --elements or --manifest")
    if args.exclude:
        excl = set(args.exclude)
        zlist = [z for z in zlist if z not in excl]
        print(f"after exclude ({excl}): {len(zlist)} elements")

    # the elemental refs use the same CHGNet oracle the loop uses
    oracle = CHGNetOracle(
        cache_path=None,        # don't pollute the per-run cache
        device=args.device,
        steps=args.lbfgs_steps,
    )
    refs = ElementalRefs(args.out, oracle)
    t0 = time.time()
    n_new_ok = n_new_fail = n_cached = 0
    for i, z in enumerate(zlist):
        if refs.get(z) is not None:
            n_cached += 1
            continue
        entry = refs.ensure(z)
        if entry.get("status") == "ok":
            n_new_ok += 1
        else:
            n_new_fail += 1
        print(f"  [{i+1}/{len(zlist)}] Z={z} ({entry.get('symbol','?')}) "
              f"-> {entry.get('status')}  "
              f"E={entry.get('E_per_atom')}  "
              f"reason={entry.get('reason')}", flush=True)

    cov = refs.coverage()
    print(f"\ncache: {args.out}")
    print(f"  attempted total : {cov['total_attempted']}")
    print(f"  ok              : {cov['ok']}")
    print(f"  failed (cached) : {cov['failed']}")
    print(f"  this run: new_ok={n_new_ok} new_fail={n_new_fail} cached={n_cached}")
    print(f"  wallclock {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
