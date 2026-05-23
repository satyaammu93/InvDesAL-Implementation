"""Memory-safe streaming manifest builder (for low-RAM machines).

Two passes over the JSONL inputs so we never hold full geometry in RAM:
  pass 1: parse each line, apply filters + (reduced-formula, spacegroup)
          de-dup, keep only a lightweight (file_idx, line_idx, bucket_key)
  diversity-select indices (round-robin across buckets -> coverage)
  pass 2: re-stream the files, copy the selected lines into the manifest

Same filtering / diversity semantics as build_pretrain_dataset.py, but O(#kept
keys) memory instead of O(full records). Used by the overnight orchestrator.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from functools import reduce
from math import gcd


def _keys(z: list[int], sg):
    c = Counter(z)
    g = reduce(gcd, c.values()) if c else 1
    reduced = "-".join(f"{e}:{n // g}" for e, n in sorted(c.items()))
    sg = sg if sg is not None else -1
    anon = tuple(sorted(n // g for n in c.values()))
    return reduced, (sg, anon, len(c))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--max-atoms", type=int, default=20)
    ap.add_argument("--e-form-max", type=float, default=None)
    ap.add_argument("--e-hull-max", type=float, default=None)
    ap.add_argument("--target-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    entries: list[tuple[int, int]] = []     # (file_idx, line_idx)
    bucket_of: list[tuple] = []             # parallel: bucket key
    buckets: dict[tuple, list[int]] = defaultdict(list)
    seen: set[tuple] = set()
    n_seen = n_kept = 0

    for fi, path in enumerate(args.inputs):
        with open(path) as fh:
            for li, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                n_seen += 1
                r = json.loads(line)
                z = r["z"]
                na = len(z)
                if na == 0 or na > args.max_atoms:
                    continue
                if args.e_form_max is not None and (
                    r.get("e_form") is None or r["e_form"] > args.e_form_max
                ):
                    continue
                if args.e_hull_max is not None and (
                    r.get("e_hull") is None or r["e_hull"] > args.e_hull_max
                ):
                    continue
                reduced, bkey = _keys(z, r.get("spacegroup"))
                dkey = (reduced, bkey[0])
                if dkey in seen:
                    continue
                seen.add(dkey)
                idx = len(entries)
                entries.append((fi, li))
                bucket_of.append(bkey)
                buckets[bkey].append(idx)
                n_kept += 1
        print(f"  scanned {path}: seen={n_seen} kept={n_kept} "
              f"({time.time()-t0:.0f}s)")

    target = min(args.target_size or n_kept, n_kept)
    rng = random.Random(args.seed)
    order = list(buckets)
    rng.shuffle(order)
    pools = {k: rng.sample(v, len(v)) for k, v in buckets.items()}
    chosen: set[int] = set()
    while len(chosen) < target:
        progressed = False
        for k in order:
            if pools[k]:
                chosen.add(pools[k].pop())
                progressed = True
                if len(chosen) >= target:
                    break
        if not progressed:
            break

    # group selected line indices per file for pass 2
    per_file: dict[int, set[int]] = defaultdict(set)
    for idx in chosen:
        fi, li = entries[idx]
        per_file[fi].add(li)

    n_written = 0
    with open(args.out, "w") as out:
        for fi, path in enumerate(args.inputs):
            want = per_file.get(fi, set())
            if not want:
                continue
            with open(path) as fh:
                for li, line in enumerate(fh):
                    if li in want:
                        out.write(line if line.endswith("\n") else line + "\n")
                        n_written += 1

    n_sg = len({b[0] for b in bucket_of} - {-1})
    print(f"buckets={len(buckets)} spacegroups={n_sg} kept={n_kept} "
          f"-> selected={n_written} -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
