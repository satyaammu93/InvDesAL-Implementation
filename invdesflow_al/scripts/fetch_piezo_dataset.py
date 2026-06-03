"""Fetch piezoelectric tensor dataset for Plan C training.

First pass: MatBench's `piezoelectric_tensor` (941 entries, de Jong et al.
Sci. Data 2015). Same underlying DFT-PBE data MP serves through its API,
just a smaller frozen snapshot. Output JSONL schema is portable: a future
MP-API pull (~3000 entries) writes the same fields and trains the same
regressor without code changes.

Output schema (one JSON object per line):
    z              list[int]     atomic numbers
    frac           list[[3]]     fractional coords (N x 3)
    lattice        list[[3]]     row-major 3x3 lattice (rows are basis vectors)
    target         float         regression target (default: eij_max, C/m^2)
    target_name    str           "eij_max" (so MP can drop in same column)
    source         str           "matbench:piezoelectric_tensor" | "mp:api" | ...
    material_id    str           mp-... id when available
    formula        str           reduced formula
    spacegroup     int           space group number
    point_group    str           e.g. "4mm" (for stratifying piezo families)
    n_sites        int           number of atoms in the unit cell
    piezo_tensor   list[[6]] | None   3x6 e_ij Voigt tensor (bonus, optional)

Usage:
    python -m invdesflow_al.scripts.fetch_piezo_dataset \
        --out data_raw/mp_piezo.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path


def _quantiles(xs):
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    return {
        "n": n,
        "min": round(s[0], 4),
        "p5": round(s[max(0, n // 20)], 4),
        "median": round(s[n // 2], 4),
        "p95": round(s[min(n - 1, 19 * n // 20)], 4),
        "max": round(s[-1], 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_raw/mp_piezo.jsonl",
                    help="JSONL output path (one record per line)")
    ap.add_argument("--source-tag", default="matbench:piezoelectric_tensor",
                    help="value written into the `source` field of each record")
    ap.add_argument("--target-col", default="eij_max",
                    help="dataframe column used as the regression target. "
                         "default `eij_max` (scalar piezo modulus, C/m^2).")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    from matminer.datasets import load_dataset

    print(f"loading matminer:piezoelectric_tensor ...", flush=True)
    t0 = time.time()
    df = load_dataset("piezoelectric_tensor")
    print(f"  rows={len(df)}  cols={list(df.columns)}", flush=True)
    print(f"  load time {time.time()-t0:.1f}s", flush=True)

    if args.target_col not in df.columns:
        print(f"ERROR: target column {args.target_col!r} not in dataset")
        sys.exit(1)

    n_written = n_skipped = 0
    targets: list[float] = []
    spacegroups: list[int] = []
    point_groups = Counter()
    n_sites_hist: list[int] = []
    element_hist = Counter()

    with open(out, "w") as f:
        for i, row in df.iterrows():
            try:
                s = row["structure"]
                z = [int(site.specie.Z) for site in s.sites]
                frac = [list(map(float, site.frac_coords)) for site in s.sites]
                lattice = [list(map(float, v)) for v in s.lattice.matrix]
                target = float(row[args.target_col])
                # piezoelectric tensor: dataset stores it as a numpy ndarray
                # (3 x 6 Voigt). Convert to nested list of floats; None if missing.
                pt = row.get("piezoelectric_tensor", None)
                if pt is not None:
                    try:
                        piezo_tensor = [[float(x) for x in row_] for row_ in pt]
                    except Exception:
                        piezo_tensor = None
                else:
                    piezo_tensor = None
                rec = {
                    "z": z,
                    "frac": frac,
                    "lattice": lattice,
                    "target": target,
                    "target_name": args.target_col,
                    "source": args.source_tag,
                    "material_id": str(row.get("material_id", "")),
                    "formula": str(row.get("formula", "")),
                    "spacegroup": int(row.get("space_group", 0) or 0),
                    "point_group": str(row.get("point_group", "")),
                    "n_sites": int(row.get("nsites", len(z)) or len(z)),
                    "piezo_tensor": piezo_tensor,
                }
                f.write(json.dumps(rec) + "\n")
                n_written += 1
                targets.append(target)
                spacegroups.append(rec["spacegroup"])
                point_groups[rec["point_group"]] += 1
                n_sites_hist.append(rec["n_sites"])
                for zz in z:
                    element_hist[zz] += 1
            except Exception as e:
                n_skipped += 1
                print(f"  skip row {i}: {type(e).__name__} {e}", flush=True)

    print()
    print(f"wrote -> {out}")
    print(f"  written : {n_written}")
    print(f"  skipped : {n_skipped}")
    print(f"  target  ({args.target_col}, C/m^2): {_quantiles(targets)}")
    print(f"  n_sites : {_quantiles(n_sites_hist)}")
    pg_top = sorted(point_groups.items(), key=lambda x: -x[1])[:10]
    print(f"  point-group top-10: {pg_top}")
    el_top = sorted(element_hist.items(), key=lambda x: -x[1])[:15]
    print(f"  element top-15 (Z, count): {el_top}")


if __name__ == "__main__":
    main()
