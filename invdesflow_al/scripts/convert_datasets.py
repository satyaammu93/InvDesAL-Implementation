"""Convert raw Alex-MP-20 (HF parquet) and GNoME (GCS zip+csv) into the
dependency-free JSONL the pretraining pipeline consumes.

Alex-MP-20 stores CARTESIAN `positions` + `cell` -> we convert to fractional.
GNoME ships CIFs in by_id.zip + a summary CSV (e_form, decomposition energy
~= e_hull, space-group number) keyed by MaterialId == CIF stem.

Usage:
  python -m invdesflow_al.scripts.convert_datasets alex  \
      data_raw/alex_mp_20/train.parquet  data_raw/alex_mp_20.jsonl
  python -m invdesflow_al.scripts.convert_datasets gnome \
      data_raw/gnome/by_id.zip data_raw/gnome/stable_materials_summary.csv \
      data_raw/gnome.jsonl  --limit 150000
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
import zipfile

import numpy as np


def _sg_number(symbol_or_num) -> int | None:
    try:
        return int(symbol_or_num)
    except (TypeError, ValueError):
        pass
    try:
        from pymatgen.symmetry.groups import SpaceGroup

        return int(SpaceGroup(str(symbol_or_num)).int_number)
    except Exception:
        return None


def convert_alex(parquet_path: str, out: str) -> int:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    n = 0
    t = time.time()
    with open(out, "w") as f:
        for batch in pf.iter_batches(batch_size=4096):
            for r in batch.to_pylist():
                cell = np.asarray(r["cell"], dtype=float)  # rows = basis
                cart = np.asarray(r["positions"], dtype=float)
                try:
                    frac = cart @ np.linalg.inv(cell)
                except np.linalg.LinAlgError:
                    continue
                frac = frac - np.floor(frac)
                rec = {
                    "z": [int(z) for z in r["atomic_numbers"]],
                    "frac": frac.tolist(),
                    "lattice": cell.tolist(),
                    "e_form": None,  # Alex-MP-20 has no formation energy column
                    "e_hull": r.get("energy_above_hull"),
                    "spacegroup": _sg_number(r.get("space_group")),
                    "source": "alex-mp-20",
                }
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                n += 1
            if n % 50000 < 4096:
                print(f"  alex {n} ({time.time()-t:.0f}s)", file=sys.stderr)
    print(f"alex: wrote {n} -> {out} ({time.time()-t:.0f}s)")
    return n


def convert_gnome(zip_path: str, csv_path: str, out: str,
                   limit: int | None = None, seed: int = 0) -> int:
    from pymatgen.core import Structure

    meta: dict[str, dict] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            meta[row["MaterialId"]] = row

    zf = zipfile.ZipFile(zip_path)
    names = [x for x in zf.namelist() if x.lower().endswith(".cif")]
    if limit and limit < len(names):
        random.Random(seed).shuffle(names)
        names = names[:limit]

    n = 0
    t = time.time()
    with open(out, "w") as f:
        for name in names:
            mid = name.split("/")[-1].rsplit(".", 1)[0]
            try:
                s = Structure.from_str(zf.read(name).decode(), fmt="cif")
            except Exception:
                continue
            m = meta.get(mid, {})

            def _flt(key):
                v = m.get(key)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None

            rec = {
                "z": [site.specie.Z for site in s],
                "frac": (s.frac_coords - np.floor(s.frac_coords)).tolist(),
                "lattice": [list(map(float, r)) for r in s.lattice.matrix],
                "e_form": _flt("Formation Energy Per Atom"),
                "e_hull": _flt("Decomposition Energy Per Atom"),
                "spacegroup": _sg_number(m.get("Space Group Number")),
                "source": "gnome",
            }
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            n += 1
            if n % 20000 == 0:
                print(f"  gnome {n}/{len(names)} ({time.time()-t:.0f}s)", file=sys.stderr)
    print(f"gnome: wrote {n} -> {out} ({time.time()-t:.0f}s)")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("alex"); a.add_argument("parquet"); a.add_argument("out")
    g = sub.add_parser("gnome")
    g.add_argument("zip"); g.add_argument("csv"); g.add_argument("out")
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cmd == "alex":
        convert_alex(args.parquet, args.out)
    else:
        convert_gnome(args.zip, args.csv, args.out, args.limit, args.seed)


if __name__ == "__main__":
    main()
