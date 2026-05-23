"""Dataset ingestion for pretraining (REBUILD_PLAN.md sec. 6).

Pretraining corpus = Alex-MP-20 (607,683, ref 2) + GNoME 381,000 (ref 1),
diversity-sampled (paper, Methods: "diversity sampling strategy to cover
different regions of the inorganic materials distribution").

This module only *ingests + filters + de-duplicates*. Diversity sampling lives
in `diversity.py`; torch wrapping in `torch_dataset.py`.

Supported inputs (auto-detected by extension):
  * .jsonl / .json   -- one self-describing record per line (NO heavy deps).
                        This is also the cached-manifest format we emit.
  * directory        -- CIF / POSCAR / *.vasp / *.json structures (needs
                        pymatgen), optionally + a sibling metadata .csv keyed
                        by filename (columns: e_form_per_atom, e_hull, ...).
  * .db / .traj      -- ASE database/trajectory (needs ase).

Record schema (the JSONL line / cached manifest):
  {"z":[int...], "frac":[[f,f,f]...], "lattice":[[3x3]],
   "e_form": float|null, "e_hull": float|null,
   "spacegroup": int|null, "source": str}
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator

import torch

from .representation import Crystal


@dataclass
class StructureRecord:
    z: list[int]
    frac: list[list[float]]
    lattice: list[list[float]]
    e_form: float | None = None  # eV/atom
    e_hull: float | None = None  # eV/atom (energy above hull)
    spacegroup: int | None = None
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def num_atoms(self) -> int:
        return len(self.z)

    def to_crystal(self) -> Crystal:
        return Crystal(
            atom_types=torch.tensor(self.z, dtype=torch.long),
            frac_coords=torch.tensor(self.frac, dtype=torch.float),
            lattice=torch.tensor(self.lattice, dtype=torch.float),
        )

    @classmethod
    def from_crystal(cls, c: Crystal, **kw) -> "StructureRecord":
        return cls(
            z=c.atom_types.tolist(),
            frac=c.frac_coords.tolist(),
            lattice=c.lattice.tolist(),
            **kw,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def _tag(r: StructureRecord, src: str | None) -> StructureRecord:
    """JSONL is self-describing; only fill `source` if the record omits it."""
    if src and not r.source:
        r.source = src
    return r


def _load_jsonl(path: str, src: str | None = None) -> Iterator[StructureRecord]:
    with open(path) as f:
        if path.endswith(".json"):
            data = json.load(f)
            rows = data if isinstance(data, list) else data.get("records", [])
            for r in rows:
                yield _tag(StructureRecord(**r), src)
            return
        for line in f:
            line = line.strip()
            if line:
                yield _tag(StructureRecord(**json.loads(line)), src)


def _spacegroup_of(structure, symprec: float = 0.1) -> int | None:
    try:  # pragma: no cover - optional dep
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        return int(SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number())
    except Exception:
        return None


def _load_dir(path: str, source: str) -> Iterator[StructureRecord]:
    try:  # pragma: no cover - optional dep
        from pymatgen.core import Structure
    except ImportError as e:
        raise ImportError(
            "Reading a directory of CIF/POSCAR needs pymatgen "
            "(`pip install pymatgen`). Use the JSONL format to stay dep-free."
        ) from e

    # optional metadata sidecar: <dir>.csv or <dir>/metadata.csv keyed by 'file'
    meta: dict[str, dict] = {}
    for cand in (path.rstrip("/") + ".csv", os.path.join(path, "metadata.csv")):
        if os.path.isfile(cand):
            with open(cand) as fh:
                for row in csv.DictReader(fh):
                    meta[row.get("file", "")] = row
            break

    exts = (".cif", ".vasp", ".poscar", "poscar", "contcar", ".json")
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            if not fn.lower().endswith(exts) and "poscar" not in fn.lower():
                continue
            fp = os.path.join(root, fn)
            try:
                s = Structure.from_file(fp)
            except Exception:
                continue
            m = meta.get(fn, {})

            def _f(key):
                v = m.get(key)
                return float(v) if v not in (None, "", "nan") else None

            yield StructureRecord(
                z=[site.specie.Z for site in s],
                frac=[list(map(float, fc)) for fc in s.frac_coords],
                lattice=[list(map(float, row)) for row in s.lattice.matrix],
                e_form=_f("e_form_per_atom"),
                e_hull=_f("e_hull"),
                spacegroup=_spacegroup_of(s),
                source=source,
                meta={"file": fn},
            )


def _load_ase(path: str, source: str) -> Iterator[StructureRecord]:
    try:  # pragma: no cover - optional dep
        from ase.io import iread
    except ImportError as e:
        raise ImportError("ASE input needs `pip install ase`.") from e
    for atoms in iread(path):
        yield StructureRecord(
            z=list(map(int, atoms.get_atomic_numbers())),
            frac=[list(map(float, fc)) for fc in atoms.get_scaled_positions()],
            lattice=[list(map(float, row)) for row in atoms.cell[:]],
            source=source,
        )


def load_structures(path: str, source: str | None = None) -> Iterator[StructureRecord]:
    """Auto-dispatch on `path`. `source` tags provenance (default: basename)."""
    src = source or os.path.splitext(os.path.basename(path.rstrip("/")))[0]
    if os.path.isdir(path):
        yield from _load_dir(path, src)
    elif path.endswith((".jsonl", ".json")):
        yield from _load_jsonl(path, src)
    elif path.endswith((".db", ".traj", ".xyz", ".extxyz")):
        yield from _load_ase(path, src)
    else:
        raise ValueError(f"unrecognized dataset path: {path}")


# --------------------------------------------------------------------------- #
# filters + de-duplication
# --------------------------------------------------------------------------- #
def filter_records(
    records: Iterable[StructureRecord],
    max_atoms: int = 20,            # Table S.2
    e_form_max: float | None = None,  # e.g. -0.5 for low-formation fine-tune (Fig S.7a)
    e_hull_max: float | None = None,  # e.g. 0.05 for low-Ehull (Fig S.7b)
    elements: set[int] | None = None,
) -> Iterator[StructureRecord]:
    for r in records:
        if r.num_atoms == 0 or r.num_atoms > max_atoms:
            continue
        if e_form_max is not None and (r.e_form is None or r.e_form > e_form_max):
            continue
        if e_hull_max is not None and (r.e_hull is None or r.e_hull > e_hull_max):
            continue
        if elements is not None and not set(r.z).issubset(elements):
            continue
        yield r


def _reduced_formula_key(z: list[int]) -> str:
    from collections import Counter
    from math import gcd
    from functools import reduce

    cnt = Counter(z)
    g = reduce(gcd, cnt.values())
    return "_".join(f"{el}:{n // g}" for el, n in sorted(cnt.items()))


def dedup_records(
    records: Iterable[StructureRecord],
    exact: bool = False,
    symprec: float = 0.1,           # Table S.2 "tolerance for structure matching"
) -> Iterator[StructureRecord]:
    """Cheap de-dup by (reduced composition, spacegroup). With `exact=True`
    and pymatgen present, run StructureMatcher inside each collision bucket."""
    seen: dict[tuple, list[StructureRecord]] = {}
    matcher = None
    if exact:
        try:  # pragma: no cover - optional dep
            from pymatgen.analysis.structure_matcher import StructureMatcher

            matcher = StructureMatcher(ltol=symprec, stol=symprec, angle_tol=5)
        except ImportError:
            matcher = None

    for r in records:
        key = (_reduced_formula_key(r.z), r.spacegroup)
        bucket = seen.setdefault(key, [])
        if not bucket:
            bucket.append(r)
            yield r
            continue
        if matcher is None:
            continue  # composition+sg collision -> treat as duplicate
        from pymatgen.core import Lattice, Structure  # pragma: no cover

        s_new = Structure(Lattice(r.lattice), r.z, r.frac)
        dup = any(
            matcher.fit(s_new, Structure(Lattice(b.lattice), b.z, b.frac))
            for b in bucket
        )
        if not dup:
            bucket.append(r)
            yield r


def compute_lattice_stats(records: Iterable[StructureRecord], max_n: int = 8000):
    """Per-entry mean/std of the 3x3 lattice over up to `max_n` records.

    Used to normalize the lattice channel to a sane O(1) scale before
    diffusion (statistical normalization -- invertible from these stats with
    no ground-truth volume needed). Returns (mean[3,3], std[3,3]).
    """
    mats = []
    for i, r in enumerate(records):
        mats.append(torch.tensor(r.lattice, dtype=torch.float))
        if i + 1 >= max_n:
            break
    M = torch.stack(mats)
    return M.mean(0), M.std(0).clamp_min(1e-2)


def write_manifest(records: Iterable[StructureRecord], path: str) -> int:
    n = 0
    with open(path, "w") as f:
        for r in records:
            f.write(r.to_json() + "\n")
            n += 1
    return n
