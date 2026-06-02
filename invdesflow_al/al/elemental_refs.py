"""Stage-1 lazy elemental reference table for paper-faithful E_form.

E_form_per_atom = E_relaxed/N − Σ_i (n_i / N) · E_elem[Z_i]

Per-element protocol (lazy, on demand):
  1. ASE `bulk(symbol)` builds a sensible elemental unit cell.
  2. CHGNet (the same oracle the main loop uses) relaxes it.
  3. Result cached to JSON. Failures (molecular ground states, OOM,
     CHGNet errors) are *also* cached with status='failed' + reason
     — negative cache, do not retry on the next round.

When any Z in a candidate has no usable ref, `e_form_per_atom` returns
`(None, "no_ref_Z=…")` and the caller falls back to the Stage-0 ΔE score
for that candidate. This is the staged behavior from PROGRESS Entry 14 v2.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..data.representation import Crystal

# index 0 unused; Z=1..100
ATOMIC_SYMBOLS = [""] + [
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
]


class ElementalRefs:
    def __init__(self, cache_path: str | Path, oracle):
        self.cache_path = Path(cache_path)
        self.oracle = oracle
        self.cache: dict[int, dict] = {}
        if self.cache_path.exists():
            try:
                raw = json.loads(self.cache_path.read_text())
                self.cache = {int(k): v for k, v in raw.items()}
            except Exception:
                self.cache = {}

    # ------------------------------------------------------------- public
    def get(self, z: int) -> Optional[dict]:
        return self.cache.get(int(z))

    def ensure(self, z: int) -> dict:
        z = int(z)
        if z in self.cache:
            return self.cache[z]
        entry = self._compute(z)
        self.cache[z] = entry
        self._save()
        return entry

    def e_form_per_atom(self, z_list: list[int], e_relax_per_atom: float
                        ) -> tuple[Optional[float], Optional[str]]:
        n = len(z_list)
        if n == 0:
            return None, "empty z_list"
        if not math.isfinite(e_relax_per_atom):
            return None, "non-finite e_relax"
        cnt = Counter(int(z) for z in z_list)
        # Eager: ensure ALL refs first so they all hit the cache (successes and
        # negative cache for failures), even if one is missing. Avoids the
        # earlier bug where the loop returned on the first failure and never
        # cached the others.
        entries = {z: self.ensure(z) for z in cnt}
        missing = [z for z, e in entries.items()
                   if e.get("status") != "ok" or e.get("E_per_atom") is None]
        if missing:
            return None, f"no_ref_Z={missing}"
        elemental = sum((c / n) * float(entries[z]["E_per_atom"]) for z, c in cnt.items())
        ef = e_relax_per_atom - elemental
        if not math.isfinite(ef):
            return None, "non-finite e_form"
        return float(ef), None

    def coverage(self) -> dict:
        ok = sum(1 for v in self.cache.values() if v.get("status") == "ok")
        failed = len(self.cache) - ok
        return {"total_attempted": len(self.cache), "ok": ok, "failed": failed}

    # ------------------------------------------------------------ internal
    def _build_atoms_for(self, sym: str, z: int):
        """Try ase.bulk first; fall back to explicit cells for elements whose
        ASE defaults need an atomic basis.

        The fallbacks are deliberately small, deterministic, multi-atom cells:
        enough for CHGNet to produce a consistent elemental reference without
        depending on ASE's per-element basis table. They are approximants, not
        a claim of experimental ground-state fidelity."""
        from ase.build import bulk
        try:
            return bulk(sym), "ase.bulk"
        except Exception as bulk_err:
            pass
        from ase import Atoms

        explicit = self._explicit_fallback_atoms(sym, z)
        if explicit is not None:
            atoms, source = explicit
            return atoms, source

        # diatomic / molecular fallback in a large vacuum cell
        diatomic = {
            1:  ("H",  0.74),
            7:  ("N",  1.10),
            8:  ("O",  1.21),
            9:  ("F",  1.42),
            17: ("Cl", 1.99),
            35: ("Br", 2.28),
            53: ("I",  2.66),
        }
        if z in diatomic:
            s, d = diatomic[z]
            atoms = Atoms([s, s], positions=[[6.0, 6.0, 6.0],
                                              [6.0 + d, 6.0, 6.0]],
                          cell=[12.0, 12.0, 12.0], pbc=True)
            return atoms, "diatomic-in-vacuum"
        # nothing we can do
        raise RuntimeError(f"no fallback cell for Z={z} ({sym})")

    def _explicit_fallback_atoms(self, sym: str, z: int):
        from ase import Atoms
        from ase.build import bulk

        # B: alpha-boron-like rhombohedral approximant. ASE supports
        # rhombohedral once an explicit basis is supplied.
        if z == 5:
            atoms = bulk(
                sym,
                crystalstructure="rhombohedral",
                a=5.06,
                alpha=58.06,
                basis=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
            )
            return atoms, "explicit-rhombohedral-B"

        # P: white-phosphorus P4 tetrahedron in a vacuum box.
        if z == 15:
            d = 2.21
            h = math.sqrt(2.0 / 3.0) * d
            r = d / math.sqrt(3.0)
            positions = [
                [6.0, 6.0, 6.0 + h],
                [6.0 + r, 6.0, 6.0],
                [6.0 - 0.5 * r, 6.0 + math.sqrt(3.0) * 0.5 * r, 6.0],
                [6.0 - 0.5 * r, 6.0 - math.sqrt(3.0) * 0.5 * r, 6.0],
            ]
            atoms = Atoms([sym] * 4, positions=positions,
                          cell=[12.0, 12.0, 12.0], pbc=True)
            return atoms, "explicit-P4-in-vacuum"

        # S: cyclo-S8-like puckered ring in a vacuum box.
        if z == 16:
            positions = []
            for i in range(8):
                theta = 2.0 * math.pi * i / 8.0
                positions.append([
                    7.0 + 2.1 * math.cos(theta),
                    7.0 + 2.1 * math.sin(theta),
                    7.0 + (0.45 if i % 2 else -0.45),
                ])
            atoms = Atoms([sym] * 8, positions=positions,
                          cell=[14.0, 14.0, 14.0], pbc=True)
            return atoms, "explicit-S8-in-vacuum"

        # Mn: alpha-Mn is complex; bcc Mn is the standard compact fallback for
        # this gate and avoids ASE's missing default basis.
        if z == 25:
            atoms = bulk(sym, crystalstructure="bcc", a=2.89, cubic=True)
            return atoms, "explicit-bcc-Mn"

        # Ga: alpha-Ga orthorhombic approximant with eight atoms.
        if z == 31:
            cell = [4.51, 4.52, 7.65]
            basis = [
                (0.000, 0.000, 0.000),
                (0.500, 0.500, 0.000),
                (0.000, 0.500, 0.500),
                (0.500, 0.000, 0.500),
                (0.078, 0.152, 0.250),
                (0.578, 0.652, 0.250),
                (0.078, 0.652, 0.750),
                (0.578, 0.152, 0.750),
            ]
            atoms = Atoms([sym] * len(basis), scaled_positions=basis,
                          cell=cell, pbc=True)
            return atoms, "explicit-orthorhombic-Ga"

        # Se/Te: trigonal-chain approximants in hexagonal cells.
        if z in (34, 52):
            a, c = (4.366, 4.954) if z == 34 else (4.456, 5.921)
            basis = [(0.0, 0.0, 0.00), (0.0, 0.0, 0.333), (0.0, 0.0, 0.667)]
            cell = [[a, 0.0, 0.0],
                    [-0.5 * a, math.sqrt(3.0) * 0.5 * a, 0.0],
                    [0.0, 0.0, c]]
            atoms = Atoms([sym] * len(basis), scaled_positions=basis,
                          cell=cell, pbc=True)
            return atoms, f"explicit-trigonal-chain-{sym}"

        return None

    def _compute(self, z: int) -> dict:
        if z < 1 or z >= len(ATOMIC_SYMBOLS):
            return {"E_per_atom": None, "status": "failed",
                    "reason": f"Z={z} out of supported range"}
        sym = ATOMIC_SYMBOLS[z]
        try:
            atoms, source = self._build_atoms_for(sym, z)
            n = len(atoms)
            if n == 0:
                raise RuntimeError("empty cell")
            crystal = Crystal(
                atom_types=torch.full((n,), z, dtype=torch.long),
                frac_coords=torch.tensor(
                    np.asarray(atoms.get_scaled_positions()), dtype=torch.float
                ),
                lattice=torch.tensor(np.asarray(atoms.cell[:]), dtype=torch.float),
            )
            r = self.oracle.relax_one(crystal)
            if r.status != "ok":
                return {"E_per_atom": None, "status": "failed",
                        "reason": (r.reason or "relax_failed")[:200],
                        "symbol": sym, "unit_cell_n": n, "source": source}
            if not math.isfinite(r.energy_per_atom):
                return {"E_per_atom": None, "status": "failed",
                        "reason": "non-finite energy_per_atom",
                        "symbol": sym, "source": source}
            return {"E_per_atom": float(r.energy_per_atom),
                    "status": "ok", "reason": None,
                    "symbol": sym, "unit_cell_n": n,
                    "converged_ml": bool(r.converged_ml),
                    "source": f"{source} + CHGNet relax"}
        except Exception as e:
            return {"E_per_atom": None, "status": "failed",
                    "reason": f"{type(e).__name__}: {str(e)[:200]}",
                    "symbol": sym}

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {str(k): v for k, v in self.cache.items()},
            indent=2, sort_keys=True))
        tmp.replace(self.cache_path)
