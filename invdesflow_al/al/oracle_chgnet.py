"""Stage-0 CHGNet relaxation oracle (Entry 14 v2).

Per candidate, runs CHGNet's StructOptimizer (LBFGS via ASE) and records:
  energy_per_atom, max_force,
  converged_ml     (max_force < ml_thresh, default 0.05 eV/A — round-0 gate),
  converged_strict (max_force < strict_thresh, default 1e-4 — paper target,
                    recorded only),
  delta_e          (E_initial/atom - E_relaxed/atom),
  volume_change,
  min_distance_post (min periodic interatomic distance after relax),
  spacegroup_pre / spacegroup_post  (pymatgen, symprec=0.1),
  relaxed_frac, relaxed_lattice.

Failures (CHGNet divergence, NaN forces, OOM, pymatgen parse errors) are
caught per candidate and recorded as status="failed" with a short reason --
the whole AL job must not crash because of one bad candidate.

A persistent JSON cache keyed by
  sha256(reduced_formula | spacegroup | quantized_frac | quantized_lattice)
skips re-relaxing identical structures across rounds (both successes and
failures are cached — failures with negative-cache semantics).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Novelty key helpers (shared between oracle and the AL script)
# --------------------------------------------------------------------------- #
def _reduced_formula_key(z) -> str:
    """Composition key in the same shape as data.datasets._reduced_formula_key."""
    cnt = Counter(int(x) for x in z)
    g = reduce(gcd, cnt.values()) if cnt else 1
    return "-".join(f"{e}:{n // g}" for e, n in sorted(cnt.items()))


def _spacegroup_safe(structure, symprec: float = 0.1) -> Optional[int]:
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        return int(SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number())
    except Exception:
        return None


def novelty_key(crystal_or_structure, *, symprec: float = 0.1) -> tuple[str, Optional[int]]:
    """(reduced_formula_key, spacegroup) — matches build_manifest dedup shape."""
    if hasattr(crystal_or_structure, "atom_types"):  # our Crystal
        z = crystal_or_structure.atom_types.tolist()
        s = _crystal_to_pymatgen(crystal_or_structure)
    else:                                            # pymatgen Structure
        z = [site.specie.Z for site in crystal_or_structure]
        s = crystal_or_structure
    return (_reduced_formula_key(z), _spacegroup_safe(s, symprec))


def manifest_novelty_set(manifest_path: str, symprec: float = 0.1) -> set[tuple]:
    """Set of novelty keys for everything in a JSONL manifest.

    Records that already carry `spacegroup` use that value directly (cheap);
    records without a spacegroup fall back to pymatgen analysis if available.
    """
    from ..data.datasets import load_structures

    keys: set[tuple] = set()
    for r in load_structures(manifest_path):
        cf = _reduced_formula_key(r.z)
        sg = r.spacegroup
        if sg is None:
            try:
                s = _crystal_to_pymatgen(r.to_crystal())
                sg = _spacegroup_safe(s, symprec)
            except Exception:
                sg = None
        keys.add((cf, sg))
    return keys


# --------------------------------------------------------------------------- #
# Crystal <-> pymatgen Structure conversions (kept local to avoid coupling)
# --------------------------------------------------------------------------- #
def _crystal_to_pymatgen(c):
    from pymatgen.core import Lattice, Structure

    lat = c.lattice.detach().cpu().numpy() if hasattr(c.lattice, "detach") else np.asarray(c.lattice)
    frac = c.frac_coords.detach().cpu().numpy() if hasattr(c.frac_coords, "detach") else np.asarray(c.frac_coords)
    z = c.atom_types.tolist() if hasattr(c.atom_types, "tolist") else list(c.atom_types)
    return Structure(Lattice(lat), species=z, coords=frac, coords_are_cartesian=False)


def _pymatgen_min_periodic_distance(s) -> float:
    if len(s) < 2:
        return float("inf")
    try:
        dm = s.distance_matrix
        np.fill_diagonal(dm, np.inf)
        d = float(dm.min())
        return d if math.isfinite(d) else 0.0
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# Cache key
# --------------------------------------------------------------------------- #
def _quantize_floats(arr, ndigits: int = 4) -> tuple:
    a = np.asarray(arr).flatten()
    return tuple(round(float(x), ndigits) for x in a)


def relax_cache_key(crystal, symprec: float = 0.1) -> str:
    """sha256(reduced_formula | spacegroup | quantized_frac | quantized_lattice).

    spacegroup may be None (non-symmetrizable / pymatgen error) — treated as 0.
    Quantization to 4 decimal places makes the key robust to FP roundoff but
    sensitive to real structural changes.
    """
    z = crystal.atom_types.tolist() if hasattr(crystal.atom_types, "tolist") else list(crystal.atom_types)
    cf = _reduced_formula_key(z)
    try:
        s = _crystal_to_pymatgen(crystal)
        sg = _spacegroup_safe(s, symprec) or 0
    except Exception:
        sg = 0
    payload = "|".join([
        cf,
        str(sg),
        ",".join(map(str, _quantize_floats(crystal.frac_coords))),
        ",".join(map(str, _quantize_floats(crystal.lattice))),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Per-candidate result
# --------------------------------------------------------------------------- #
@dataclass
class RelaxResult:
    """Entry-14 v2 schema (Stage 0 fields)."""

    energy_per_atom: float
    max_force: float
    converged_ml: bool
    converged_strict: bool
    delta_e: float
    volume_change: float
    min_distance_post: float
    spacegroup_pre: Optional[int]
    spacegroup_post: Optional[int]
    relaxed_frac: list
    relaxed_lattice: list
    status: str
    reason: Optional[str]
    cached: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------- #
# CHGNet oracle
# --------------------------------------------------------------------------- #
class CHGNetOracle:
    """Stateful CHGNet relaxer with persistent JSON cache.

    One instance owns one StructOptimizer (CHGNet model loaded once). Call
    `relax_one(crystal)` per candidate; per-candidate errors are caught and
    returned as `status="failed"` results.
    """

    def __init__(
        self,
        cache_path: Optional[str] = None,
        device: str = "cuda",
        ml_thresh: float = 0.05,
        strict_thresh: float = 1e-4,
        fmax: Optional[float] = None,
        steps: int = 100,
        symprec: float = 0.1,
        relax_cell: bool = True,
    ):
        from chgnet.model import StructOptimizer

        # CHGNet 0.3.8 uses keyword `use_device`.
        self.opt = StructOptimizer(use_device=device)
        self.ml_thresh = float(ml_thresh)
        self.strict_thresh = float(strict_thresh)
        # ASE LBFGS fmax defaults to the ML threshold so we don't waste steps
        # chasing 1e-4 while round-0 gates on 0.05.
        self.fmax = float(fmax) if fmax is not None else float(ml_thresh)
        self.steps = int(steps)
        self.symprec = float(symprec)
        self.relax_cell = bool(relax_cell)
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text())
            except Exception:
                self.cache = {}

    # ------------------------------------------------------------------ public
    def relax_one(self, crystal, verbose: bool = False) -> RelaxResult:
        key = relax_cache_key(crystal, self.symprec)
        if key in self.cache:
            d = dict(self.cache[key])
            d["cached"] = True
            return RelaxResult(**d)
        try:
            s_pre = _crystal_to_pymatgen(crystal)
            v_pre = float(s_pre.lattice.volume)
            sg_pre = _spacegroup_safe(s_pre, self.symprec)
            res = self.opt.relax(
                s_pre,
                fmax=self.fmax,
                steps=self.steps,
                relax_cell=self.relax_cell,
                verbose=verbose,
            )
            s_post = res["final_structure"]
            traj = res["trajectory"]
            energies = traj.energies                # list of total energies
            forces_last = np.asarray(traj.forces[-1])
            n_post = max(len(s_post), 1)
            e_init = float(energies[0]) / max(len(s_pre), 1)
            e_final = float(energies[-1]) / n_post
            max_f = float(np.linalg.norm(forces_last, axis=-1).max())
            if not (math.isfinite(max_f) and math.isfinite(e_final) and math.isfinite(e_init)):
                raise RuntimeError("non-finite energy/force from CHGNet")

            converged_ml = bool(max_f < self.ml_thresh)
            converged_strict = bool(max_f < self.strict_thresh)
            delta_e = float(e_init - e_final)
            v_post = float(s_post.lattice.volume)
            vol_change = (v_post - v_pre) / max(v_pre, 1e-9)
            min_d_post = _pymatgen_min_periodic_distance(s_post)
            sg_post = _spacegroup_safe(s_post, self.symprec)

            out = {
                "energy_per_atom": e_final,
                "max_force": max_f,
                "converged_ml": converged_ml,
                "converged_strict": converged_strict,
                "delta_e": delta_e,
                "volume_change": float(vol_change),
                "min_distance_post": float(min_d_post),
                "spacegroup_pre": sg_pre,
                "spacegroup_post": sg_post,
                "relaxed_frac": [[float(x) for x in row] for row in s_post.frac_coords],
                "relaxed_lattice": [[float(x) for x in row] for row in s_post.lattice.matrix],
                "status": "ok",
                "reason": None,
            }
        except Exception as e:
            out = {
                "energy_per_atom": float("nan"),
                "max_force": float("nan"),
                "converged_ml": False,
                "converged_strict": False,
                "delta_e": float("nan"),
                "volume_change": float("nan"),
                "min_distance_post": float("nan"),
                "spacegroup_pre": None,
                "spacegroup_post": None,
                "relaxed_frac": [],
                "relaxed_lattice": [],
                "status": "failed",
                "reason": f"{type(e).__name__}: {str(e)[:240]}",
            }
        self.cache[key] = out
        return RelaxResult(**out, cached=False)

    # --- bulk helper with periodic flush so a long run can't lose progress
    def relax_many(self, crystals: Iterable, flush_every: int = 25):
        for i, c in enumerate(crystals):
            yield i, self.relax_one(c)
            if self.cache_path and (i + 1) % max(flush_every, 1) == 0:
                self.flush()

    def flush(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache))
            tmp.replace(self.cache_path)
