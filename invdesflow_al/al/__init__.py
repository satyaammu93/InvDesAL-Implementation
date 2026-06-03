"""Active-learning oracles and helpers (Entry 14, staged).

Stage 0 — CHGNet relax + delta_e score; no elemental refs.
Stage 1 — CHGNet + lazy elemental refs -> E_form (paper Eq. 1).
Stage 2 — CHGNet + FormEGNN committee.
Stage 3 — piezoelectric / symmetry oracle.
"""
from .oracle_chgnet import (
    CHGNetOracle,
    RelaxResult,
    novelty_key,
    manifest_novelty_set,
    relax_cache_key,
)
from .elemental_refs import ElementalRefs, ATOMIC_SYMBOLS
from .oracle_piezo import PiezoOracle

__all__ = [
    "CHGNetOracle",
    "RelaxResult",
    "novelty_key",
    "manifest_novelty_set",
    "relax_cache_key",
    "ElementalRefs",
    "ATOMIC_SYMBOLS",
    "PiezoOracle",
]
