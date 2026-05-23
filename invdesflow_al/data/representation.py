"""Crystal representation M = (A, F, L).

Paper (Methods): a crystal is its unit cell, with three components
  A in R^{n x h}  -- chemical species (one-hot)
  F in R^{3 x N}  -- FRACTIONAL coordinates, anchored to the lattice
  L in R^{3 x 3}  -- lattice basis vectors (rows = l1, l2, l3)

We store F as [N, 3] (row per atom) for convenience; lattice rows are the
basis vectors so that  cartesian = frac @ L.

`w(.)` in Algorithms 1 & 2 is the periodic wrap of fractional coordinates
into [0, 1); implemented here as `wrap_frac`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

MAX_ATOMIC_NUM = 100  # one-hot over atomic numbers 1..100


def wrap_frac(frac: Tensor) -> Tensor:
    """w(.) : map fractional coords onto the torus [0, 1).

    `frac - floor(frac)` can round up to exactly 1.0 in float32 for tiny
    negative inputs; on the torus 1.0 == 0.0, so fold it back.
    """
    f = frac - torch.floor(frac)
    return torch.where(f >= 1.0, torch.zeros_like(f), f)


def frac_to_cart(frac: Tensor, lattice: Tensor) -> Tensor:
    """frac [..., 3], lattice [..., 3, 3] (rows = basis) -> cartesian [..., 3]."""
    return torch.einsum("...i,...ij->...j", frac, lattice)


def cart_to_frac(cart: Tensor, lattice: Tensor) -> Tensor:
    inv = torch.linalg.inv(lattice)
    return torch.einsum("...i,...ij->...j", cart, inv)


def atomic_numbers_to_onehot(z: Tensor, num_classes: int = MAX_ATOMIC_NUM) -> Tensor:
    """z: long tensor of atomic numbers (1..num_classes) -> one-hot [N, num_classes]."""
    return torch.nn.functional.one_hot((z - 1).long(), num_classes).float()


def onehot_to_atomic_numbers(onehot: Tensor) -> Tensor:
    return onehot.argmax(dim=-1).long() + 1


@dataclass
class Crystal:
    """A single crystal. atom_types are atomic numbers (long, 1..100)."""

    atom_types: Tensor  # [N]    long
    frac_coords: Tensor  # [N, 3] float, in [0, 1)
    lattice: Tensor  # [3, 3] float (rows = basis vectors)

    def __post_init__(self) -> None:
        self.atom_types = self.atom_types.long()
        self.frac_coords = wrap_frac(self.frac_coords.float())
        self.lattice = self.lattice.float()
        assert self.lattice.shape == (3, 3)
        assert self.frac_coords.shape == (self.num_atoms, 3)

    @property
    def num_atoms(self) -> int:
        return int(self.atom_types.shape[0])

    @property
    def cart_coords(self) -> Tensor:
        return frac_to_cart(self.frac_coords, self.lattice)

    @classmethod
    def from_pymatgen(cls, structure) -> "Crystal":  # pragma: no cover - optional dep
        import numpy as np

        return cls(
            atom_types=torch.tensor([s.specie.Z for s in structure]),
            frac_coords=torch.tensor(np.array(structure.frac_coords), dtype=torch.float),
            lattice=torch.tensor(np.array(structure.lattice.matrix), dtype=torch.float),
        )

    @classmethod
    def random(cls, num_atoms: int, generator: torch.Generator | None = None) -> "Crystal":
        """A random (physically meaningless) crystal -- for smoke tests only."""
        g = generator
        lat = torch.eye(3) * 4.0 + torch.randn(3, 3, generator=g) * 0.3
        return cls(
            atom_types=torch.randint(1, MAX_ATOMIC_NUM + 1, (num_atoms,), generator=g),
            frac_coords=torch.rand(num_atoms, 3, generator=g),
            lattice=lat,
        )
