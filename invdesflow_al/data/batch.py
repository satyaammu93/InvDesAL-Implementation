"""Batching of crystals into one disjoint graph (PyG-style, but dependency-free).

Atoms of all crystals are concatenated; `batch` maps each atom to its crystal.
Edges/offsets are recomputed per crystal and shifted into the global index.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .graph import complete_graph, periodic_radius_graph
from .representation import Crystal


@dataclass
class CrystalBatch:
    atom_types: Tensor  # [Ntot] long
    frac_coords: Tensor  # [Ntot, 3]
    lattices: Tensor  # [B, 3, 3]
    num_atoms: Tensor  # [B] long
    batch: Tensor  # [Ntot] long -> crystal id
    edge_index: Tensor  # [2, E] long (global indices, src j -> dst i)
    cell_offset: Tensor  # [E, 3] float

    @property
    def num_graphs(self) -> int:
        return int(self.num_atoms.shape[0])

    def to(self, device) -> "CrystalBatch":
        return CrystalBatch(
            self.atom_types.to(device),
            self.frac_coords.to(device),
            self.lattices.to(device),
            self.num_atoms.to(device),
            self.batch.to(device),
            self.edge_index.to(device),
            self.cell_offset.to(device),
        )


def collate(
    crystals: list[Crystal],
    cutoff: float = 7.0,
    max_neighbors: int = 20,
    mode: str = "complete",
) -> CrystalBatch:
    """mode: "complete" (fully-connected, geometry-independent -- default) or
    "radius" (periodic radius graph). See data/graph.py."""
    atom_types, frac, lattices, num_atoms, batch = [], [], [], [], []
    edge_index, cell_offset = [], []
    node_offset = 0
    for gid, c in enumerate(crystals):
        if mode == "complete":
            ei, off = complete_graph(c.num_atoms)
        else:
            ei, off = periodic_radius_graph(c.frac_coords, c.lattice, cutoff, max_neighbors)
        atom_types.append(c.atom_types)
        frac.append(c.frac_coords)
        lattices.append(c.lattice)
        num_atoms.append(c.num_atoms)
        batch.append(torch.full((c.num_atoms,), gid, dtype=torch.long))
        edge_index.append(ei + node_offset)
        cell_offset.append(off)
        node_offset += c.num_atoms

    return CrystalBatch(
        atom_types=torch.cat(atom_types),
        frac_coords=torch.cat(frac),
        lattices=torch.stack(lattices),
        num_atoms=torch.tensor(num_atoms, dtype=torch.long),
        batch=torch.cat(batch),
        edge_index=torch.cat(edge_index, dim=1),
        cell_offset=torch.cat(cell_offset, dim=0),
    )
