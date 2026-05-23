"""Periodic graph construction for the EGNN denoiser.

Table S.2 specifies "CrystalNN method with lattice scaling", neighborhood
cutoff 7.0 A, max 20 neighbors/atom. CrystalNN itself is geometric and has no
fixed cutoff; the 7.0 A / 20-neighbor numbers describe the radius graph that
EGNN message passing actually runs on. We therefore reconstruct a periodic
radius graph with those limits (this is the standard crystal-graph choice for
EGNN/CSPNet-style denoisers) and expose CrystalNN as an optional alternative.

Returned edges carry the integer lattice translation `cell_offset` of the
source image, so the network can reconstruct periodic relative vectors:
    delta_frac = frac[j] + cell_offset - frac[i]
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _replications(lattice: Tensor, cutoff: float) -> tuple[int, int, int]:
    """How many cells to replicate per axis so every neighbor < cutoff is seen."""
    # interplanar spacing per axis = volume / area of the opposite face
    vol = torch.det(lattice).abs().clamp_min(1e-6)
    a, b, c = lattice[0], lattice[1], lattice[2]
    d_a = vol / torch.linalg.cross(b, c).norm().clamp_min(1e-6)
    d_b = vol / torch.linalg.cross(c, a).norm().clamp_min(1e-6)
    d_c = vol / torch.linalg.cross(a, b).norm().clamp_min(1e-6)
    return (
        int(math.ceil(cutoff / float(d_a))),
        int(math.ceil(cutoff / float(d_b))),
        int(math.ceil(cutoff / float(d_c))),
    )


def periodic_radius_graph(
    frac_coords: Tensor,
    lattice: Tensor,
    cutoff: float = 7.0,
    max_neighbors: int = 20,
) -> tuple[Tensor, Tensor]:
    """Build the periodic radius graph for a single crystal.

    Args:
        frac_coords: [N, 3] in [0, 1)
        lattice:     [3, 3] rows = basis vectors
        cutoff:      A (Table S.2: 7.0)
        max_neighbors: keep the nearest `max_neighbors` per atom (Table S.2: 20)

    Returns:
        edge_index:  [2, E] long, rows = (src j, dst i)  (message j -> i)
        cell_offset: [E, 3] float, integer lattice translation applied to src j
    """
    n = frac_coords.shape[0]
    na, nb, nc = _replications(lattice, cutoff)
    rng = [
        torch.arange(-na, na + 1),
        torch.arange(-nb, nb + 1),
        torch.arange(-nc, nc + 1),
    ]
    offsets = torch.cartesian_prod(*rng).float().to(frac_coords)  # [O, 3]

    # cartesian positions of every periodic image of every atom
    cart = frac_coords @ lattice  # [N, 3]
    img_frac = frac_coords[None, :, :] + offsets[:, None, :]  # [O, N, 3]
    img_cart = img_frac @ lattice  # [O, N, 3]

    diff = img_cart[None, :, :, :] - cart[:, None, None, :]  # [N(i), O, N(j), 3]
    dist = diff.norm(dim=-1)  # [N, O, N]

    # exclude self image (i == j and offset == 0)
    zero_off = (offsets.abs().sum(-1) == 0).nonzero(as_tuple=True)[0]
    if zero_off.numel():
        o0 = int(zero_off.item())
        idx = torch.arange(n)
        dist[idx, o0, idx] = float("inf")

    src_list, dst_list, off_list = [], [], []
    for i in range(n):
        d_i = dist[i].reshape(-1)  # [O*N]
        mask = d_i < cutoff
        if not mask.any():
            continue
        cand = mask.nonzero(as_tuple=True)[0]
        if cand.numel() > max_neighbors:
            topk = torch.topk(d_i[cand], max_neighbors, largest=False).indices
            cand = cand[topk]
        o_idx = cand // n
        j_idx = cand % n
        src_list.append(j_idx)
        dst_list.append(torch.full_like(j_idx, i))
        off_list.append(offsets[o_idx])

    if not src_list:  # degenerate (e.g. single atom, tiny cell)
        empty = torch.zeros(2, 0, dtype=torch.long, device=frac_coords.device)
        return empty, torch.zeros(0, 3, device=frac_coords.device)

    edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
    cell_offset = torch.cat(off_list, dim=0)
    return edge_index.long(), cell_offset


def complete_graph(num_atoms: int) -> tuple[Tensor, Tensor]:
    """Fully-connected directed graph over the N atoms of one crystal.

    Unlike the radius graph, the edge set is GEOMETRY-INDEPENDENT: it never
    depends on the (evolving) coordinates or lattice, so it cannot go stale
    during sampling and there is no train/sample topology mismatch. For N<=20
    this is <=380 edges/crystal -- cheap. Periodicity is handled downstream by
    the minimum-image convention + periodic Fourier edge features (EGNN); the
    returned cell_offset is therefore zero.
    """
    idx = torch.arange(num_atoms)
    src, dst = torch.meshgrid(idx, idx, indexing="ij")
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]], dim=0).long()
    cell_offset = torch.zeros(edge_index.shape[1], 3)
    return edge_index, cell_offset
