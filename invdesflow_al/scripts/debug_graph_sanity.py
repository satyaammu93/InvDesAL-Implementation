"""Gate 2 (PROGRESS.md Entry 4): is batching / graph construction sane?

Collates M real crystals and reports edge-count/atom, neighbor-distance
quantiles, cell-offset stats, and empty-neighbor atoms. Prints PASS/FAIL.

Usage:
  python -m invdesflow_al.scripts.debug_graph_sanity \
      --manifest data_raw/pretrain.jsonl --m 20
"""

from __future__ import annotations

import argparse

import torch

from ..data.batch import collate
from ..data.datasets import load_structures
from ..models.generator import load_config


def quants(t: torch.Tensor):
    if t.numel() == 0:
        return (0.0, 0.0, 0.0)
    s = t.float().sort().values
    n = s.numel()
    return (float(s[0]), float(s[n // 2]), float(s[-1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_raw/pretrain.jsonl")
    ap.add_argument("--m", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config()
    cutoff = float(cfg["data"]["neighbor_cutoff_radius"])
    max_nbr = int(cfg["data"]["max_neighbors_per_atom"])

    crystals = []
    for r in load_structures(args.manifest):
        crystals.append(r.to_crystal())
        if len(crystals) >= args.m:
            break
    batch = collate(crystals, cutoff, max_nbr)
    n_atoms = batch.atom_types.shape[0]
    n_edges = batch.edge_index.shape[1]
    print(f"collated {batch.num_graphs} crystals, {n_atoms} atoms, "
          f"{n_edges} edges  (cutoff {cutoff} A, max_nbr {max_nbr})\n")

    # incoming edges per atom (dst)
    dst = batch.edge_index[1]
    deg = torch.zeros(n_atoms, dtype=torch.long)
    deg.index_add_(0, dst, torch.ones_like(dst))
    empty = int((deg == 0).sum())
    dmin, dmed, dmax = quants(deg)

    # neighbor cartesian distances
    src = batch.edge_index[0]
    dfrac = batch.frac_coords[src] + batch.cell_offset - batch.frac_coords[dst]
    lat_e = batch.lattices[batch.batch[dst]]
    dcart = torch.einsum("ei,eij->ej", dfrac, lat_e)
    dist = dcart.norm(dim=-1)
    rmin, rmed, rmax = quants(dist)
    over_cut = int((dist > cutoff + 1e-4).sum())
    nonzero_off = int((batch.cell_offset.abs().sum(-1) > 0).sum())

    print(f"edges/atom (incoming): min/med/max = {dmin:.0f}/{dmed:.0f}/{dmax:.0f}")
    print(f"empty-neighbor atoms : {empty}/{n_atoms}")
    print(f"neighbor distance (A): min/med/max = {rmin:.2f}/{rmed:.2f}/{rmax:.2f}")
    print(f"edges over cutoff    : {over_cut}")
    print(f"edges using periodic image (offset != 0): {nonzero_off}/{n_edges}")

    fails = []
    if empty > 0.2 * n_atoms:
        fails.append(f"{empty}/{n_atoms} atoms have no neighbors")
    if dmax > max_nbr:
        fails.append(f"edge explosion: {dmax:.0f} > max_nbr {max_nbr}")
    if over_cut:
        fails.append(f"{over_cut} edges exceed cutoff {cutoff} A")
    if rmax > 3 * cutoff:
        fails.append(f"implausible neighbor distance {rmax:.1f} A")
    if not torch.isfinite(dist).all():
        fails.append("NaN/Inf in neighbor distances")

    print()
    if fails:
        print("GATE 2 FAIL:")
        for f in fails:
            print("  -", f)
    else:
        print("GATE 2 PASS - graph construction is sane")


if __name__ == "__main__":
    main()
