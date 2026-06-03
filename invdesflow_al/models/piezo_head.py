"""Piezoelectric tensor predictor — Plan C scoring head.

Small EGNN regressor: structure -> scalar log|eij_max|. Reuses EGNNLayer
from the generator denoiser but with smaller hidden width and depth, no
diffusion conditioning, and a graph-level mean-pool readout.

Why log-space target: the dataset eij_max has a heavy tail
(median ~0.25, max ~46 C/m^2). Training in log-space stabilizes the
regression and makes Spearman ranking robust to the tail.

Input  : `CrystalBatch` (from data/batch.py, mode="complete")
Output : per-graph scalar prediction `[B]` in log-space
         (caller should `exp()` it to recover C/m^2)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ..data.batch import CrystalBatch
from .egnn import EGNNLayer, FourierFrac, _scatter_mean


class PiezoHead(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        num_layers: int = 3,
        num_atom_types: int = 100,
        fourier_freqs: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_atom_types = num_atom_types
        act = nn.SiLU()

        self.atom_embed = nn.Embedding(num_atom_types + 1, hidden)
        self.lattice_in = nn.Linear(6, hidden)

        self.fourier = FourierFrac(fourier_freqs)
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden, self.fourier.out_dim, act) for _ in range(num_layers)]
        )

        self.readout = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), act,
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), act,
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _lattice_params(lat: Tensor) -> Tensor:
        """[B,3,3] -> [B,6] = (a, b, c, cos alpha, cos beta, cos gamma)."""
        a, b, c = lat[:, 0], lat[:, 1], lat[:, 2]
        la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
        eps = 1e-8
        cos_al = (b * c).sum(-1) / (lb * lc + eps)
        cos_be = (a * c).sum(-1) / (la * lc + eps)
        cos_ga = (a * b).sum(-1) / (la * lb + eps)
        return torch.stack([la, lb, lc, cos_al, cos_be, cos_ga], dim=-1)

    def forward(self, batch: CrystalBatch) -> Tensor:
        n = batch.atom_types.shape[0]
        ei, off = batch.edge_index, batch.cell_offset
        src, dst = ei[0], ei[1]

        # periodic fractional difference (translation-invariant, EGNN-style)
        dfrac = batch.frac_coords[src] + off - batch.frac_coords[dst]
        dfrac = dfrac - dfrac.round()
        lat_e = batch.lattices[batch.batch[dst]]
        dcart = torch.einsum("ei,eij->ej", dfrac, lat_e)
        dist2 = (dcart * dcart).sum(-1)
        edge_feat = self.fourier(dfrac)

        # node init: atom embedding + per-graph lattice signal
        lat6 = self._lattice_params(batch.lattices)
        h = (
            self.atom_embed(batch.atom_types.clamp(0, self.num_atom_types))
            + self.lattice_in(lat6)[batch.batch]
        )

        # EGNNLayer applies its own internal residual; we just take its output.
        for layer in self.layers:
            h, _dpos = layer(h, dfrac, dist2, edge_feat, ei, n)

        # graph mean-pool then MLP
        graph_h = _scatter_mean(h, batch.batch, batch.num_graphs)
        return self.readout(graph_h).squeeze(-1)
