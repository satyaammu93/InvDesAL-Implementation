"""EGNN-based denoising network phi  (Table S.2: EGNN, 6 layers, hidden 512, SiLU).

phi(L_t, A_t, F_t, N, t) -> (eps_L, eps_A, eps_F)   [Algorithms 1 & 2]

Equivariance (paper, Methods):
  * translation equivariance  : messages use RELATIVE fractional differences
                                only (periodic, translation invariant).
  * rotation/reflection equiv.: scalar messages depend on SQUARED cartesian
                                distances; the fractional-coordinate update is
                                a sum of relative-difference vectors weighted
                                by those scalars (EGNN coordinate update).
  * permutation equivariance  : sum aggregation over neighbors.

Heads:
  eps_A : per-atom, R^{Ntot x K}        (DDPM noise on one-hot atom types)
  eps_F : per-atom, R^{Ntot x 3}        (score / noise on fractional coords)
  eps_L : per-crystal, R^{B x 3 x 3}    (DDPM noise on the lattice)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ..data.batch import CrystalBatch


def _scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


def _scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    s = _scatter_sum(src, index, dim_size)
    cnt = _scatter_sum(torch.ones_like(index, dtype=src.dtype), index, dim_size)
    return s / cnt.clamp_min(1.0).unsqueeze(-1)


class SinusoidalEmbedding(nn.Module):
    """Standard transformer/diffusion timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        ang = t.float()[:, None] * freqs[None, :]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


class FourierFrac(nn.Module):
    """Periodic, translation-invariant embedding of a fractional difference.

    sin/cos of 2*pi*k*delta_frac is invariant to integer lattice translations,
    so it is the natural periodic analogue of EGNN's relative coordinates.
    """

    def __init__(self, num_freqs: int = 64):
        super().__init__()
        self.register_buffer(
            "k", 2.0 * math.pi * torch.arange(1, num_freqs + 1).float(), persistent=False
        )

    @property
    def out_dim(self) -> int:
        return 3 * 2 * self.k.numel()

    def forward(self, dfrac: Tensor) -> Tensor:  # [E, 3] -> [E, 3*2*F]
        ang = dfrac[..., None] * self.k  # [E, 3, F]
        return torch.cat([ang.sin(), ang.cos()], dim=-1).reshape(dfrac.shape[0], -1)


class EGNNLayer(nn.Module):
    def __init__(self, hidden: int, edge_in: int, act: nn.Module):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + edge_in + 1, hidden), act,
            nn.Linear(hidden, hidden), act,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), act,
            nn.Linear(hidden, hidden),
        )
        # scalar gate that weights each relative-difference vector (EGNN coord update)
        self.coord_mlp = nn.Sequential(nn.Linear(hidden, hidden), act, nn.Linear(hidden, 1))

    def forward(
        self, h: Tensor, dfrac: Tensor, dist2: Tensor, edge_feat: Tensor, ei: Tensor, n: int
    ) -> tuple[Tensor, Tensor]:
        src, dst = ei[0], ei[1]
        m = self.edge_mlp(torch.cat([h[dst], h[src], edge_feat, dist2[:, None]], dim=-1))
        agg = _scatter_sum(m, dst, n)
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        # equivariant fractional-coordinate signal: weighted relative differences
        coord_msg = dfrac * self.coord_mlp(m)
        dpos = _scatter_mean(coord_msg, dst, n)
        return h, dpos


class EGNNDenoiser(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_atom_types: int = 100,
        fourier_freqs: int = 64,
        time_embed_dim: int = 256,
        lattice_x0_bound: float = 8.0,
    ):
        super().__init__()
        self.num_atom_types = num_atom_types
        # The L head predicts the NORMALIZED clean lattice (x0). A normalized
        # lattice is genuinely bounded (~+/-5 sigma over real data), so we cap
        # the head with B*tanh(raw/B). This is a bound on a legitimately
        # bounded *signal* prediction -- unlike the removed +-1e4 state clamp,
        # it cannot mask divergence, only prevents OOD extrapolation blow-ups.
        self.lattice_x0_bound = float(lattice_x0_bound)
        act = nn.SiLU()

        self.fourier = FourierFrac(fourier_freqs)
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim), act,
            nn.Linear(hidden_dim, hidden_dim),
        )
        # node init from noised atom one-hot + 6 lattice params (a,b,c,alpha,beta,gamma)
        self.node_in = nn.Linear(num_atom_types, hidden_dim)
        self.lattice_in = nn.Linear(6, hidden_dim)

        edge_in = self.fourier.out_dim
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim, edge_in, act) for _ in range(num_layers)]
        )

        self.atom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), act, nn.Linear(hidden_dim, num_atom_types)
        )
        self.frac_scale = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), act, nn.Linear(hidden_dim, 1)
        )
        self.lattice_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), act, nn.Linear(hidden_dim, 9)
        )

    @staticmethod
    def _lattice_params(lat: Tensor) -> Tensor:
        """[B,3,3] -> [B,6] = (a, b, c, cos a, cos b, cos g)  (SE(3) invariant)."""
        a, b, c = lat[:, 0], lat[:, 1], lat[:, 2]
        la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
        eps = 1e-8
        cos_al = (b * c).sum(-1) / (lb * lc + eps)
        cos_be = (a * c).sum(-1) / (la * lc + eps)
        cos_ga = (a * b).sum(-1) / (la * lb + eps)
        return torch.stack([la, lb, lc, cos_al, cos_be, cos_ga], dim=-1)

    def forward(
        self, batch: CrystalBatch, atom_onehot: Tensor, t_per_graph: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        n = atom_onehot.shape[0]
        ei, off = batch.edge_index, batch.cell_offset
        src, dst = ei[0], ei[1]

        # periodic relative fractional difference (translation invariant).
        # minimum-image convention: for the complete graph (off == 0) this
        # selects the nearest periodic image; for the radius graph (off the
        # explicit image) it is a no-op since dfrac is already small.
        dfrac = batch.frac_coords[src] + off - batch.frac_coords[dst]
        dfrac = dfrac - dfrac.round()
        # cartesian squared distance (rotation/reflection invariant)
        lat_e = batch.lattices[batch.batch[dst]]  # [E,3,3]
        dcart = torch.einsum("ei,eij->ej", dfrac, lat_e)
        dist2 = (dcart * dcart).sum(-1)
        edge_feat = self.fourier(dfrac)

        lat6 = self._lattice_params(batch.lattices)  # [B,6]
        h = (
            self.node_in(atom_onehot)
            + self.lattice_in(lat6)[batch.batch]
            + self.time_embed(t_per_graph)[batch.batch]
        )

        dpos_total = torch.zeros_like(batch.frac_coords)
        for layer in self.layers:
            h, dpos = layer(h, dfrac, dist2, edge_feat, ei, n)
            dpos_total = dpos_total + dpos

        # A head -> x0 (clean atom types). Clean A is per-atom one-hot
        # (probability simplex), so the natural bounded head is softmax: output
        # in [0,1]^K, sums to 1 per atom. This is the categorical analog of L's
        # tanh-bounded x0 -- mirrors the lattice lesson that eps-prediction is
        # fragile at high t (1/sqrt(abar) amplification + clamp saturation).
        logits_A = self.atom_head(h)
        x0_A = torch.softmax(logits_A, dim=-1)
        eps_F = dpos_total * self.frac_scale(h)  # equivariant per-atom 3-vector
        graph_h = _scatter_mean(h, batch.batch, batch.num_graphs)
        # L head -> normalized clean lattice (x0), bounded by B*tanh(raw/B).
        raw_L = self.lattice_head(graph_h).reshape(-1, 3, 3)
        B = self.lattice_x0_bound
        x0_L = B * torch.tanh(raw_L / B)
        return x0_L, x0_A, eps_F
