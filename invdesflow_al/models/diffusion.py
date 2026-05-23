"""Diffusion process: noise schedules, Algorithm 1 (training), Algorithm 2 (sampling).

Mixed corruption (paper, Methods):
  * L (lattice) and A (atom types): standard DDPM (Ho et al. 2020), Gaussian.
  * F (fractional coords)         : score-matching (Song et al. 2021) with a
                                    WRAPPED normal on the torus -- via w(.).

beta-schedule shape is not given in the paper; we default to the cosine
schedule (Nichol & Dhariwal) -- configurable (see configs/generator.yaml,
REBUILD_PLAN.md sec. 11). lambda_t for the coord-loss target is set to
sigma_t (DiffCSP convention).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..data.representation import wrap_frac


def _cosine_betas(T: int, s: float = 0.008) -> Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * torch.pi / 2) ** 2
    abar = f / f[0]
    betas = 1 - abar[1:] / abar[:-1]
    return betas.clamp(1e-8, 0.999).float()


def _linear_betas(T: int, b0: float, b1: float) -> Tensor:
    return torch.linspace(b0, b1, T).float()


class DiffusionProcess:
    """Holds the schedules and implements the train loss and the sampler."""

    def __init__(self, cfg: dict, device: str | torch.device = "cpu"):
        d = cfg["diffusion"]
        self.T: int = int(d["num_steps"])
        self.device = torch.device(device)

        if d.get("beta_schedule", "cosine") == "cosine":
            betas = _cosine_betas(self.T)
        else:
            betas = _linear_betas(self.T, d["beta_start"], d["beta_end"])
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.abar = abar.to(self.device)
        self.abar_prev = torch.cat(
            [torch.ones(1, device=self.device), self.abar[:-1]]
        )

        # geometric sigma schedule for the score-matched fractional coords
        smin, smax = float(d["sigma_min"]), float(d["sigma_max"])
        self.sigmas = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(smin)),
                torch.log(torch.tensor(smax)),
                self.T,
            )
        ).to(self.device)
        self.trunc = int(d.get("wrapped_normal_trunc", 3))
        self.lambda_is_sigma = bool(d.get("lambda_coord_is_sigma", True))

        w = cfg["loss_weights"]
        self.w_coord = float(w["coord"])
        self.w_lattice = float(w["lattice"])
        self.w_type = float(w["type"])

        # Lattice channel: x0-prediction in a STATISTICALLY normalized space.
        # The raw 3x3 lattice is normalized by dataset per-entry mean/std so the
        # diffused quantity is O(1); the network predicts the clean (normalized)
        # lattice directly (x0-parameterization), avoiding the 1/sqrt(abar)
        # amplification of eps-prediction at high t. Stats come from the data
        # (see data.datasets.compute_lattice_stats) via cfg; identity by default.
        lm = d.get("lattice_mean")
        ls = d.get("lattice_std")
        self.lattice_mean = (torch.tensor(lm, dtype=torch.float)
                             if lm is not None else torch.zeros(3, 3)).to(self.device)
        self.lattice_std = (torch.tensor(ls, dtype=torch.float)
                            if ls is not None else torch.ones(3, 3)).to(self.device)

    def normalize_lattice(self, L: Tensor) -> Tensor:
        return (L - self.lattice_mean) / self.lattice_std

    def denormalize_lattice(self, Ln: Tensor) -> Tensor:
        return Ln * self.lattice_std + self.lattice_mean

    # ---- wrapped normal (torus) -------------------------------------------------
    def _wrapped_score(self, dy: Tensor, sigma: Tensor) -> Tensor:
        """grad_y log sum_n N(dy + n; 0, sigma^2), summed over n in [-K, K].

        dy: fractional offset (already the *signed* perturbation F_t - F_0,
        wrapped into (-0.5, 0.5)). sigma broadcast to dy's shape.
        """
        ns = torch.arange(-self.trunc, self.trunc + 1, device=dy.device).float()
        # shape [..., 2K+1]
        x = dy.unsqueeze(-1) + ns
        s2 = (sigma.unsqueeze(-1)) ** 2
        logp = -(x ** 2) / (2 * s2)
        w = torch.softmax(logp, dim=-1)  # responsibilities
        return (w * (-x / s2)).sum(-1)

    @staticmethod
    def _to_signed(d: Tensor) -> Tensor:
        """map a wrapped difference to (-0.5, 0.5]."""
        return d - torch.round(d)

    # ---- Algorithm 1: training loss --------------------------------------------
    def training_loss(self, model, batch, atom_onehot: Tensor) -> dict[str, Tensor]:
        B = batch.num_graphs
        dev = atom_onehot.device
        # 1: sample timestep t ~ U{1, T}
        t = torch.randint(0, self.T, (B,), device=dev)  # 0-indexed
        t_atom = t[batch.batch]

        # 2-3: noise vectors and schedule coefficients
        eps_L = torch.randn_like(batch.lattices)
        eps_A = torch.randn_like(atom_onehot)
        eps_F = torch.randn_like(batch.frac_coords)

        sa = self.abar[t].sqrt()
        sb = (1 - self.abar[t]).sqrt()
        sa_n = self.abar[t_atom].sqrt()[:, None]
        sb_n = (1 - self.abar[t_atom]).sqrt()[:, None]
        sigma = self.sigmas[t_atom][:, None]

        # 5-7: perturbed components
        # L: diffuse in normalized space; A: DDPM on one-hot; F: wrapped score.
        L0n = self.normalize_lattice(batch.lattices)
        L_t = sa[:, None, None] * L0n + sb[:, None, None] * eps_L
        A_t = sa_n * atom_onehot + sb_n * eps_A
        F0 = batch.frac_coords
        F_t = wrap_frac(F0 + sigma * eps_F)

        noisy = _replace(batch, frac_coords=F_t, lattices=L_t)

        # 8-9: predict network outputs.  L -> x0 (normalized lattice);
        # A -> x0 (per-atom softmax over atom types); F -> lambda_t * score.
        L_pred, A_x0_pred, eps_F_hat = model(noisy, A_t, t.float())

        # 10-13: losses (MSE against the clean targets)
        loss_L = ((L0n - L_pred) ** 2).mean()              # x0 for L
        loss_A = ((atom_onehot - A_x0_pred) ** 2).mean()   # x0 for A

        dy = self._to_signed(F_t - F0)
        score = self._wrapped_score(dy, sigma)
        lam = sigma if self.lambda_is_sigma else torch.ones_like(sigma)
        target_F = lam * score
        loss_F = ((target_F - eps_F_hat) ** 2).mean()

        total = self.w_coord * loss_F + self.w_lattice * loss_L + self.w_type * loss_A
        return {"total": total, "coord": loss_F, "lattice": loss_L, "type": loss_A}

    # ---- Algorithm 2: sampling --------------------------------------------------
    @torch.no_grad()
    def sample(self, model, batch, num_atom_types: int, gamma: float = 0.5,
               stats: dict | None = None):
        """Predictor-corrector sampling. `batch` supplies num_atoms / graph
        structure; its frac/lattice/atoms are overwritten by the sampler.

        If `stats` is a dict, per-timestep A-channel diagnostics are appended
        (lists keyed by 'max_a', 'med_a', 'sat_frac', 't'). The caller is
        responsible for aggregating across batches.
        """
        dev = next(model.parameters()).device
        n = batch.frac_coords.shape[0]
        B = batch.num_graphs

        # 2: initialize noise
        L = torch.randn(B, 3, 3, device=dev)
        A = torch.randn(n, num_atom_types, device=dev)
        F = torch.rand(n, 3, device=dev)
        bidx = batch.batch

        for t in range(self.T - 1, -1, -1):
            tt = torch.full((B,), t, device=dev, dtype=torch.float)
            cur = _replace(batch, frac_coords=wrap_frac(F), lattices=L)
            L_x0, A_x0, eps_F = model(cur, A, tt)   # L head -> x0; A head -> x0

            beta = self.betas[t]
            alpha = self.alphas[t]
            abar = self.abar[t]
            abar_prev = self.abar_prev[t]
            noise_scale = (beta * (1 - abar_prev) / (1 - abar)).clamp_min(0).sqrt()

            # 10-12: reverse update for L and A.
            # L: x0-parameterized DDPM posterior mean (no 1/sqrt(abar)
            #    amplification -> stable at high t). No clamp.
            zL = torch.randn_like(L) if t > 0 else torch.zeros_like(L)
            coef_x0 = abar_prev.sqrt() * beta / (1 - abar)
            coef_xt = alpha.sqrt() * (1 - abar_prev) / (1 - abar)
            L = coef_x0 * L_x0 + coef_xt * L + noise_scale * zL
            # A: x0-parameterized DDPM posterior (mirrors L's fix). The
            # softmax-bounded head puts A_x0 in [0,1]^K and sums to 1 per atom;
            # combined with the convex coefficients below, this keeps A
            # naturally bounded -- no state clamp needed (the +-50 clamp was a
            # symptom of the eps-prediction fragility we just removed).
            zA = torch.randn_like(A) if t > 0 else torch.zeros_like(A)
            A = coef_x0 * A_x0 + coef_xt * A + noise_scale * zA
            if stats is not None:
                a_abs = A.abs()
                stats.setdefault("max_a", []).append(float(a_abs.max()))
                stats.setdefault("med_a", []).append(float(a_abs.median()))
                # sat_frac at the OLD 49 boundary kept as a sanity tripwire --
                # with x0 it should remain ~0.
                stats.setdefault("sat_frac", []).append(
                    float((a_abs >= 49.0).float().mean()))
                stats.setdefault("t", []).append(int(t))

            # 13-18: predictor-corrector for fractional coords.
            # The network is trained to predict lambda_t * score (lambda_t =
            # sigma_t); recover the raw score before stepping.
            sig_t = self.sigmas[t]
            sig_tm1 = self.sigmas[t - 1] if t > 0 else torch.zeros_like(sig_t)
            lam_t = sig_t if self.lambda_is_sigma else torch.ones_like(sig_t)
            score = eps_F / lam_t.clamp_min(1e-8)
            step = (sig_t ** 2 - sig_tm1 ** 2).clamp_min(0)
            zF = torch.randn_like(F)
            # VE ancestral predictor (Song et al. 2021, SMLD ancestral
            # sampling): with the exact score this provably contracts
            # F_t -> F_0 by a factor sigma_{t-1}^2 / sigma_t^2 per step.
            F_half = wrap_frac(
                F + step * score
                + (sig_tm1 * step.sqrt() / sig_t.clamp_min(1e-8)) * zF
            )

            half = _replace(batch, frac_coords=F_half, lattices=L)
            t_prev = torch.full((B,), max(t - 1, 0), device=dev, dtype=torch.float)
            _, _, eps_F2 = model(half, A, t_prev)
            lam_tm1 = sig_tm1 if self.lambda_is_sigma else torch.ones_like(sig_tm1)
            score2 = eps_F2 / lam_tm1.clamp_min(1e-8)
            # Langevin corrector with a SAFE step d_t = gamma * sigma_{t-1}^2.
            # (The previous d_t = gamma * sigma_{t-1}^2 / sigma_min^2 reached
            #  ~1e4 and kicked F to random every step -- Gate 6 failure.)
            d_t = gamma * (sig_tm1 ** 2)
            zF2 = torch.randn_like(F)
            F = wrap_frac(F_half + d_t * score2 + (2 * d_t).clamp_min(0).sqrt() * zF2)

        atom_types = A.argmax(dim=-1).long() + 1
        # L was diffused in normalized space -> restore physical units.
        return atom_types, wrap_frac(F), self.denormalize_lattice(L), bidx


def _replace(batch, *, frac_coords: Tensor, lattices: Tensor):
    from ..data.batch import CrystalBatch

    return CrystalBatch(
        atom_types=batch.atom_types,
        frac_coords=frac_coords,
        lattices=lattices,
        num_atoms=batch.num_atoms,
        batch=batch.batch,
        edge_index=batch.edge_index,
        cell_offset=batch.cell_offset,
    )
