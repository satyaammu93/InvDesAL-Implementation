"""Gate 5 + 6 (PROGRESS.md Entry 4): oracle reverse samplers.

Runs the reverse process with the TRUE noise/score substituted for the
network output. If the reverse algebra + schedule are correct, this must
recover the known A/L/F and stay bounded WITHOUT any clamp. Divergence here
means the sampler math itself is wrong (independent of the trained model).

Gate 5: oracle DDPM reverse for atom types A and lattice L.
Gate 6: oracle wrapped-coordinate reverse for fractional coords F.

Usage:
  python -m invdesflow_al.scripts.debug_oracle_sampler --device cpu
"""

from __future__ import annotations

import argparse

import torch

from ..data.datasets import load_structures
from ..data.representation import atomic_numbers_to_onehot, wrap_frac
from ..models.generator import CrystalGenerator, config_with_lattice_stats


def signed(d: torch.Tensor) -> torch.Tensor:
    return d - torch.round(d)


def oracle_ddpm(dif, x0: torch.Tensor, name: str) -> tuple[float, float]:
    """Reverse DDPM using the exact eps at every step (deterministic, z=0)."""
    T = dif.T
    abarT = dif.abar[T - 1]
    x = abarT.sqrt() * x0 + (1 - abarT).sqrt() * torch.randn_like(x0)  # x_T
    max_abs = float(x.abs().max())
    for t in range(T - 1, -1, -1):
        abar, alpha, beta = dif.abar[t], dif.alphas[t], dif.betas[t]
        eps_true = (x - abar.sqrt() * x0) / (1 - abar).sqrt()          # oracle
        x = (x - beta / (1 - abar).sqrt() * eps_true) / alpha.sqrt()   # reverse step
        max_abs = max(max_abs, float(x.abs().max()))
    relerr = float((x - x0).norm() / x0.norm().clamp_min(1e-8))
    print(f"  {name}: recovered rel-err = {relerr:.4e}   "
          f"max|x| over trajectory = {max_abs:.3e}")
    return relerr, max_abs


def oracle_ddpm_x0(dif, x0n: torch.Tensor, name: str) -> tuple[float, float]:
    """Reverse the x0-parameterized DDPM with the exact x0 at every step
    (deterministic, z=0). x0n must be in NORMALIZED space."""
    T = dif.T
    abarT = dif.abar[T - 1]
    x = abarT.sqrt() * x0n + (1 - abarT).sqrt() * torch.randn_like(x0n)  # x_T
    max_abs = float(x.abs().max())
    for t in range(T - 1, -1, -1):
        abar, alpha, beta = dif.abar[t], dif.alphas[t], dif.betas[t]
        abar_prev = dif.abar_prev[t]
        coef_x0 = abar_prev.sqrt() * beta / (1 - abar)
        coef_xt = alpha.sqrt() * (1 - abar_prev) / (1 - abar)
        x = coef_x0 * x0n + coef_xt * x        # perfect net: x0_hat = x0n
        max_abs = max(max_abs, float(x.abs().max()))
    relerr = float((x - x0n).norm() / x0n.norm().clamp_min(1e-8))
    print(f"  {name}: recovered rel-err = {relerr:.4e}   "
          f"max|x| over trajectory = {max_abs:.3e}")
    return relerr, max_abs


def oracle_coord(dif, F0: torch.Tensor, gamma: float = 0.5) -> tuple[float, float]:
    """Reverse the wrapped-coordinate predictor-corrector with the exact
    score at every step (deterministic, z=0). Mirrors the corrected
    DiffusionProcess.sample: a perfect network outputs lambda_t*score, which
    the sampler converts back to the raw score - so the oracle uses the true
    wrapped score directly."""
    T = dif.T
    F = wrap_frac(torch.rand_like(F0))               # F_T ~ U(0,1)
    max_dev = 0.0
    for t in range(T - 1, -1, -1):
        sig_t = dif.sigmas[t]
        sig_tm1 = dif.sigmas[t - 1] if t > 0 else torch.zeros_like(sig_t)

        sc = dif._wrapped_score(signed(F - F0), torch.full_like(F, float(sig_t)))
        step = (sig_t ** 2 - sig_tm1 ** 2).clamp_min(0)
        F_half = wrap_frac(F + step * sc)                       # VE predictor

        sc2 = dif._wrapped_score(signed(F_half - F0),
                                 torch.full_like(F, float(sig_tm1.clamp_min(1e-8))))
        d_t = gamma * (sig_tm1 ** 2)
        F = wrap_frac(F_half + d_t * sc2)                       # Langevin corrector
        max_dev = max(max_dev, float(signed(F - F0).abs().max()))
    err = float(signed(F - F0).abs().mean())
    print(f"  F: recovered mean wrapped-err = {err:.4e}   "
          f"max deviation over trajectory = {max_dev:.3e}")
    return err, max_dev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_raw/pretrain.jsonl")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    crystal = next(iter(load_structures(args.manifest))).to_crystal()
    cfg = config_with_lattice_stats(args.manifest)
    gen = CrystalGenerator(cfg, device=args.device)  # schedules + lattice stats
    dif = gen.diffusion
    L0 = crystal.lattice.to(args.device)
    A0 = atomic_numbers_to_onehot(crystal.atom_types, gen.num_atom_types).to(args.device)
    F0 = crystal.frac_coords.to(args.device)
    print(f"target crystal: N={crystal.num_atoms}  Z={crystal.atom_types.tolist()}\n")

    print("Gate 5 - oracle DDPM reverse (A: eps-param, L: x0-param):")
    relA, maxA = oracle_ddpm(dif, A0, "A")
    relL, maxL = oracle_ddpm_x0(dif, dif.normalize_lattice(L0), "L")
    print("\nGate 6 - oracle wrapped-coordinate reverse (F):")
    errF, devF = oracle_coord(dif, F0)

    print()
    g5 = []
    if relA > 0.05 or maxA > 1e3:
        g5.append(f"A not recovered (rel-err {relA:.2e}, max {maxA:.1e})")
    if relL > 0.05 or maxL > 1e3:
        g5.append(f"L not recovered (rel-err {relL:.2e}, max {maxL:.1e})")
    print("GATE 5 " + ("FAIL: " + "; ".join(g5) if g5 else
          "PASS - DDPM reverse algebra recovers A/L with no clamp"))

    g6 = []
    if errF > 0.05 or devF > 0.5 + 1e-3:
        g6.append(f"F not recovered (err {errF:.2e}, max dev {devF:.2e})")
    print("GATE 6 " + ("FAIL: " + "; ".join(g6) if g6 else
          "PASS - wrapped-coordinate reverse recovers F"))


if __name__ == "__main__":
    main()
