"""Gate 4 + Gate 8 (PROGRESS.md Entry 4): overfit ONE crystal.

Gate 4 (default): train one crystal, then probe the denoiser at fixed t -
do predicted eps_A/eps_L/eps_F correlate with the true noise, and does the
estimated x0 (A/F/L) match the original?  No sampling.

Gate 8 (--sample): after overfitting, run the full Algorithm-2 sampler 32x
for the same N and check the formula / lattice / volume.

Usage:
  python -m invdesflow_al.scripts.debug_overfit_one --steps 1500 --device cuda
  python -m invdesflow_al.scripts.debug_overfit_one --steps 1500 --sample --device cuda
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch

from ..data.batch import collate
from ..data.datasets import load_structures
from ..data.representation import atomic_numbers_to_onehot, wrap_frac
from ..models.diffusion import _replace
from ..models.generator import CrystalGenerator, config_with_lattice_stats


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float() - a.float().mean()
    b = b.flatten().float() - b.float().mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def signed(d: torch.Tensor) -> torch.Tensor:
    return d - torch.round(d)


@torch.no_grad()
def denoise_probe(gen, crystal, ts):
    """A: eps-prediction.  L: x0-prediction (normalized).  F: wrapped score.
    corr_L is corr(true normalized x0, predicted x0)."""
    batch = collate([crystal], gen.cutoff, gen.max_nbr, gen.graph_mode).to(gen._device)
    onehot = atomic_numbers_to_onehot(batch.atom_types, gen.num_atom_types)
    dif = gen.diffusion
    L0n = dif.normalize_lattice(batch.lattices)            # true normalized x0
    print(f"  {'t':>5} {'corr_A':>8} {'corr_L':>8} {'corr_F':>8} "
          f"{'A0 acc':>8} {'L0 relerr':>10} {'F0 err':>8}")
    rows = []
    for t in ts:
        t = min(t, dif.T - 1)
        abar, sigma = dif.abar[t], dif.sigmas[t]
        eps_L = torch.randn_like(L0n)
        eps_A = torch.randn_like(onehot)
        eps_F = torch.randn_like(batch.frac_coords)
        L_t = abar.sqrt() * L0n + (1 - abar).sqrt() * eps_L      # normalized space
        A_t = abar.sqrt() * onehot + (1 - abar).sqrt() * eps_A
        F_t = wrap_frac(batch.frac_coords + sigma * eps_F)
        cur = _replace(batch, frac_coords=F_t, lattices=L_t)
        tt = torch.full((1,), float(t), device=gen._device)
        L_x0, eA, eF = gen.net(cur, A_t, tt)                    # L head -> x0

        # true coord target = lambda_t * wrapped score (training convention)
        dy = signed(F_t - batch.frac_coords)
        score = dif._wrapped_score(dy, torch.full_like(dy, float(sigma)))
        targetF = (sigma if dif.lambda_is_sigma else 1.0) * score

        L0_pred = dif.denormalize_lattice(L_x0)                 # physical units
        A0 = (A_t - (1 - abar).sqrt() * eA) / abar.sqrt()
        F0 = wrap_frac(F_t + sigma * eF)

        a_acc = float((A0.argmax(-1) == onehot.argmax(-1)).float().mean())
        l_err = float((L0_pred - batch.lattices).norm() / batch.lattices.norm())
        f_err = float(signed(F0 - batch.frac_coords).abs().mean())
        cA, cL, cF = pearson(eps_A, eA), pearson(L0n, L_x0), pearson(targetF, eF)
        print(f"  {t:>5} {cA:>8.3f} {cL:>8.3f} {cF:>8.3f} "
              f"{a_acc:>8.2f} {l_err:>10.3f} {f_err:>8.3f}")
        rows.append((cA, cL, cF, a_acc, l_err, f_err))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_raw/pretrain.jsonl")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--sample", action="store_true", help="also run Gate 8")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    crystal = next(iter(load_structures(args.manifest))).to_crystal()
    true_z = sorted(crystal.atom_types.tolist())
    print(f"target crystal: N={crystal.num_atoms}  Z={true_z}\n")

    cfg = config_with_lattice_stats(args.manifest)   # dataset lattice mean/std
    gen = CrystalGenerator(cfg, device=args.device)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-4)
    for step in range(args.steps):
        out = gen.train_step([crystal])
        opt.zero_grad()
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
        opt.step()
        if step % max(args.steps // 6, 1) == 0 or step == args.steps - 1:
            print(f"step {step:5d}  c/l/t = {out['coord']:.4f}/"
                  f"{out['lattice']:.4f}/{out['type']:.4f}")

    print("\ndenoiser probe (Gate 4):")
    rows = denoise_probe(gen, crystal, ts=(50, 250, 500, 900))

    cA = sum(r[0] for r in rows) / len(rows)
    cL = sum(r[1] for r in rows) / len(rows)
    cF = sum(r[2] for r in rows) / len(rows)
    aacc = sum(r[3] for r in rows) / len(rows)
    fails = []
    if cA < 0.3:
        fails.append(f"eps_A weakly correlated with true noise (mean r={cA:.2f})")
    if cL < 0.3:
        fails.append(f"L x0-prediction weakly correlated with truth (mean r={cL:.2f})")
    if cF < 0.3:
        fails.append(f"eps_F weakly correlated with coord target (mean r={cF:.2f})")
    if aacc < 0.6:
        fails.append(f"x0 atom-type accuracy low (mean {aacc:.2f})")
    print()
    if fails:
        print("GATE 4 FAIL:")
        for f in fails:
            print("  -", f)
    else:
        print("GATE 4 PASS - denoiser overfits one crystal")

    if args.sample:
        print("\nfull sampling (Gate 8): 32x")
        samples = gen.sample(num_atoms=[crystal.num_atoms] * 32)
        forms = Counter("-".join(map(str, sorted(s.atom_types.tolist())))
                        for s in samples)
        dets = [float(torch.det(s.lattice).abs()) for s in samples]
        vpa = [d / crystal.num_atoms for d in dets]
        print(f"  distinct formulas: {len(forms)}   top: {forms.most_common(3)}")
        print(f"  lattice |det| min/med/max: {min(dets):.2e} / "
              f"{sorted(dets)[16]:.2e} / {max(dets):.2e}")
        print(f"  volume/atom min/med/max: {min(vpa):.1f} / "
              f"{sorted(vpa)[16]:.1f} / {max(vpa):.1f}")
        g8 = []
        if max(vpa) > 500 or min(vpa) <= 0:
            g8.append("volume/atom out of (0,500]")
        if max(dets) > 1e6:
            g8.append("lattice determinant exploded")
        print("GATE 8 " + ("FAIL: " + "; ".join(g8) if g8 else
                            "PASS - sane sampled crystal"))
        import json
        sv = sorted(vpa)
        print("RESULT_JSON " + json.dumps({
            "gate": 8, "pass": not g8,
            "distinct_formulas": len(forms),
            "vpa_min": round(min(vpa), 3),
            "vpa_median": round(sv[len(sv) // 2], 3),
            "vpa_max": round(max(vpa), 3),
        }))


if __name__ == "__main__":
    main()
