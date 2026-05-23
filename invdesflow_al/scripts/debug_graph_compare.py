"""Gate 7 (PROGRESS.md Entry 4): is the frozen sampling graph hurting things?

The sampler builds the graph once from RANDOM templates and freezes it; the
model was trained on graphs built from real (clean) geometry. This probes the
train/sample topology mismatch directly: at several noise levels, run the
trained denoiser on the SAME noised state with three graphs -
  clean    : edges from the true crystal geometry (what training used)
  noised   : edges rebuilt from the current noised state (geometry-consistent)
  random   : edges from a random template (what Algorithm 2 actually uses)
and compare the eps predictions. If `random` differs sharply from `clean`,
the frozen-template graph is silently corrupting denoising.

Usage:
  python -m invdesflow_al.scripts.debug_graph_compare \
      --ckpt generator.ckpt --device cuda
"""

from __future__ import annotations

import argparse

import torch

from ..data.batch import collate
from ..data.datasets import load_structures
from ..data.representation import (Crystal, atomic_numbers_to_onehot,
                                   wrap_frac)
from ..models.diffusion import _replace
from ..models.generator import CrystalGenerator, load_config


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_raw/pretrain.jsonl")
    ap.add_argument("--ckpt", default="generator.ckpt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    try:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        gen = CrystalGenerator(ck["cfg"], device=args.device)
        gen.load_state_dict(ck["state_dict"])
        tag = f"trained ckpt {args.ckpt}"
    except FileNotFoundError:
        gen = CrystalGenerator(load_config(), device=args.device)
        tag = "fresh (untrained) model - ckpt not found"
    gen.eval()
    print(f"model: {tag}\n")

    crystal = next(iter(load_structures(args.manifest))).to_crystal()
    onehot = atomic_numbers_to_onehot(crystal.atom_types, gen.num_atom_types)
    dif = gen.diffusion
    cutoff, mn = gen.cutoff, gen.max_nbr

    clean_batch = collate([crystal], cutoff, mn).to(args.device)
    rand_crystal = Crystal.random(crystal.num_atoms)
    rand_batch = collate([rand_crystal], cutoff, mn).to(args.device)

    print(f"  {'t':>5} {'cos(clean,random)':>18} {'cos(clean,noised)':>18}  "
          f"(eps_L / eps_A / eps_F)")
    worst = 1.0
    for t in (50, 250, 500, 900):
        abar, sigma = dif.abar[t], dif.sigmas[t]
        L0 = crystal.lattice.to(args.device).unsqueeze(0)        # [1,3,3]
        L_t = abar.sqrt() * L0 + (1 - abar).sqrt() * torch.randn_like(L0)
        A_t = abar.sqrt() * onehot.to(args.device) + \
            (1 - abar).sqrt() * torch.randn_like(onehot).to(args.device)
        F_t = wrap_frac(crystal.frac_coords.to(args.device) +
                        sigma * torch.randn_like(crystal.frac_coords).to(args.device))
        tt = torch.full((1,), float(t), device=args.device)

        # graph rebuilt from the current noised state
        noised_crystal = Crystal(crystal.atom_types,
                                 wrap_frac(F_t.cpu()), L_t[0].cpu())
        noised_batch = collate([noised_crystal], cutoff, mn).to(args.device)

        outs = {}
        for name, base in (("clean", clean_batch), ("noised", noised_batch),
                           ("random", rand_batch)):
            cur = _replace(base, frac_coords=F_t, lattices=L_t)
            with torch.no_grad():
                outs[name] = gen.net(cur, A_t, tt)

        cr = tuple(cos(outs["clean"][i], outs["random"][i]) for i in range(3))
        cn = tuple(cos(outs["clean"][i], outs["noised"][i]) for i in range(3))
        worst = min(worst, *cr)
        print(f"  {t:>5}  L{cr[0]:+.2f} A{cr[1]:+.2f} F{cr[2]:+.2f}      "
              f" L{cn[0]:+.2f} A{cn[1]:+.2f} F{cn[2]:+.2f}")

    print()
    if worst < 0.8:
        print(f"GATE 7 FAIL: random-template graph changes the denoiser output "
              f"(cosine drops to {worst:.2f}) - frozen sampling graph corrupts "
              f"denoising; rebuild topology or use a complete graph.")
    else:
        print(f"GATE 7 PASS: graph choice barely changes predictions "
              f"(min cosine {worst:.2f}).")


if __name__ == "__main__":
    main()
