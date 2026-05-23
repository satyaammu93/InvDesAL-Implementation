"""Gate 3 (PROGRESS.md Entry 4): does the forward loss learn per channel?

Trains a fresh model for a few hundred mini-steps on a small fixed set of
real crystals, logging coord/lattice/type loss separately. Periodically probes
the atom-type channel for the collapse signature: estimate x0 for atom types
at several t and check whether predicted elements stay diverse or collapse to
a constant while the type loss falls.

PASS: all three channel losses decrease AND the type channel does not go
near-zero while its predictions become constant.

Usage:
  python -m invdesflow_al.scripts.debug_forward_loss \
      --manifest data_raw/pretrain.jsonl --k 16 --steps 400 --device cuda
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch

from ..data.batch import collate
from ..data.datasets import load_structures
from ..data.representation import atomic_numbers_to_onehot
from ..models.generator import CrystalGenerator, config_with_lattice_stats


@torch.no_grad()
def atom_type_probe(gen, crystals, ts=(50, 250, 500, 900)):
    """Estimate x0 for atom types at several t; return predicted-element set."""
    batch = collate(crystals, gen.cutoff, gen.max_nbr).to(gen._device)
    onehot = atomic_numbers_to_onehot(batch.atom_types, gen.num_atom_types)
    dif = gen.diffusion
    pred_elems = Counter()
    for t in ts:
        t = min(t, dif.T - 1)
        abar = dif.abar[t]
        eps_A = torch.randn_like(onehot)
        A_t = abar.sqrt() * onehot + (1 - abar).sqrt() * eps_A
        from ..models.diffusion import _replace
        cur = _replace(batch, frac_coords=batch.frac_coords, lattices=batch.lattices)
        tt = torch.full((batch.num_graphs,), float(t), device=gen._device)
        _, eps_A_hat, _ = gen.net(cur, A_t, tt)
        A0_hat = (A_t - (1 - abar).sqrt() * eps_A_hat) / abar.sqrt()
        pred_elems.update((A0_hat.argmax(-1) + 1).tolist())
    return pred_elems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_raw/pretrain.jsonl")
    ap.add_argument("--k", type=int, default=16, help="# fixed crystals")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    crystals = []
    for r in load_structures(args.manifest):
        crystals.append(r.to_crystal())
        if len(crystals) >= args.k:
            break
    true_elems = sorted({z for c in crystals for z in c.atom_types.tolist()})
    print(f"{len(crystals)} fixed crystals, {len(true_elems)} distinct true "
          f"elements: {true_elems}\n")

    gen = CrystalGenerator(config_with_lattice_stats(args.manifest), device=args.device)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-4)

    hist = {"coord": [], "lattice": [], "type": []}
    for step in range(args.steps):
        out = gen.train_step(crystals)
        opt.zero_grad()
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
        opt.step()
        for k in hist:
            hist[k].append(float(out[k]))
        if step % max(args.steps // 8, 1) == 0 or step == args.steps - 1:
            pe = atom_type_probe(gen, crystals)
            ndist = len(pe)
            top = pe.most_common(1)[0]
            print(f"step {step:4d}  c/l/t = {out['coord']:.4f}/"
                  f"{out['lattice']:.4f}/{out['type']:.4f}   "
                  f"type-probe: {ndist} distinct elems, "
                  f"top Z={top[0]} ({top[1] / sum(pe.values()):.0%})")

    def drop(k):
        a = sum(hist[k][:5]) / 5
        b = sum(hist[k][-5:]) / 5
        return a, b

    print()
    fails = []
    for k in ("coord", "lattice", "type"):
        a, b = drop(k)
        dec = b < a
        print(f"{k:8s}: {a:.4f} -> {b:.4f}  {'decreased' if dec else 'NO DECREASE'}")
        if not dec:
            fails.append(f"{k} loss did not decrease")

    pe = atom_type_probe(gen, crystals)
    _, type_end = drop("type")
    n_pred = len(pe)
    collapsed = n_pred <= max(2, len(true_elems) // 5)
    print(f"\nfinal type-probe: {n_pred} distinct predicted elements "
          f"(true set has {len(true_elems)})")
    if type_end < 0.05 and collapsed:
        fails.append(f"type loss near zero ({type_end:.4f}) WHILE predictions "
                     f"collapsed to {n_pred} element(s) - the collapse signature")
    elif collapsed:
        fails.append(f"type predictions collapsed to {n_pred} element(s)")

    print()
    if fails:
        print("GATE 3 FAIL:")
        for f in fails:
            print("  -", f)
    else:
        print("GATE 3 PASS - all channels learn, no atom-type collapse signature")


if __name__ == "__main__":
    main()
