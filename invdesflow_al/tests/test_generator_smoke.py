"""Smoke test for step 1 (pretrained crystal generation model).

Uses a tiny config (small net, short diffusion) so it runs on CPU in seconds.
Verifies:
  * Algorithm 1 loss is finite and trainable (drops on a 1-crystal overfit)
  * Algorithm 2 sampling returns valid periodic crystals
    (frac in [0,1), non-degenerate lattice, atom types in 1..K)

Run:  python -m invdesflow_al.tests.test_generator_smoke
"""

from __future__ import annotations

import copy

import torch

from ..models.generator import CrystalGenerator, load_config
from ..data.representation import Crystal


def _tiny_cfg() -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["model"]["hidden_dim"] = 64
    cfg["model"]["num_gnn_layers"] = 3
    cfg["model"]["num_atom_types"] = 12
    cfg["model"]["fourier_freqs"] = 8
    cfg["model"]["time_embed_dim"] = 32
    cfg["diffusion"]["num_steps"] = 100
    return cfg


def main() -> None:
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    gen = CrystalGenerator(cfg, device="cpu")
    opt, sched = gen.configure_optimizer()

    # fixed mini-batch of 2 small crystals (atomic numbers within tiny K)
    g = torch.Generator().manual_seed(1)
    crystals = []
    for n in (3, 4):
        c = Crystal.random(n, generator=g)
        c.atom_types = torch.randint(1, cfg["model"]["num_atom_types"] + 1, (n,))
        crystals.append(c)

    losses = []
    for step in range(60):
        out = gen.train_step(crystals)
        loss = out["total"]
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step(loss.detach())
        losses.append(float(loss))

    print(f"Algorithm 1  loss: {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"(coord/lattice/type = {out['coord']:.3f}/{out['lattice']:.3f}/{out['type']:.3f})")
    assert losses[-1] < losses[0], "overfit loss did not decrease"

    gen.eval()
    samples = gen.sample(num_atoms=[3, 5])
    for i, s in enumerate(samples):
        assert s.frac_coords.min() >= 0.0 and s.frac_coords.max() < 1.0, "frac not wrapped"
        assert torch.det(s.lattice).abs() > 1e-6, "degenerate lattice"
        assert s.atom_types.min() >= 1 and s.atom_types.max() <= cfg["model"]["num_atom_types"]
        print(f"Algorithm 2  sample[{i}]: N={s.num_atoms}  "
              f"|det L|={float(torch.det(s.lattice).abs()):.3f}  "
              f"Z={s.atom_types.tolist()}")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
