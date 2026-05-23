"""CrystalGenerator: ties the EGNN denoiser to the diffusion process.

Provides the public surface step 2 of REBUILD_PLAN.md needs:
  * train_step(crystals)  -> loss dict   (Algorithm 1)
  * sample(num_atoms)     -> list[Crystal] (Algorithm 2)
  * configure_optimizer() -> (Adam, ReduceLROnPlateau)   (Table S.2)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml

from ..data.batch import CrystalBatch, collate
from ..data.representation import Crystal, atomic_numbers_to_onehot
from .diffusion import DiffusionProcess
from .egnn import EGNNDenoiser

_DEFAULT_CFG = Path(__file__).resolve().parents[1] / "configs" / "generator.yaml"


def load_config(path: str | Path = _DEFAULT_CFG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_with_lattice_stats(
    manifest: str, cfg: dict | None = None, max_n: int = 8000
) -> dict:
    """Config with the lattice channel's dataset mean/std filled in.

    The lattice x0-channel diffuses in a normalized space; the stats are
    computed once from the data and travel inside cfg (so they are saved in
    the checkpoint and restored at sampling). See DiffusionProcess.
    """
    import copy

    from ..data.datasets import compute_lattice_stats, load_structures

    cfg = copy.deepcopy(cfg) if cfg is not None else load_config()
    mean, std = compute_lattice_stats(load_structures(manifest), max_n)
    cfg["diffusion"]["lattice_mean"] = mean.tolist()
    cfg["diffusion"]["lattice_std"] = std.tolist()
    return cfg


class CrystalGenerator(nn.Module):
    def __init__(self, cfg: dict | None = None, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg or load_config()
        m, dt = self.cfg["model"], self.cfg["data"]
        self.num_atom_types = int(m["num_atom_types"])
        self.cutoff = float(dt["neighbor_cutoff_radius"])
        self.max_nbr = int(dt["max_neighbors_per_atom"])
        self.graph_mode = str(dt.get("graph_construction", "complete"))

        self.net = EGNNDenoiser(
            hidden_dim=int(m["hidden_dim"]),
            num_layers=int(m["num_gnn_layers"]),
            num_atom_types=self.num_atom_types,
            fourier_freqs=int(m["fourier_freqs"]),
            time_embed_dim=int(m["time_embed_dim"]),
            lattice_x0_bound=float(m.get("lattice_x0_bound", 8.0)),
        )
        self.diffusion = DiffusionProcess(self.cfg, device=device)
        self.to(device)
        self._device = torch.device(device)

    # -- Algorithm 1 -----------------------------------------------------------
    def loss_on_batch(self, batch: "CrystalBatch") -> dict[str, torch.Tensor]:
        """Algorithm 1 on a pre-collated batch (from a DataLoader)."""
        batch = batch.to(self._device)
        onehot = atomic_numbers_to_onehot(batch.atom_types, self.num_atom_types)
        return self.diffusion.training_loss(self.net, batch, onehot)

    def train_step(self, crystals: list[Crystal]) -> dict[str, torch.Tensor]:
        return self.loss_on_batch(
            collate(crystals, self.cutoff, self.max_nbr, self.graph_mode)
        )

    # -- Algorithm 2 -----------------------------------------------------------
    @torch.no_grad()
    def sample(self, num_atoms: list[int], gamma: float = 0.5,
               stats: dict | None = None) -> list[Crystal]:
        # template batch only supplies graph structure; values get overwritten
        templates = [Crystal.random(n) for n in num_atoms]
        batch = collate(
            templates, self.cutoff, self.max_nbr, self.graph_mode
        ).to(self._device)
        z, frac, lat, bidx = self.diffusion.sample(
            self.net, batch, self.num_atom_types, gamma=gamma, stats=stats
        )
        out: list[Crystal] = []
        for gid in range(len(num_atoms)):
            mask = bidx == gid
            out.append(
                Crystal(
                    atom_types=z[mask].cpu(),
                    frac_coords=frac[mask].cpu(),
                    lattice=lat[gid].cpu(),
                )
            )
        return out

    # -- Table S.2 optimizer ---------------------------------------------------
    def configure_optimizer(self):
        o = self.cfg["optim"]
        opt = torch.optim.Adam(self.parameters(), lr=float(o["base_lr"]))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(o["rlrop_factor"]),
            patience=int(o["rlrop_patience"]),
            min_lr=float(o["min_lr"]),
        )
        return opt, sched
