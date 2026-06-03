"""Plan C — piezoelectric scoring oracle.

Wraps a trained PiezoHead checkpoint and exposes a single method for the
AL loop: `score_relaxed(zlist, frac, lattice) -> float`.

The model was trained on log(eij_max + 0.01); we return |e_max| in C/m^2
(clipped to >= 0). Validation Spearman ~0.72 on the held-out point-group-
stratified split (see checkpoints/piezo_head_history.json).

Design choices:
- One PiezoHead loaded in eval() mode, batched=1 by default. The head is
  ~430k params; per-structure forward is sub-ms on GPU.
- No persistent cache. The candidate stream is small (<= 200 per round)
  and the inference is faster than the CHGNet relax that produced the
  structure in the first place.
- Operates on post-relaxation geometry. Callers should pass `r.relaxed_frac`
  and `r.relaxed_lattice` from `CHGNetOracle.relax_one(...)`.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from ..data.batch import collate
from ..data.representation import Crystal
from ..models.piezo_head import PiezoHead


class PiezoOracle:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
    ):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ck.get("cfg", {})
        self.head = PiezoHead(
            hidden=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 3)),
            fourier_freqs=int(cfg.get("fourier_freqs", 16)),
            dropout=float(cfg.get("dropout", 0.1)),
        ).to(device).eval()
        self.head.load_state_dict(ck["state_dict"])
        self.device = device
        self.target_offset = float(ck.get("target_offset", 0.01))
        self.val_spearman = float(ck.get("val_spearman", float("nan")))
        self.ckpt_path = ckpt_path
        self.ckpt_epoch = int(ck.get("epoch", -1))
        self.source_data = ck.get("source_data", "")

    @torch.no_grad()
    def score_relaxed(
        self,
        zlist: list[int],
        frac: list,
        lattice: list,
    ) -> float:
        """Return predicted |e_max| in C/m^2 (>= 0)."""
        c = Crystal(
            atom_types=torch.tensor(zlist, dtype=torch.long),
            frac_coords=torch.tensor(frac, dtype=torch.float),
            lattice=torch.tensor(lattice, dtype=torch.float),
        )
        batch = collate([c], mode="complete").to(self.device)
        log_pred = float(self.head(batch).item())
        e_max = math.exp(log_pred) - self.target_offset
        return max(e_max, 0.0)

    def config(self) -> dict:
        return {
            "ckpt_path": self.ckpt_path,
            "ckpt_epoch": self.ckpt_epoch,
            "val_spearman": self.val_spearman,
            "target_offset": self.target_offset,
            "source_data": self.source_data,
        }
