# InvDesAL — Implementation

A clean-room reconstruction of **InvDesFlow-AL** (Han et al., *npj
Computational Materials* 2025, 11:364, [doi:10.1038/s41524-025-01830-z](https://doi.org/10.1038/s41524-025-01830-z)),
built from the paper and supplementary alone — the upstream repo's code was
intentionally not used as a reference.

Authoritative log of what was built, why, what failed, and what fixed it:
**[PROGRESS.md](PROGRESS.md)** (read this first). The architectural plan
extracted from the paper is in **[REBUILD_PLAN.md](REBUILD_PLAN.md)**.

## Current status (May 2026)

Step 1 of the paper — the **pretrained crystal generation model** (diffusion +
EGNN, Algorithms 1 & 2 from the paper, Table S.2 hyperparameters) — is
working end-to-end on real Alex-MP-20 + GNoME data:

| Model | Data | Unique rate @ N=512–1k | Sane lattice fraction | Lattice vpa max (Å³) |
|---|---|---|---|---|
| `gen_1k.ckpt` (Entry 6) | 1 k diversity-sampled | **0.991** (paper Fig S.4: 0.992 @ N=1000) | 0.997 | 4 029 |
| `gen_10k.ckpt` (Entry 7, eps-A) | 10 k | 0.836 | 0.887 (after a sampler-side A clamp; 0.82 NaN-rate before) | 10 725 |
| **`gen_10k_ax0.ckpt`** (Entry 8, **A x₀-prediction**) | 10 k | **0.902** | **1.00** | **59.7** |

The journey from a totally collapsed initial overnight run (unique rate
**0.065**, every sample identical C₈ in a 10¹² Å³ cell) to a working
generator went through a 7-gate falsifiable debug sequence (Entries 4–5) that
isolated three concrete bugs:

1. **F sampler math** — the wrapped-coordinate Langevin corrector normalized
   by the smallest σ, reaching ~10⁴; even with the *exact* score it kicked
   F to random every step. Fixed in
   [`invdesflow_al/models/diffusion.py`](invdesflow_al/models/diffusion.py).
2. **Frozen sampling graph** — the periodic radius graph was built once from
   random templates and frozen; the trained model only ever saw graphs built
   from clean geometry. Replaced with a geometry-independent **complete
   graph** + minimum-image convention
   ([`invdesflow_al/data/graph.py`](invdesflow_al/data/graph.py),
   [`invdesflow_al/models/egnn.py`](invdesflow_al/models/egnn.py)).
3. **Lattice channel** — raw 3×3 DDPM + ±10⁴ state clamp masked divergence.
   Replaced with statistical normalization + **x₀-prediction** + a bounded
   `B·tanh(raw/B)` head. The same lesson then forced a similar fix for the
   atom-type channel: A is now **x₀-prediction with a softmax-bounded head**
   (Entry 8) — A no longer saturates during sampling (max\|A\| stays ≲ 5
   throughout the reverse process, the lattice tail dropped from 10 725 → 60
   Å³, sane fraction went 0.89 → 1.00).

## Layout

```
invdesflow_al/
  configs/generator.yaml          # = paper Table S.2, source-commented
  data/                           # representation, periodic + complete graph,
                                  # ingest (jsonl / cif / ase), filter, dedup,
                                  # diversity sampler, lazy DataLoaders
  models/                         # EGNNDenoiser, DiffusionProcess (Alg. 1 & 2),
                                  # CrystalGenerator wrapper
  scripts/                        # convert_datasets, build_manifest,
                                  # train_generator, eval_unique_rate,
                                  # debug_overfit_one, debug_oracle_sampler,
                                  # debug_graph_compare, debug_tiny_dataset,
                                  # debug_data/graph/forward_*, run_*.sh
  tests/test_generator_smoke.py
evals/   # JSON eval results referenced by PROGRESS.md
logs/    # training / eval logs referenced by PROGRESS.md
PROGRESS.md
REBUILD_PLAN.md
plot_eval_quick.py + eval_quick_compare.png
```

## What is *not* in the repo (and why)

| Excluded | Why | How to obtain / reproduce |
|---|---|---|
| `*.ckpt` checkpoints (~50–160 MB each) | too large | retrain via `scripts/train_generator.py` per PROGRESS.md §Reproducibility |
| `data_raw/` (~3 GB: Alex-MP-20 parquet, GNoME zip, derived JSONLs) | large + redistributable from source | scripts in `invdesflow_al/scripts/convert_datasets.py` and `build_manifest.py` rebuild them; sources listed in PROGRESS.md |
| Paper / supplementary PDFs | copyrighted (npj) | open-access: <https://doi.org/10.1038/s41524-025-01830-z> |
| Upstream's original code (`Crystal-structure-prediction/`, `FormEGNN/`, `Functional-materials-generation/`, `SuperconGNN/`, `fig/`) | not part of this clean-room rebuild | see upstream <https://github.com/xqh19970407/InvDesFlow-AL> |

## How to run

See **[PROGRESS.md → Reproducibility](PROGRESS.md#reproducibility--environment--how-to-run)**
for the exact Python interpreter path, package versions (`torch 2.6.0+cu126`,
`pymatgen 2024.8.9`, …), dataset sources, and the verbatim command sequence
that produced every number in this log.

## Citation

If you use this implementation, cite the original paper:
```bibtex
@article{InvDesFlow-AL,
  author  = {Xiao-Qi Han and Peng-Jie Guo and Ze-Feng Gao and Hao Sun and Zhong-Yi Lu},
  title   = {InvDesFlow-AL: active learning-based workflow for inverse design of functional materials},
  journal = {npj Computational Materials},
  year    = {2025},
  volume  = {11},
  pages   = {364},
  doi     = {10.1038/s41524-025-01830-z}
}
```
