# invdesflow_al — Step 1: Pretrained Crystal Generation Model

Clean-room reconstruction from the paper + supplementary (npj Comput. Mater.
2025, 11:364). **Not** derived from the repo's existing code. Implements the
diffusion + EGNN generator: Algorithms 1 (training) & 2 (sampling) with every
hyperparameter from **Supplementary Table S.2**.

## Layout
```
invdesflow_al/
  configs/generator.yaml      # = Table S.2 verbatim (with sourced comments)
  data/representation.py      # M=(A,F,L); frac<->cart; periodic wrap w(.)
  data/graph.py               # periodic radius graph (7.0 A, 20 nbrs)
  data/batch.py               # dependency-free PyG-style batching
  data/datasets.py            # ingest (jsonl/CIF/ASE) + filter + de-dup
  data/diversity.py           # diversity sampler (sec. 6 / Methods)
  data/torch_dataset.py       # lazy Dataset + 96/64/64 DataLoaders
  models/egnn.py              # EGNN denoiser phi: 6 layers, hidden 512, SiLU
  models/diffusion.py         # schedules; Alg.1 loss; Alg.2 PC sampler
  models/generator.py         # train_step / sample / configure_optimizer
  scripts/train_generator.py  # Table S.2 training loop (Adam+RLROP)
  tests/test_generator_smoke.py
```

## What maps to what
| Paper | Code |
|---|---|
| `M = (A, F, L)` | `data/representation.Crystal` |
| `w(.)` periodic wrap | `representation.wrap_frac` |
| Alg. 1 lines 1–14 | `diffusion.DiffusionProcess.training_loss` |
| Alg. 2 lines 1–20 | `diffusion.DiffusionProcess.sample` |
| `phi(L_t,A_t,F_t,N,t)` | `egnn.EGNNDenoiser.forward` |
| DDPM for L, A | Gaussian schedule in `diffusion` |
| score-matching + wrapped normal for F | `_wrapped_score` (torus theta-sum) |
| loss weights 1/1/20 | `configs/generator.yaml: loss_weights` |
| Adam 1e-4 + ReduceLROnPlateau(0.6,30) | `generator.configure_optimizer` |

## Run the smoke test
```
/home/satya/anaconda3/envs/py39/bin/python -m invdesflow_al.tests.test_generator_smoke
```
Verifies Alg. 1 loss is finite & decreases on a tiny overfit, and Alg. 2
returns valid periodic crystals (wrapped frac, non-degenerate lattice, valid
atom types). It uses a shrunk net/schedule for CPU speed.

## Pretraining data pipeline (REBUILD_PLAN.md sec. 6)

Paper corpus = Alex-MP-20 + GNoME, diversity-sampled. Build the manifest then
train:

```
# 1. assemble: ingest -> filter(<=20 atoms) -> de-dup -> [exclude CSP sets]
#    -> diversity sample -> JSONL manifest
.../py39/bin/python -m invdesflow_al.scripts.build_pretrain_dataset \
    --inputs alex_mp_20.jsonl:alex-mp-20 gnome/cifs:gnome \
    --exclude perov5.jsonl mp20.jsonl mpts52.jsonl \
    --target-size 800000 --out pretrain_manifest.jsonl

# 2. pretrain (Table S.2: Adam 1e-4, RLROP 0.6/30, batch 96/64/64)
.../py39/bin/python -m invdesflow_al.scripts.train_generator \
    --manifest pretrain_manifest.jsonl --device cuda
```

Fine-tune variants (Fig S.7): add `--e-form-max -0.5` (low formation) or
`--e-hull-max 0.05` (low E_hull) to step 1.

**Inputs**: `.jsonl`/`.json` (dependency-free, self-describing — also the
manifest format), a directory of CIF/POSCAR (`pymatgen`), or ASE `.db/.traj`
(`ase`). JSONL record schema is documented at the top of `data/datasets.py`.

Pipeline smoke test (no heavy deps):
```
.../py39/bin/python -m invdesflow_al.tests.test_data_pipeline_smoke
```

## Documented assumptions (paper underspecified — see REBUILD_PLAN.md sec. 11)
- **β-schedule shape**: cosine (Nichol & Dhariwal). Configurable.
- **λ_t** in the coord-loss target: `λ_t = σ_t` (DiffCSP convention).
- **"CrystalNN with lattice scaling"**: reconstructed as the periodic radius
  graph EGNN actually runs on (7.0 Å / 20 nbrs from Table S.2); a true
  CrystalNN backend can be slotted into `data/graph.py`.
- **EGNN coordinate update for periodic F**: relative *fractional* differences
  embedded with sin/cos (periodic + translation-invariant), gated by scalars
  of the squared cartesian distance — the periodic analogue of EGNN.

These are isolated behind config flags and do not affect the algorithm logic.
```
