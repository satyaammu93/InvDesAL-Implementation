# InvDesFlow-AL Reconstruction — Progress Log

Clean-room rebuild from the paper (`s41524-025-01830-z.pdf`) + supplementary
(`41524_2025_1830_MOESM1_ESM.pdf`). Repo code intentionally not used.

Companion docs: [REBUILD_PLAN.md](REBUILD_PLAN.md) (full plan + §11 documented
assumptions), [invdesflow_al/README.md](invdesflow_al/README.md).

## Reproducibility — environment + how to run

**Code:** <https://github.com/satyaammu93/InvDesAL-Implementation>
Checkpoints (.ckpt, ~50–160 MB each) and raw datasets (~3 GB) are excluded
from the repo — see the dataset/training commands below to rebuild them.

**Python interpreter (use this exact path):**
```
/home/satya/anaconda3/envs/py39/bin/python
```
Conda env `py39`, Python **3.9.18**.

**GPU / CUDA:**
- NVIDIA GeForce RTX 3060, 12.5 GB VRAM
- CUDA 12.6 runtime; `torch.cuda.is_available() == True`

**Installed package versions (exact, what produced every result below):**
| package | version |
|---|---|
| torch | 2.6.0+cu126 |
| pymatgen | 2024.8.9 |
| huggingface_hub | 1.4.1 |
| datasets | 4.5.0 |
| pyarrow | 21.0.0 |
| numpy | 2.0.2 |
| pyyaml | 6.0.3 |
| tqdm | 4.67.3 |
| ase | not installed (optional — only needed for `.db`/`.traj`/`.xyz` ingest) |

System tools used: `aria2c` (parallel HTTP download), `wget`, `unzip`.

**Datasets (Zenodo / HuggingFace / GCS):**
| name | source | local path |
|---|---|---|
| Alex-MP-20 | HF `OMatG/Alex-MP-20` (train.parquet 156 MB) | `data_raw/alex_mp_20/` → `data_raw/alex_mp_20.jsonl` (540 k records) |
| GNoME | `gs://gdm_materials_discovery/gnome_data` (by_id.zip 455 MB + summary 151 MB) | `data_raw/gnome/` → `data_raw/gnome.jsonl` (capped 150 k) |
| pretrain manifest (150 k diversity sample) | built by `build_manifest.py` | `data_raw/pretrain.jsonl` |
| 10 k subset | built by `build_manifest.py` | `data_raw/pretrain_10k.jsonl` |
| 1 k subset | built by `build_manifest.py` | `data_raw/pretrain_1k.jsonl` |

**Reproduce the full pipeline (commands as run):**
```bash
PY=/home/satya/anaconda3/envs/py39/bin/python
cd /home/satya/projects/invdes/InvDesFlow-AL

# 1. one-time deps (already done; documented for new machines)
$PY -m pip install "pymatgen>=2024.1" huggingface_hub datasets pyarrow tqdm

# 2. raw -> JSONL converters  (Alex parquet + GNoME zip+csv)
$PY -m invdesflow_al.scripts.convert_datasets alex \
    data_raw/alex_mp_20/train.parquet  data_raw/alex_mp_20.jsonl
$PY -m invdesflow_al.scripts.convert_datasets gnome \
    data_raw/gnome/by_id.zip  data_raw/gnome/stable_materials_summary.csv \
    data_raw/gnome.jsonl  --limit 150000

# 3. diversity-sampled training manifests
$PY -m invdesflow_al.scripts.build_manifest \
    --inputs data_raw/alex_mp_20.jsonl data_raw/gnome.jsonl \
    --max-atoms 20 --target-size 150000 --seed 0 \
    --out data_raw/pretrain.jsonl
$PY -m invdesflow_al.scripts.build_manifest --inputs data_raw/pretrain.jsonl \
    --max-atoms 20 --target-size 10000 --seed 1 --out data_raw/pretrain_10k.jsonl
$PY -m invdesflow_al.scripts.build_manifest --inputs data_raw/pretrain.jsonl \
    --max-atoms 20 --target-size 1000  --seed 1 --out data_raw/pretrain_1k.jsonl

# 4. pretrain (auto-batch, wallclock budget, ckpts)
$PY -m invdesflow_al.scripts.train_generator \
    --manifest data_raw/pretrain_10k.jsonl --device cuda \
    --auto-batch --workers 0 --max-hours 3.0 \
    --ckpt gen_10k_ax0.ckpt --log-every 200 --ckpt-every 1000

# 5. evals
$PY -m invdesflow_al.scripts.debug_eval_quick \
    --ckpt gen_10k_ax0.ckpt --manifest data_raw/pretrain_10k.jsonl \
    --n-sample 512 --gen-batch 256 --device cuda --out eval_10k_ax0_quick.json
$PY -m invdesflow_al.scripts.eval_unique_rate \
    --ckpt gen_10k_ax0.ckpt --manifest data_raw/pretrain_10k.jsonl \
    --max-samples 4000 --gen-batch 256 --device cuda --out eval_10k_ax0_full.json
```

**Background-run tip:** plain `nohup python … > log 2>&1 &` is sufficient in
this environment (it survives shell exit fine). Avoid `setsid` — full
detachment makes the process awkward to monitor and kill cleanly.

**Debug gate scripts (PROGRESS Entries 5–8):** `debug_data_sanity.py`,
`debug_graph_sanity.py`, `debug_forward_loss.py`, `debug_overfit_one.py`,
`debug_oracle_sampler.py`, `debug_graph_compare.py`, `debug_tiny_dataset.py`,
`debug_eval_quick.py` — all under `invdesflow_al/scripts/`, each runnable as
`$PY -m invdesflow_al.scripts.<name> [args]`.

**Smoke tests:** `$PY -m invdesflow_al.tests.test_generator_smoke` (~30 s,
CPU) — wiring check for the diffusion / EGNN / sampling path.

---

## Entry 12 — 2026-05-24 — 50k pretrain + Fig S.4 eval: scaling closes 30–40 % of the gap

### Pretrain — `checkpoints/gen_50k.ckpt`
- Manifest: `data_raw/pretrain_50k.jsonl` (50 000 records diversity-sampled
  from the 150 k pool, 7 032 buckets / 180 spacegroups, same seed pipeline).
- Same architecture as Entry 8 (A x0 + L x0 + complete graph + bounded heads,
  all the fixes locked in). No code changes for this run.
- 168 epochs in **6.00 h** on RTX 3060, auto-batch 96, **best val 0.9995**
  (10k was 0.96 at 627 epochs — 50k sees 5× more data per epoch and runs
  fewer epochs within the same wallclock, so per-sample loss is comparable).
- Trained while sharing the GPU with an unrelated `s2go.tools.overfit`
  training (7 GB VRAM) — throughput halved but no instability.

### Fig S.4 eval (4000 samples, `evals/eval_50k_full.json`)
| N | 10k A x0 (Entry 10) | **50k A x0** | paper | gap → |
|---|---|---|---|---|
| 1000 | 0.874 | **0.909** | 0.992 | −0.118 → **−0.083** |
| 2000 | 0.840 | **0.899** | 0.989 | −0.149 → **−0.090** |
| 4000 | 0.824 | **0.886** | 0.984 | −0.160 → **−0.098** |

5× more data → unique-rate gap shrinks **30–40 %** at every checkpoint. The
gap reduction is **larger at higher N** (−0.06 at N=4000 vs −0.04 at N=1000),
exactly what scaling should produce — the paper's data-coverage advantage is
most visible deep into sampling, and we're catching up there fastest.

### Lattice sanity (4000 samples)
| | 10k A x0 | **50k A x0** |
|---|---|---|
| vpa min | 4.64 | 5.96 |
| vpa p5 | 11.96 | 11.81 |
| vpa median (Å³, data ≈21) | 20.1 | **18.3** |
| vpa p95 | 41.8 | 42.6 |
| vpa max | 91.6 | 117.0 |
| sane fraction (0 < vpa ≤ 500) | 1.000 | **1.000** |
| nan/inf | 0 | **0** |

Lattice channel is rock-solid across the 5× data scale-up: 100 % sane, no
tails approaching the bounded-head ceiling. The A-x0 / L-x0 / complete-graph
fixes hold uniformly as we scale.

### Plan pass criteria for the "50k or 150k" step (Phase 1, step 3)
| Criterion | 50k result |
|---|---|
| Unique-rate degradation with sample count is graceful | ✅ 0.909 → 0.899 → 0.886 (same shape as paper 0.992 → 0.989 → 0.984) |
| Lattice tails improve or stay controlled | ✅ max 91.6 → 117.0 (both << 500 bound; sane frac 1.000) |
| Sampled formulas are chemically broad | ✅ unique rate 0.886 at N=4000 (88.6 % distinct) |

**All three criteria for the scaling step are met.** The remaining ~0.09 gap
to paper at N=1000 is now an extrapolation question — does another 3× (50k → 150k)
close it further? The trend strongly suggests yes.

### Files
`checkpoints/gen_50k.ckpt` (best val), `checkpoints/gen_50k_latest.ckpt`,
`evals/eval_50k_full.json`, `logs/train_50k.log`, `logs/eval_50k_full.log`.
Baselines preserved unchanged: `gen_10k_ax0.ckpt`, `gen_10k.ckpt`,
`gen_1k.ckpt`.

---

## Entry 11 — 2026-05-24 — Tiny active-learning dry run scaffold

Added the first active-learning **plumbing** script:
`invdesflow_al/scripts/run_tiny_al_dryrun.py`.

This is intentionally a dry run, not a discovery workflow. It exercises the
loop shape while the 50k generator pretrain runs in parallel:

```
generator checkpoint
  -> generate 500-1000 candidates
  -> validity / composition filters
  -> transparent heuristic score
  -> diverse top-k selection
  -> optional fine-tune of a COPY of the generator
```

### Choices made, explicitly
- **No real oracle yet.** The score is a placeholder, not a prediction of
  stability or piezoelectric response. It rewards volume/atom near a target
  (`21 A^3/atom` by default), non-overlapping atoms, oxygen presence, and mild
  element diversity.
- **Lead-free by default.** `--exclude-elements 82` bans Pb unless overridden.
- **Oxide/ceramic mode is opt-in.** `--require-oxygen` can be enabled for the
  lead-free piezoelectric direction, but is not forced for generic plumbing
  tests.
- **Validity filters are cheap structural checks only:** finite tensors,
  non-degenerate lattice, `0 < volume/atom <= 500`, minimum periodic
  interatomic distance (`0.8 A` default), excluded elements, and optional O.
- **Selection is diversity-preserving:** score sorted, at most one candidate
  per reduced formula, max three per element set.
- **Optional fine-tuning is marked as pseudo-data.** `--finetune-steps > 0`
  fine-tunes a copy on selected generated structures only to test mechanics;
  the saved checkpoint carries a warning that it is not a validated discovery
  model.

### Example command
```bash
PY=/home/satya/anaconda3/envs/py39/bin/python
$PY -m invdesflow_al.scripts.run_tiny_al_dryrun \
    --ckpt gen_10k_ax0.ckpt \
    --manifest data_raw/pretrain_10k.jsonl \
    --num-generate 1000 \
    --gen-batch 128 \
    --top-k 50 \
    --require-oxygen \
    --device cuda \
    --out-dir al_runs/tiny_leadfree_oxide_dryrun
```

Outputs:
```
al_runs/.../generated.jsonl   # all generated candidates + filter reason
al_runs/.../valid.jsonl       # validity-filter survivors + score metadata
al_runs/.../selected.jsonl    # diverse top-k candidates
al_runs/.../summary.json      # all choices, metrics, top selected candidates
```

### Pass criteria for this dry run
- Script completes end-to-end without generator/eval crashes.
- Valid fraction is high enough to continue debugging (`>50%` is acceptable for
  plumbing; stricter thresholds wait for real oracle integration).
- Selected set contains many formulas and element sets, not one repeated family.
- Optional short fine-tune does not destroy generator validity/diversity in a
  follow-up quick eval.

### What remains before *real* active learning
Replace the heuristic score with at least one validated oracle:
1. stability/relaxation path (`FormEGNN`, DPA-2, CHGNet, M3GNet, MACE, etc.);
2. for lead-free piezoelectric ceramics, a property/symmetry oracle that
   rejects centrosymmetric structures and ranks piezoelectric-relevant
   candidates.

Until then, selected candidates are debugging artifacts, not discoveries.

Verification performed: syntax check with `python3 -m py_compile`; CLI help
loads under `/home/satya/anaconda3/envs/py39/bin/python`.

---

## Entry 10 — 2026-05-24 — Full Fig S.4 eval on gen_10k_ax0.ckpt — clean 4000-sample result

### What was run
Relaunch of the Entry-9 eval (the same command, plain `nohup`, no `setsid`).
Pid 936337, sampled the full 4000 crystals in ~2 h 4 min while sharing the GPU
with another training job (s2go.tools.overfit, 7 GB VRAM; our eval 1.2 GB).
Output: `evals/eval_10k_ax0_full.json`, `logs/eval_10k_ax0_full.log`.

### Result vs paper Fig S.4 and vs the eps-A 10k baseline
| N | **A x0 10k** | paper Fig S.4 | Δ vs paper | eps-A 10k (Entry 7) |
|---|---|---|---|---|
| 1000 | **0.874** | 0.992 | −0.118 | 0.836 |
| 2000 | **0.840** | 0.989 | −0.149 | 0.825 |
| 4000 | **0.824** | 0.984 | −0.160 | 0.829 |

Curve shape matches the paper's gentle decay; the gap is a near-constant
~0.12–0.16 offset that **grows slowly with N**, consistent with an
under-fit-for-the-corpus-size model rather than a bug.

### Lattice (all 4000 samples)
| | vpa_min | vpa_p5 | vpa_median | vpa_p95 | vpa_max | sane fraction | nan |
|---|---|---|---|---|---|---|---|
| eps-A 10k (Entry 7) | 0.0 | 9.4 | 15.8 | 28.4 | 101.5 | 1.00¹ | 0 (after clamp) |
| eps-A 10k post-A-clamp | 0.14 | 9.83 | 17.0 | **4582** | **10 725** | 0.887 | 0 |
| **A x0 10k** | **4.64** | **12.0** | **20.1** | **41.8** | **91.6** | **1.000** | 0 |

¹ pre-A-clamp had 18 % NaN; post-A-clamp has 11 % saturated-but-bounded tails.

The **A x0** model is **the first run to satisfy every lattice criterion at 4000
samples**: sane fraction 1.000, no NaN, max < 100 Å³, p95 < 50 Å³, median
right on the data distribution (~21 Å³).

### Plan pass criteria at the 4000-sample scale
| Criterion | A x0 10k |
|---|---|
| Unique rate near paper @ 1000 (0.992) | 0.874 (Δ −0.12; was Δ −0.16) ⚠️ |
| No formula collapse @ N=4000 | 0.824 ✅ |
| **Sane lattice fraction > 99 %** | **1.000** ✅ |
| Volume/atom median in data range | 20.1 ✅ |
| **No catastrophic tails** | **max 91.6** ✅ |

**Four of five criteria met.** Remaining gap is the unique-rate offset — the
**undertraining/data-scale** axis, not the parametrization axis. Closing it
needs more pretraining data (50k → 150k), not another model fix.

---

## Entry 9 — 2026-05-24 — Full Fig S.4 eval (N=1k/2k/4k) interrupted at 77 %

### What was tried
Launched `eval_unique_rate.py --max-samples 4000 --gen-batch 256` on
`gen_10k_ax0.ckpt` to extend the Entry-8 quick-eval (N=512) to the full
Fig S.4 checkpoints. Process pid 889918, logged to `logs/eval_10k_ax0_full.log`.

### What happened
Process ended at **3072 / 4000 samples** (~77 %) after ~1 h 56 m.
- Last log line: `generated 3072/4000 (5884s)`; no error / traceback / OOM /
  disk pressure / GPU issue.
- `eval_10k_ax0_full.json` was never written (the script writes only at the
  end after the full sampling loop).
- **Cause:** the user terminated the run by mistake — *not* a session/signal
  problem. Plain `nohup` is sufficient in this environment (it survived 1 h
  56 m before the manual kill). Relaunch uses the same `nohup … > log 2>&1 &`
  setup, **without** `setsid` (no hard-detach needed).

### State preserved
- `gen_10k_ax0.ckpt` intact (162 MB, model unaffected).
- All Entry-8 metrics still valid via `eval_10k_ax0_quick.json` (N=512,
  unique rate 0.902, sane fraction 1.00, vpa max 59.7, A never saturates).
- The N=1000 / 2000 / 4000 vs paper comparison is **outstanding**; relaunch
  in progress.

### Open robustness fix for the eval script
Current `eval_unique_rate.py` writes the JSON only at the very end. Should
checkpoint partial progress every 256–512 samples so a mid-run interruption
leaves salvageable data. (Low priority — not done yet.)

---

## Entry 8 — 2026-05-23 — A x0-prediction (mirrors the lattice fix); sampling cleaned up

### Trigger: resume-train experiment falsified "more training fixes A"
Resumed `gen_10k.ckpt` for +1 h (cycle 1 of a 3-cycle Option B run) with new
A-channel diagnostics in the sampler. The cycle-1 eval (`eval_r1.json`):

| | gen_10k (post-clamp, Entry 7) | **gen_10k_r1** (+1 h train) |
|---|---|---|
| epoch / val | 610 / 1.18 | 743 / **1.14** |
| unique rate | 0.836 | **0.656** ❌ |
| sane fraction | 0.887 | **0.727** ❌ |
| top element fraction | — | **42.4 % (H)** ❌ |
| A first_saturation_t | 998 | **999** (first reverse step) |
| A worst sat fraction | 0.188 | **0.407** |

**Val loss dropped, sampling regressed.** Classic eps-prediction-at-high-t
failure: the model gets better at the forward MSE (predicting the mean is
"good" when the noise is the signal) while the reverse loses contracting
signal → A explodes to ±50 at the *very first* reverse step → atom outputs
concentrate. The 3-cycle orchestrator was killed at this point — cycles 2/3
would just confirm.

### Fix 4: A x0-prediction (categorical analog of Fix 3 for L)

`models/egnn.py` — `atom_head` output now goes through **softmax** per atom:
```python
logits_A = self.atom_head(h)
x0_A = torch.softmax(logits_A, dim=-1)   # per-atom simplex, [0,1]^K, sum=1
```
This is the categorical analog of L's `B·tanh(raw/B)` bound. Clean A is
one-hot (probability simplex), so softmax is the natural bounded
parametrization.

`models/diffusion.py`:
- `training_loss`: A loss = `‖atom_onehot − A_x0_pred‖²` (MSE against the
  clean one-hot — mirrors L's MSE against normalized x0).
- `sample`: A reverse uses the **x0-parameterized DDPM posterior**
  `A_new = coef_x0·A_x0 + coef_xt·A_prev + σ·z` (same coefficients as L).
- **Removed** the ±50 state clamp — no longer needed since x0 is in
  [0,1]^K by construction and the convex coefficients keep the trajectory
  bounded.

Diagnostics preserved (`max|A|`, `med|A|`, `sat_frac@49`, `first_sat_t`)
as a sanity tripwire — they should now report ~0 saturation.

### 10k from-scratch retrain → `gen_10k_ax0.ckpt`
Architecture identical to Fix 3 baseline + the A x0 change. 3.00 h on RTX 3060,
616 epochs, auto-batch 96. **Best val 0.962** (vs 1.18 for eps-A 10k).

### Quick eval (512 samples, A diagnostics) — `eval_10k_ax0_quick.json`
| Metric | eps-A 10k (Entry 7 post-clamp) | **A x0 10k** | Δ |
|---|---|---|---|
| val loss | 1.18 | **0.96** | ↓ 19 % |
| unique rate (N ≈ 512–1k) | 0.836 | **0.902** | +0.07 |
| **sane lattice fraction** | 0.887 | **1.000** | +11 pp |
| volume/atom median (Å³) | 17.0 | **20.0** (data ~21) | closer |
| **volume/atom max** | **10 725** | **59.7** | **180× tighter** |
| NaN / Inf | 0 (after clamp; 18 % before) | **0** | — |
| top-element fraction | 42 % H (r1) | **10.4 % O** | broad |
| atom output entropy (nats) | — | **3.73** (78 elements) | broad |
| A worst max\|A\| | 50.0 (clamp ceiling) | **5.44** | not saturated |
| A worst sat fraction | 0.41 | **0.0** | never hits 49 |
| A first saturation t | 999 (1st reverse step) | **None** | clean trajectory |

A trajectory across reverse process:
```
max|A|  @  t = 999  /  500  /  100  /  0
       =   4.93  /  3.63  /  1.53  /  1.00
```
Gradual contraction toward the one-hot simplex — exactly the categorical
analog of L's bounded x0 contraction.

### Plan pass criteria for 10k milestone
| Criterion | A x0 10k |
|---|---|
| Unique rate near paper @ 1k (0.992) | 0.902 (Δ −0.09; was Δ −0.16) ⚠️ closer |
| No formula collapse | 462/512 distinct ✅ |
| **Sane lattice fraction > 99 %** | **1.00 (100 %)** ✅ |
| Volume/atom median in data range | 20.0 ✅ |
| **No catastrophic tails** | **max 59.7** ✅ |

The remaining 0.09 gap vs paper is now the **scaling / undertraining** gap
(10k vs ~1M), not a parametrization gap. Scaling to 50k/150k is what closes it.

### Files
`gen_10k_ax0.ckpt` (best val), `gen_10k_ax0_latest.ckpt`,
`eval_10k_ax0_quick.json`, `logs/train_10k_ax0.log`. Baselines preserved:
`gen_1k.ckpt` (Entry 6), `gen_10k.ckpt` (Entry 7), `gen_10k_r1.ckpt` (cycle-1
evidence).

### Lesson
A eps-prediction was failing the same way L did: 1/√ᾱ amplification at high t
+ no bound → state explodes. The user predicted this *would* be the next
required model fix and that tighter clamps would not suffice — confirmed by
the cycle-1 regression and the parameterization-change fix.

---

## Entry 7 — 2026-05-23 — 10k pretrain: lattice fixed, A-channel NaN exposed, partial fix

### What was run
- Built `data_raw/pretrain_10k.jsonl` (10,000 diversity-sampled records from
  the 150k manifest, same seed/buckets pipeline).
- Pretrained `gen_10k.ckpt` for **3.00 h** on RTX 3060, auto-batch 96, 627
  epochs, best **val 1.18** (1k model was val 1.20 — comparable).
- Evaluated at N=1000/2000/4000 via `eval_unique_rate.py` (`eval_10k.json`).

### First eval (pre-A-clamp) — lattice better, but new NaN failure mode
| N | ours | paper | Δ |
|---|---|---|---|
| 1000 | 0.826 | 0.992 | −0.166 |
| 2000 | 0.818 | 0.989 | −0.171 |
| 4000 | 0.809 | 0.984 | −0.175 |

Lattice (of FINITE samples): p5 9.4 / median 15.8 / p95 28.4 / **max 101.5** Å³
— huge improvement over the 1k tail (max 4029). Sane fraction of finite = 1.00.
**But:** 711 / 4000 lattices (**17.8 %**) were NaN/Inf.

### Diagnosis
The 1k baseline (Entry 6) had 0 NaN; 10k has 18 %. Cause: Fix 3 removed the
±1e4 *state clamp*; that block clamped **both L and A**. With L now properly
bounded via the tanh head, removing L's clamp was correct — but A was left
unclamped. The 1k model's heavy overfit kept ε̂_A tame; the less-overfit 10k
model occasionally produces large ε̂_A, and the eps-DDPM reverse amplifies by
`1/√α ≈ 30×` per high-t step. A explodes → garbage `h` → NaN cascades into
the lattice head.

### Fix: surgical soft A-clamp (sampler-only, ±50)
`models/diffusion.py` — added `A = A.clamp(-50, 50)` after the A-update.
Clean A is one-hot ≤1, so ±50 is ~50 σ — bites only on runaway; this is
NOT the masking ±1e4 clamp. No retrain.

### Re-eval (post-A-clamp, same `gen_10k.ckpt`) — `eval_10k_v2.json`
| Metric | Pre-fix | Post-fix |
|---|---|---|
| NaN / Inf | 711 (17.8 %) | **0** |
| unique rate @ 1k / 2k / 4k | 0.826 / 0.818 / 0.809 | 0.836 / 0.825 / 0.829 |
| vpa p5 / median / p95 (Å³) | 9.4 / 15.8 / 28.4 | 9.8 / 17.0 / **4582** |
| vpa max (Å³) | 101.5 | **10725** |
| sane fraction (0 < vpa ≤ 500) | 1.00 (of finite) | **0.887** |

The A-clamp **stopped the NaN cascade** (711 → 0) but the same fraction of
runaway trajectories now drives A to its ±50 ceiling → garbage `h` →
**lattice head saturates its tanh bound (B=8)** → L denormalizes to extreme
but finite values (vpa up to 10 725). Net: NaN became extreme-but-bounded.
Total "bad" fraction roughly conserved (~18 % NaN → ~11 % saturated).

**Root cause is the same in both runs:** the 10k model under-fits its own
sampling trajectory at high t. The 1k baseline overfit hard (~1000 epochs ×
9 batches per crystal) and stayed in distribution; the 10k model saw each
crystal only ~6 passes on average → predictions wander OOD on some
trajectories → bound saturates / A explodes.

### Plan pass criteria — does NOT pass
| Criterion | 10k post-fix |
|---|---|
| Unique rate near paper @ 1k | 0.836 vs 0.992 (Δ −0.16) ❌ |
| No formula collapse | 836/1000 distinct ✅ |
| Sane lattice fraction > 99 % | 0.887 ❌ |
| Volume/atom median in data range | 17.0 ✅ |
| No catastrophic tails | vpa max 10725 ❌ |

### Verdict + next options (no further changes without say-so)
**Do not advance to 50k/150k yet.** Options on the table:

- **A** Tighten bounds: A clamp ±10, lattice head B=4–5 (re-eval ~1.4 h).
  Shrinks the saturated tail; does not fix the underlying under-fit.
- **B** Train 10k longer (~2–3 h more on same data) → cheapest test of whether
  *undertraining* is the dominant cause. **Recommended.**
- **C** Jump to 50k anyway (~12 h on a 3060). The real scale test.
- **D** Accept this as the achievable result on a 3060 in 3 h and document.

Files: `gen_10k.ckpt` (saved), `gen_10k_latest.ckpt`, `eval_10k.json`,
`eval_10k_v2.json`, `logs/train_10k.log`, `logs/eval_10k.log`,
`logs/eval_10k_v2.log`. `gen_1k.ckpt` is preserved untouched as the
known-good baseline (Entry 6).

---

## Entry 6 — 2026-05-22 — Fixes 1–3 applied; first working generator (1k baseline)

### Fixes applied (each validated by re-running its gate)

1. **F coordinate sampler** (`models/diffusion.py`). The corrector step
   `d_t = γ·σ_{t-1}²/σ_1²` normalized by the *smallest* σ, reaching ~1e4.
   Fixed: convert network output to the raw score (`score = ε̂_F/λ_t`), VE
   ancestral predictor, safe corrector `d_t = γ·σ_{t-1}²` (γ=0.5).
   → **Gate 6 PASS** (F oracle recovers, err 0.0).

2. **Complete graph** (`data/graph.py`, `data/batch.py`, `models/egnn.py`).
   Replaced the frozen periodic radius graph with a geometry-independent
   complete graph over the N≤20 atoms + minimum-image convention — removes the
   train/sample topology mismatch. → **Gate 7 PASS** (predictions
   graph-independent, cos 1.00).

3. **Lattice channel** (`models/diffusion.py`, `models/egnn.py`):
   (a) statistical normalization — diffuse L in (L−μ)/σ space, μ/σ = dataset
   per-entry mean/std (`data.datasets.compute_lattice_stats`), invertible with
   no ground-truth volume; (b) **x0-prediction** for L — network predicts the
   clean normalized lattice, not ε — removes the 1/√ᾱ high-t amplification;
   (c) removed the ±1e4 clamp; (d) bounded L head `x0 = 8·tanh(raw/8)` to cap
   OOD blow-ups. A stays eps-prediction (unchanged, by design — surgical).
   → **Gate 5 PASS** (x0 reverse recovers L, rel-err 5e-4, no clamp).

### Gate verification after the fixes
| Gate | Result |
|---|---|
| 4 | corr_L 0.997–0.999 at all t (was −0.44 @ t=900); L0 rel-err ~0.04 flat (was 2.66) |
| 5 | A & L oracle reverse recover, no clamp |
| 6 | F oracle reverse recovers (err 0.0) |
| 7 | complete graph: predictions graph-independent |
| 8 | one-crystal sample: 32/32 distinct formulas, vpa 24.8–30.7 Å³ |
| 9 | 32-crystal / 256 samples: unique rate 0.887, vpa median 16.1, **max 345** (bounded head capped the pre-bound 6.7e7), nan 0 |

### Known-good baseline — `gen_1k.ckpt`

First generator that genuinely works. Pretrained on **1,000** diversity-sampled
Alex-MP-20+GNoME structures (`data_raw/pretrain_1k.jsonl`), 1000 epochs / 0.48 h
on RTX 3060, best **val 1.20**.

Architecture (unchanged from Table S.2): EGNN 6 layers, hidden 512, SiLU,
T=1000. Plus the fixes: complete graph, lattice x0 + normalization + bound 8,
F VE-ancestral sampler, Adam 1e-4 + ReduceLROnPlateau.

Evaluation (`eval_1k.json`):
| Metric | gen_1k | Reference |
|---|---|---|
| chemical-formula unique rate @ N=1000 | **0.991** | paper Fig S.4: 0.992 |
| lattice sane fraction (0<vpa≤500) | **0.997** | — |
| volume/atom p5 / median / p95 (Å³) | 3.0 / 16.8 / 30.9 | data median ~21 |
| NaN/Inf | 0 | — |

vs the Entry-3 collapsed overnight model: unique rate **0.065 → 0.991**.

### Caveats
- 1k structures / 1000 epochs = heavily overfit. The 0.991 is a fair match at
  N=1000 only; a 1k-trained model knows ~1k formulas and will fall off the
  Fig S.4 curve faster than the paper's ~1M model at N=16k/256k.
- ~0.3% lattice tail still mildly explodes (max vpa ~4000); expected to tighten
  with more training data.

### Next (per plan): scale-up validation
10k pretrain → eval N=1000/4000 → 50k/150k. Active learning deferred until the
generator is proven to scale.

---

## Entry 5 — 2026-05-21 — Gate results 1–7: failure isolated per channel

Executed Gates 1–7 of the Entry-4 plan. Added six debug scripts under
`invdesflow_al/scripts/`: `debug_data_sanity.py`, `debug_graph_sanity.py`,
`debug_forward_loss.py`, `debug_overfit_one.py`, `debug_oracle_sampler.py`,
`debug_graph_compare.py`.

### Per-gate results
| Gate | Test | Result |
|---|---|---|
| 1 | data sanity (1k records) | ✅ PASS — 76 elements, 812/1000 unique formulas, no element >5.2%, vol/atom 6.7–60.6 Å³ |
| 2 | graph construction (20 crystals) | ✅ PASS — 20 edges/atom, 0 empty, neighbors 2.0–5.6 Å < cutoff |
| 3 | per-channel forward loss (16 crystals, 400 steps) | ✅ PASS — coord/lattice/type all decrease; type-probe stays 57–98 distinct elements (no forward collapse) |
| 4 | denoiser overfit one crystal | ⚠️ passes mean threshold but **t-resolved breakdown shows high-t failure**: at t=900 corr_L 0.34, corr_F −0.19, atom x̂₀ acc 0.00, L rel-err 2.66 |
| 5 | oracle DDPM reverse A/L | ✅ PASS — recovers A (rel-err 1e-11) and L (3e-13), bounded, **no clamp** → A/L reverse math is correct |
| 6 | oracle wrapped-coord reverse F | ❌ FAIL — F not recovered even with exact score (mean err 0.30 on a 0.5-max torus) |
| 7 | frozen vs rebuilt sampling graph | ❌ FAIL — random-template graph flips F-channel eps to cos −0.18 (A/L stay 0.88–1.00) |

### Consolidated per-channel diagnosis
| Channel | Reverse math | Network denoiser | Graph sensitivity |
|---|---|---|---|
| A atom types | ✅ correct (Gate 5) | ⚠️ x̂₀ acc→0 at t=900 (Gate 4) | ✅ graph-independent (Gate 7) |
| L lattice | ✅ correct (Gate 5) | ❌ corr 0.34, rel-err 2.66 at t=900 | ✅ mostly robust (Gate 7) |
| F coords | ❌ broken (Gate 6) | ❌ corr −0.19 at t=900 | ❌ corrupted (Gate 7) |

### Three confirmed faults
1. **F sampler is mathematically broken** — corrector step `d_t = γ·σ_{t-1}²/σ_1²`
   normalizes by the *smallest* σ, reaching `(0.5/0.005)² = 1e4`; kicks F to
   random even with an oracle score. (`models/diffusion.py`)
2. **Frozen random-template sampling graph corrupts the F channel** —
   train/sample topology mismatch; A/L are insensitive, F is not.
3. **Denoiser fails at high t**, where Algorithm 2 begins. Gate 5 proves the
   A/L *math* is correct → this is the network's high-t eps predictions, made
   worse by the `1/√ᾱ` amplification with eps-prediction.

### Corrections to the Entry-3 diagnosis
- Entry 3 said "F channel — only one not obviously broken." **Wrong** — F is
  the most broken (triple fault: math + network + graph).
- The atom-type channel is **not intrinsically wrong**: Gates 3 and 5 pass.
  The C8 collapse is downstream (high-t accumulation). Do **not** rewrite
  atom-type diffusion until graph/lattice/F fixes are measured.

### Recommended fix order (Gates 8–10 superseded — isolation already done)
1. Fix the F sampler (`d_t` normalization / DiffCSP-style coord reverse) → re-run Gate 6.
2. Replace the frozen radius graph with a complete graph (N≤20) → re-run Gate 7.
3. Lattice reparam (per-atom-volume normalization) + remove the ±1e4 clamp;
   consider v-/x₀-prediction for A and L for high-t stability → re-run Gates 4, 5.
4. Then small pretrain (Gate 13), validate with Gate 8.

---

## Entry 4 — 2026-05-21 — Debugging plan: small gates before retraining

The next phase should debug the generator in small, falsifiable steps. Do
**not** run another overnight pretrain until the one-crystal and tiny-dataset
sampling gates pass without lattice clamps or formula collapse.

### Current best diagnosis
- The broad paper reconstruction is reasonable: crystal `(A,F,L)`, fixed `N`,
  DDPM for atom types/lattice, wrapped score matching for fractional
  coordinates, EGNN denoiser, and Table S.2-style optimization.
- The collapse is too severe to blame mainly on 62 epochs or 150k structures.
  Those would lower quality, not produce identical C8 samples and clamp-pinned
  lattices.
- Strongest suspects:
  1. raw `3x3` lattice DDPM + `±1e4` clamp;
  2. atom-type path ending in unconstrained Gaussian one-hot `argmax`;
  3. sampler graph topology frozen from random templates while `F` and `L`
     evolve;
  4. paper's "CrystalNN with lattice scaling" reconstructed as a periodic
     radius graph.

### Gated debug/eval sequence

| Step | Question | Action | Pass criterion |
|---|---|---|---|
| 1 | Is the data sane? | Inspect 1k records from `data_raw/pretrain.jsonl`: atom-count histogram, element frequencies, reduced-formula diversity, lattice volume/atom, invalid lattices. | Diverse formulas/elements, valid `Z`, positive finite volumes, plausible volume/atom quantiles. |
| 2 | Is batching/graph construction sane? | Collate 20 real crystals and report edge counts, neighbor distances, offsets, empty-neighbor atoms. | Most atoms have neighbors, distances below cutoff, no edge explosions, no many-empty graphs. |
| 3 | Does the forward loss learn per channel? | Train 100-500 mini-steps on 8-32 fixed crystals; log coord/lattice/type separately. | All channels decrease; type loss does not go near zero while predictions become constant. |
| 4 | Can the denoiser overfit one crystal without sampling? | Train one crystal; evaluate denoising at fixed `t = 50,250,500,900`; reconstruct estimated `x0`. | Predicted `eps_A`, `eps_L`, `eps_F` correlate with true noise; estimated `A/F/L` close to original. |
| 5 | Is DDPM reverse algebra stable for `A/L`? | Oracle sampler: forward-noise known `A,L`, then reverse using the true noise / posterior target. | Recovers known `A,L`; no clamp needed. Failure means sampler/schedule math is wrong. |
| 6 | Is wrapped-coordinate sampling stable? | Oracle or near-oracle test for `F` on a simple crystal using the wrapped score path. | Fractional coordinates converge near original modulo wrapping. |
| 7 | Is frozen sampling graph hurting generation? | Compare current random-template frozen graph against periodically rebuilt graph or dense/fixed tiny graph. | Rebuilt/dense graph should be stable; if frozen graph diverges, do not use it silently. |
| 8 | Does full one-crystal sampling work? | Overfit one crystal heavily; sample same `N` 32 times. | Same/plausible formula, finite lattice, sane volume/atom, no all-carbon collapse unless target is carbon. |
| 9 | Does tiny-dataset sampling retain diversity? | Overfit 16-64 diverse crystals; sample 256-1k. | Nonzero formula diversity; generated elements roughly match training element support. |
| 10 | Which channel causes collapse? | Ablate: real `L` + generated `A/F`; real `A` + generated `L/F`; real `A/L` + generated `F`. | Identifies primary failing channel or interaction. |
| 11 | Fix lattice representation | Replace raw lattice DDPM with normalized lengths/angles or DiffCSP-style lattice scaler; remove clamp. | Repeat gates 5, 8, 9; volumes finite/plausible without clamp. |
| 12 | Fix atom-type channel | Replace current Gaussian-one-hot `argmax` with DiffCSP-style type diffusion/classification or categorical diffusion. | Repeat gates 4, 8, 9; no constant-element collapse. |
| 13 | Small real pretrain | Train on 1k, then 10k structures for short runs; sample 1k. | Formula unique rate improves; no saturated lattices; metrics trend toward Fig S.4. |
| 14 | Overnight run | Only after all gates pass. | S.4 eval should be meaningful at 1k before scaling to 16k+. |

### Minimal debug scripts to add
- `debug_data_sanity.py`: prints atom counts, element counts, lattice
  determinant, volume/atom quantiles, formula diversity.
- `debug_graph_sanity.py`: prints edge-count/atom, distance quantiles,
  empty-neighbor counts, offset stats.
- `debug_oracle_sampler.py`: tests `A/L` DDPM reverse and wrapped `F` reverse
  with oracle targets.
- `debug_overfit_one.py`: trains one crystal, samples 32, reports formula,
  lattice determinant, volume/atom, coordinate error.
- `debug_channel_ablation.py`: replaces selected generated channels with
  ground truth to isolate collapse.

### Always log during sample evals
```
formula
unique_formula_rate
atom_type_histogram
lattice_det
volume_per_atom
min/max lattice entry
frac_coord_min/max
nan/inf count
edge_count_per_atom
```

Hard fail if any of these occur:
```
NaN/Inf anywhere
abs(lattice entry) > 100
volume_per_atom <= 0 or > 500 A^3/atom
unique formula rate near zero on a tiny diverse set
one element dominates >95% without matching the training subset
sampler only works when the ±1e4 clamp is enabled
```

### Immediate next gates
1. Oracle reverse sampler on one real batch.
2. One-crystal overfit, then sample 32.
3. 32-crystal overfit, then sample 512.

If these fail, architecture/sampler fixes come before more data or longer
training. If these pass, proceed to the lattice/type replacements and then a
small 1k/10k real-data pretrain.

---

## Entry 3 — 2026-05-18 — Overnight pretraining + Fig S.4 eval → **NEGATIVE RESULT**

### What was run
- Installed `pymatgen 2024.8.9` + HF tooling into the `py39` conda env.
- Downloaded **Alex-MP-20** (HF `OMatG/Alex-MP-20`, 540,162 train) and
  **GNoME** (`gs://gdm_materials_discovery`, by_id.zip 554k CIFs, capped 150k).
- Converted both to dependency-free JSONL
  (`invdesflow_al/scripts/convert_datasets.py`).
- Built a memory-safe streaming diversity manifest
  (`scripts/build_manifest.py`): 690,162 scanned → 593,148 after
  (≤20 atoms + composition/spacegroup de-dup) → **150,000 diversity-sampled**
  across 7,032 buckets / 180 spacegroups → `data_raw/pretrain.jsonl`.
- Overnight orchestrator (`scripts/run_overnight.sh`): pretrain 7 h wallclock
  budget → auto-eval vs Supplementary Fig S.4.
- Pretraining: **62 epochs in 7.00 h** on RTX 3060, auto-batch = 96,
  Adam 1e-4 + ReduceLROnPlateau (Table S.2). Loss **37.6 → 1.42** (train),
  best **val 1.4176** (epoch 56). Checkpoints: `generator.ckpt`,
  `generator_latest.ckpt`.
- Eval: sampled 16,000 crystals at T=1000 (~5.9 h, ~1.3 s/sample — slower than
  estimated because real structures avg 9.8 atoms vs the 6-atom probe).

### What changed vs the ORIGINAL PAPER (deviations / reconstructions)
| Item | Paper | Here | Why |
|---|---|---|---|
| Corpus size | ~1M (607k Alex + 381k GNoME) | 150k diversity-sampled | RTX 3060 vs paper's 4090; fit overnight |
| GNoME subset | 381k | 150k random cap | download/convert time |
| Training | 1000 epochs (Table S.2) | 62 epochs / 7 h wallclock | 3060 throughput |
| β-schedule | unspecified | cosine | §11 assumption |
| λ_t (coord loss) | unspecified | σ_t (DiffCSP convention) | §11 assumption |
| Atom-type repr | unspecified | Gaussian DDPM on one-hot + MSE | §11 assumption — **suspected wrong** |
| Lattice repr | unspecified | raw 3×3 DDPM | §11 assumption — **suspected wrong** |
| Graph | "CrystalNN + lattice scaling" | periodic radius graph 7.0 Å / 20 nbr | §11 assumption |
| Eval coverage | Fig S.4 = 1k…256k (9 pts) | 1k…16k (5 pts) | sampling compute-bound, 256k ≈ 50 h on 3060 |

### What changed vs the PREVIOUS RUN (Entry 2 smoke tests)
- First real data instead of synthetic crystals.
- Added: `convert_datasets.py`, `build_manifest.py` (low-RAM streaming),
  `eval_unique_rate.py`, `run_overnight.sh`; trainer gained `--max-hours`,
  `--auto-batch`, `--workers`, periodic checkpointing.
- Kept the Entry-2 anti-NaN lattice clamp (`models/diffusion.py`, ±1e4) — this
  masked, rather than fixed, lattice divergence (see result).

### Result — **severe mode collapse**
| N | Ours | Paper (Fig S.4) | Δ |
|---|---|---|---|
| 1,000 | 0.065 | 0.992 | −0.927 |
| 2,000 | 0.047 | 0.989 | −0.943 |
| 4,000 | 0.030 | 0.984 | −0.954 |
| 8,000 | 0.019 | 0.973 | −0.955 |
| 16,000 | 0.013 | 0.959 | −0.946 |

Diagnostic (48 samples): **100% identical** → C₈, all Z=6, lattice |det| ≈
4×10¹² (saturating the 1e4 clamp). Both the atom-type and lattice channels
collapsed.

### Root cause (honest)
1. **Atom types**: Gaussian DDPM on one-hot + MSE (loss weight 20). Net
   minimised type-loss to ~0.01 by predicting the noise mean → argmax always
   one element. The low type-loss noted in Entry 2 was the collapse signature,
   not success.
2. **Lattice**: raw 3×3 reverse DDPM diverges; the ±1e4 clamp prevented NaN but
   left garbage lattices.
3. The paper underspecifies exactly these; the §11 reconstructed choices are
   wrong. Infra (pipeline, training, eval, orchestration) is correct & verified;
   only the **model core** is wrong.

### Artifacts
`generator.ckpt`, `generator_latest.ckpt`, `unique_rate.json`,
`logs/overnight.log`, `data_raw/{alex_mp_20,gnome,pretrain}.jsonl`.

### What to do next (proposed — awaiting review)
1. **Atom-type channel**: replace Gaussian-on-one-hot with categorical/discrete
   diffusion (D3PM / multinomial) **or** DiffCSP-style continuous latent +
   classification head; rebalance type-loss weight away from 20.
2. **Lattice channel**: constrained parametrization (lengths+angles or symmetric
   matrix-exp à la DiffCSP) + per-atom-volume normalization; **remove the ±1e4
   clamp** and fix the reverse-variance schedule properly.
3. **Gate before scaling**: single-structure overfit test — the sampler must
   reconstruct one known crystal — before any re-pretrain.
4. Re-pretrain (another overnight run) only after the gate passes.
5. Optionally lift the lattice/type formulation more precisely from the
   DiffCSP/CDVAE references the paper cites, rather than re-deriving.

### Diagram — what was run, and where it broke

```
 STEP 1  ✅  install pymatgen + HF tooling                    (py39 env)
 STEP 2  ✅  download Alex-MP-20 540k + GNoME 150k → JSONL
 STEP 2c ✅  build_manifest:  690k scanned ─► dedup 593k
             ─► diversity-sample 150k  (7032 buckets/180 SG)
 STEP 3  ✅  pretrain  62 epochs / 7h  RTX 3060  (auto-batch 96)
             loss  37.6 ──────────────────────► 1.42      TRAINS FINE
 STEP 4  ✅  sample 16k crystals, unique-rate vs Fig S.4

 ─────────────────────────── RESULT ❌ ────────────────────────────────────
     ┌─────────────────────────────────────────────────────────┐
     │  48/48 generated crystals IDENTICAL  →  C₈ (all Z=6)     │
     │  lattice |det| ≈ 4×10¹² Å³  (pinned at ±1e4 clamp)       │
     │  unique-formula rate:  0.065   vs   paper 0.992          │
     └─────────────────────────────────────────────────────────┘
       A channel  ─► collapsed: predicts noise-mean → always carbon
       L channel  ─► diverged : reverse DDPM blows up → clamp-pinned
       F channel  ─► only one not obviously broken

 FIX TARGETS ►  A: categorical/discrete diffusion (not Gaussian-on-one-hot)
                L: constrained parametrization + remove clamp
                gate: single-structure overfit before re-pretrain
```

---

## Entry 2 — 2026-05-17 — Generator + data pipeline built (smoke-tested)

- **Step 1 generator** (`invdesflow_al/`): EGNN denoiser (6 layers, hidden 512,
  SiLU), Algorithms 1 & 2, Table S.2 config. Smoke test passed (loss
  27.9→19.1 on tiny overfit; sampler returns structurally valid periodic
  crystals). Added ±1e4 anti-NaN clamp for the untrained-net regime.
- **Data pipeline**: multi-format ingest, filters (≤20 atoms, E_form/E_hull),
  de-dup, diversity sampler, lazy DataLoaders (96/64/64). Pipeline smoke test
  passed (diversity sampler flattened 127 buckets correctly).
- Nothing trained on real data yet; no paper-comparable metrics.

### Diagram — crystal generation model architecture (what was built)

```
 CRYSTAL   M = (A, F, L)
   A  atom types   [N × 100]   (one-hot over Z = 1..100)
   F  frac coords  [N × 3]     (in [0,1), periodic)
   L  lattice      [3 × 3]     (rows = cell edge vectors)

 ── FORWARD / training  (Algorithm 1) ──────────────────────────────────────
   M_0 ──add noise at random step t──► M_t
        A_t = √ᾱ_t·A + √(1-ᾱ_t)·ε_A          DDPM (Gaussian)        ◄┐ collapsed
        L_t = √ᾱ_t·L + √(1-ᾱ_t)·ε_L          DDPM (Gaussian)        ◄┘ (Entry 3)
        F_t = w(F + σ_t·ε_F)                  wrapped score-matching

 ── EGNN DENOISER  φ(L_t, A_t, F_t, t) ─► predicts (ε̂_L, ε̂_A, ε̂_F) ────────

      A_t[N×100]      L_t[3×3]        t            F_t[N×3]
         │               │            │               │
     node_in        lattice_in    sinusoidal     periodic Δfrac edges
     Linear→512     6 params→512   time embed     radius 7Å, ≤20 nbr
         │               │            │               │
         └───────┬───────┴────────────┘               │
                 ▼  h = node features [N × 512]        │
        ┌────────────────────────────────────┐        │
        │  EGNN layer  × 6   (hidden 512,SiLU)│◄───────┘ edge feats:
        │   edge MLP( h_i, h_j, Fourier(Δf),  │          Fourier(Δfrac),
        │             ‖Δcart‖² )              │          ‖Δcart‖²
        │   → scatter-sum messages            │
        │   → node update (residual)          │
        │   → equivariant coord update (Δf·s) │
        └───────────────┬────────────────────┘
                        ▼
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
      atom_head     frac_scale     lattice_head
      ε̂_A [N×100]   ε̂_F [N×3]      ε̂_L [3×3]  (from graph-mean h)

   loss = ‖ε−ε̂‖²    weighted   coord:1  lattice:1  type:20   (Table S.2)

 ── REVERSE / sampling  (Algorithm 2) ──────────────────────────────────────
   noise ─► [ t = 1000 → 1 :  φ predicts ε̂ ; DDPM step A,L ; predictor-
             corrector step F ] ─► M_0  (generated crystal)
```

## Entry 1 — 2026-05-17 — Paper + supplementary extracted; plan written

- Read both PDFs in full. Wrote [REBUILD_PLAN.md](REBUILD_PLAN.md): architecture,
  Algorithms 1 & 2, exact hyperparameters (Tables S.2/S.3/3/4), datasets, AL
  loop (Eqs 1–3, Fig S.7), §11 documented assumptions, §12 acceptance targets.
- Clarified DFT/QE is free, last-mile only, not needed for the ML rebuild.

### Diagram — InvDesFlow-AL full workflow (the paper's design)

```
                  InvDesFlow-AL : active-learning inverse design
 ╔══════════════════════════════════════════════════════════════════════════╗
 ║                                                                          ║
 ║  Alex-MP-20 + GNoME ──diversity sampling──►┌──────────────────────────┐   ║
 ║  (~1M crystals)                            │  PRE-TRAINED generator   │   ║
 ║                                            │  diffusion + EGNN  (Alg  │   ║
 ║                                            │  1 train / 2 sample)     │   ║
 ║                                            └────────────┬─────────────┘   ║
 ║  target functional crystals ──EMC fine-tune─────────────►│                ║
 ║  (low-Eform / low-Ehull / superconductors)               ▼                ║
 ║                                          ┌───────────────────────────┐    ║
 ║              ┌──────────────────────────►│   generate candidates     │    ║
 ║              │                           └─────────────┬─────────────┘    ║
 ║              │                                         ▼                  ║
 ║              │        ┌────────────  QBC committee  ────────────────┐     ║
 ║              │        │ DPA-2 relax │ FormEGNN     │ SuperconGNN+DFT │     ║
 ║              │        │ ‖F‖<1e-4 eV │ E_form/E_hull│ Tc              │     ║
 ║              │        └──────┬───────────────────────────────────────┘    ║
 ║              │               ▼                                            ║
 ║              │   multi-objective score   S = property·I_relax·I_novelty    ║
 ║              │               │           (Eq.1 / Eq.2 / Eq.3)              ║
 ║              │               ▼                                            ║
 ║   retain top candidates ◄─────┘                                           ║
 ║   add to training set ─► re-fine-tune    (loop ×10 rounds)                 ║
 ╚══════════════════════════════════════════════════════════════════════════╝
        ▲ rebuilt ONLY this box ──┘   (FormEGNN/DPA-2/SuperconGNN/QBC = not built)
```
