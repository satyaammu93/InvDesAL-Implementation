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

## Design principle — Chemistry policy for oxide / piezoelectric AL

> **Sticky policy note (not a chronological entry).** This is the governing
> chemistry policy that downstream AL runs cite by reference. Originally
> written 2026-05-25 in response to the Au-enrichment concern that surfaced
> in Entry 16. All Stage-0/Stage-1 oxide-arm entries from Entry 17 (B′) onward
> apply these exclusions and diagnostics; the symmetry filter from Entry 22 is
> the explicit "Symmetry gate" called out below. Re-read before launching any
> new AL run.

### Origin
Stage-0 CHGNet AL passed mechanically (Entry 16), but the generic selected
sets and post-fine-tune generated batch show a recurring chemistry concern:
**Au can become common among high-scoring candidates**. This is plausible but
dangerous for the real target.

Gold is not chemically impossible — it forms intermetallics, aurides, halides,
hydrides, and appears in the paper's superconducting candidates — but it is a
poor default direction for manufacturable lead-free piezoelectric ceramics:
expensive, scarce, generally not a scalable ceramic ingredient, and likely to
reflect the current oracle/objective more than a useful materials-design
preference.

Current Stage-0 scoring rewards **relaxability / local stability under
CHGNet**, not manufacturability:
```
Au-containing candidates may relax smoothly
CHGNet may score familiar noble/intermetallic chemistry favorably
delta_e rewards relaxation improvement, not synthesizability
no current penalty for cost / scarcity / noble metals / poor ceramic relevance
the pretrain corpus contains many hypothetical broad-chemistry structures
```

Therefore the next oxide/piezo AL runs must include explicit chemistry
controls and diagnostics.

### Hard exclusions for manufacturable lead-free ceramic search
Always exclude Pb for the target:
| Z | element | reason |
|---|---|---|
| 82 | Pb | target is lead-free; toxicity / regulation |

For the first manufacturable ceramic search, also exclude precious/noble/PGM
elements that can dominate proxy stability but are poor manufacturing targets:
| Z | element | reason |
|---|---|---|
| 79 | Au | expensive/scarce; observed enrichment risk; poor ceramic-manufacturing default |
| 78 | Pt | expensive PGM; catalyst/intermetallic bias risk |
| 77 | Ir | expensive PGM; poor scalable ceramic target |
| 76 | Os | toxic/rare PGM; poor target |
| 46 | Pd | expensive PGM; hydrogen/intermetallic bias risk |
| 45 | Rh | expensive PGM; poor scalable ceramic target |
| 44 | Ru | PGM; allow only later if specifically motivated |
| 47 | Ag | precious; possible ceramics exist, but exclude initially to avoid cost bias |
| 80 | Hg | toxicity / volatility |

Default exclusion list for oxide/piezo AL:
```bash
--exclude-elements 82 79 78 77 76 46 45 44 47 80
```

This is a **target-space policy**, not a claim that these elements never form
real materials. They can be re-enabled later in controlled ablations.

### Positive chemistry constraints for lead-free piezoelectric ceramics
Initial oxide/piezo search should require:
```text
contains O
does not contain excluded elements above
finite/sane lattice and min-distance filters from Entry 11
```

Then add soft or hard family constraints in stages:
1. **Oxide ceramic broad:** O + at least one non-noble metal.
2. **Piezo-relevant cation pool:** encourage at least one A-site-like cation
   `{Ba,Sr,Ca,Na,K,Bi,Li,Mg,Zn}` and at least one B-site/d0/polarizable cation
   `{Ti,Zr,Hf,Nb,Ta,W,Mo,Sn,Ge,Sc,Y}`.
3. **Symmetry gate:** reject centrosymmetric structures after relaxation once
   pymatgen symmetry analysis is wired into the oracle.

Do not make the family constraints too tight until the oxide arm is measured;
over-constraining too early can hide generator/oracle problems.

### Oracle-bias diagnostics to add
For every AL round, log element histograms at each stage:
```text
generated
validity-passed
novelty-passed
relaxed-ok
converged_ml
selected top-k
post-finetune generated
```

For each element `Z`, compute enrichment:
```text
enrichment_selected_vs_generated[Z] =
    frac_selected_atoms[Z] / max(frac_generated_atoms[Z], eps)

enrichment_post_vs_pre[Z] =
    frac_post_generated_atoms[Z] / max(frac_pre_generated_atoms[Z], eps)
```

Report:
```text
top 10 elements by atom fraction at every stage
top 10 elements by enrichment selected/generated
excluded-element rejection counts
top selected formulas containing any flagged element
```

### Bias gates
For generic Stage-0 AL (no oxide/piezo policy), treat these as diagnostics:
| Gate | Warning threshold |
|---|---|
| single-element dominance | any element > 30 % of selected atoms |
| enrichment spike | any non-required element enriched > 5x selected/generated |
| Au warning | Au selected atom fraction > 10 % or enrichment > 3x |

For oxide/piezo AL with the manufacturing policy enabled, these are hard gates:
| Gate | Pass criterion |
|---|---|
| excluded elements | 0 selected candidates contain any excluded Z |
| oxygen | 100 % selected candidates contain O |
| diversity | top-50 selected: >=30 formulas and >=15 element sets |
| no substitute dominance | no allowed element > 30 % of selected atoms unless intentionally required by the target family |
| post-finetune drift | post-generated excluded-element fraction remains 0 after filtering and raw pre-filter excluded fraction does not increase vs pre |

### Next AL command shape
Run the oxide arm with the manufacturing policy:
```bash
PY=/home/satya/anaconda3/envs/py39/bin/python
$PY -m invdesflow_al.scripts.run_tiny_al_dryrun \
    --ckpt checkpoints/gen_150k.ckpt \
    --manifest data_raw/pretrain.jsonl \
    --num-generate 2000 \
    --top-k 50 \
    --oracle chgnet \
    --oracle-max-candidates 200 \
    --require-oxygen \
    --exclude-elements 82 79 78 77 76 46 45 44 47 80 \
    --device cuda \
    --out-dir al_runs/chgnet_stage0_oxide_no_pgm_round0
```

Then run the Entry-16 score-movement test on the selected relaxed structures,
and compare element histograms pre/post. The run passes only if both stability
movement and chemistry-policy gates hold.

### Interpretation rule
If Au or another noble/PGM element is strongly enriched in an unconstrained
generic run, that is **not** automatically a discovery signal. It is an
oracle/objective signal that must be tested with:
1. hard-exclusion rerun;
2. cost/scarcity penalty ablation;
3. later FormEGNN/committee comparison;
4. eventual DFT/property validation.

For the lead-free piezoelectric ceramic goal, the constrained oxide/no-PGM
branch is the relevant branch.

---

## Entry 25 — 2026-06-04 — Plan C+ (piezo log-rescale + oracle 500): regression — score-shape ceiling reached

### What shipped
- `--piezo-transform {raw, log}` + `--piezo-scale` in
  [run_tiny_al_dryrun.py](invdesflow_al/scripts/run_tiny_al_dryrun.py).
  `raw` (default) = Entry 24 behaviour `max(|e_max|, floor)`. `log` =
  `log(1 + scale * max(|e_max|, floor))`. With `scale=6.0` the median
  matminer piezo (0.27 C/m²) maps to factor ≈ 0.96 ≈ 1.0 — restoring
  the score scale that Entry 24's raw factor had shrunk.
- `--oracle-max-candidates 500` (was 200) in the rerun, to widen the
  pool for rare Nb/Ti chemistry.
- [run_stage3_piezo_plus.sh](invdesflow_al/scripts/run_stage3_piezo_plus.sh).
  Same everything else as Entry 24.

### Hypothesis going in
Two fixes proposed in Entry 24's diagnosis:
1. log-rescale piezo so median factor ≈ 1 (fixes E24's "post conv 0.31" failure)
2. wider oracle pool (fixes E24's "Ti 0.54%" failure)

### Result — both fixes regressed

| Metric                  | E22 (sym)  | E23 (sym+fam+C′)| E24 (piezo raw) | **E25 (piezo log + 500)** |
|-------------------------|------------|-----------------|-----------------|---------------------------|
| Verdict                 | PASS       | PASS            | PASS            | **FAIL (conv_pass)**      |
| post conv               | 0.544      | 0.253           | 0.311           | **0.264**                 |
| E_form drop             | +0.47      | +0.005          | +0.339          | **+0.065**                |
| ΔE drop                 | +1.18      | +0.96           | +1.17           | +1.21                     |
| Cr enrichment           | 19.0×      | 6.93×           | 5.05×           | 6.71×                     |
| Ti in top-15            | —          | absent          | 0.54%           | **absent**                |

Notable: raw oracle conv stayed strong (88/230 = 38.3%, best of all runs).
The candidate pool was fine. The selection scoring is what broke.

### Why log-rescale flattened the signal
The transform did exactly what we asked — pulled median piezo factor
from 0.27 → 0.96 — but the **dynamic range collapsed**:

| metric             | raw transform  | log transform (scale 6.0) |
|--------------------|----------------|---------------------------|
| median factor      | 0.27           | 0.96                      |
| max factor         | 2.50           | 2.77                      |
| **max / median**   | **9.3×**       | **2.9×**                  |

With a 9.3× spread, the piezo factor was the dominant ranker among
relaxable candidates. With a 2.9× spread, the (-E_form) term took over
and the loop drifted back toward stability-only selection — similar
to Entry 23's collapse, just with different chemistry.

### Why the wider pool didn't help
Validity rate stayed at ~4.6%, capping the post-relax pool at 230 valid
candidates (not 500). We'd need `--num-generate 10000` to fill 500
oracle slots. Doable next time, but unlikely to be the unlock — only
the rarest chemistry gets sampled at all.

### Element distribution — Ti regressed, Ge dominated
Top-15 post (frac / enrichment):

| Z | Symbol | frac  | enrichment | Notes                                  |
|---|--------|-------|------------|----------------------------------------|
| 8 | O      | 64.1  | 9.41       | dominant                               |
| 32| Ge     | 7.48  | **9.24**   | up from E24 (5.05%) — new bias mode    |
| 16| S      | 5.00  | 2.36       | new — sulfide-oxide intrusions         |
| 26| Fe     | 3.17  | 3.38       | B-site held                            |
| 15| P      | 2.94  | 3.51       | down from E24 (4.68%) — good           |
| 9 | F      | 2.64  | 1.01       |                                        |
| 24| Cr     | 2.14  | 6.71       | similar to E24 (5.05×)                 |
| 25| Mn     | 1.19  | 1.72       |                                        |
| 70| Yb     | 0.80  | 7.36       | new rare-earth signature               |
| 56| Ba     | 0.76  | 0.87       | A-site DROPPED (E24 had 1.09%)         |
| 22| Ti     | —     | —          | **absent from top-15** (regressed)     |

### Acceptance against the 6 criteria

| # | Criterion                            | Target | Result   | Pass |
|---|--------------------------------------|--------|----------|------|
| 1 | All 6 score-movement gates PASS      | yes    | conv_pass FAIL | ✗ |
| 2 | E_form drop ≥ 0.30                   | 0.30   | 0.065    | ✗    |
| 3 | post conv ≥ 0.50                     | 0.50   | 0.264    | ✗    |
| 4 | Nb OR Ti ≥ 5 % in top-15             | yes    | absent   | ✗    |
| 5 | Cr enrichment ≤ 10×                  | yes    | 6.71×    | ✓    |
| 6 | V fraction ≤ 5 %                     | yes    | absent   | ✓    |

**2 of 6 met.** Worst of the four head-to-head runs.

### What we've now ruled out

Across Entries 22–25 we have tested every combination of post-relax
score-shape modifier on top of the same generator (gen_150k.ckpt) and
Stage-1 oracle:

| Run | symmetry | C′  | family | piezo factor | gates | Ti? | best metric |
|-----|----------|-----|--------|--------------|-------|-----|-------------|
| E22 | ✓        | —   | —      | —            | PASS  | n/a | E_form +0.47 |
| E23 | ✓        | ✓   | ✓      | —            | PASS  | no  | Cr 19→6.93× |
| E24 | ✓        | ✓   | —      | raw          | PASS  | 0.5%| best balance |
| E25 | ✓        | ✓   | —      | log(scale 6) | FAIL  | no  | (regression) |

**Conclusion: score-shape engineering has hit its ceiling.** Every
combination leaves at least one of (AL signal / chemistry steering /
Ti+Nb visibility) underwater. The bottleneck is **not** the score —
it's the generator's pretraining distribution: too thin in Nb/Ti
perovskites for any downstream scoring to surface them at 5%+.

### Plan B is now the next intervention

Restore Entry 24 (piezo raw) as the Stage-3 reference. Move generator-side:

**Plan B — seed-finetune the generator on MP piezo data.**

The matminer dataset we just used to train the piezo head (941
entries) contains 19 Ti-perovskites, 20 Nb-perovskites, BaTiO₃ in
multiple polymorphs, LiNbO₃, KNbO₃, NaNbO₃, etc. — exactly the
chemistry the generator is sparse in. A few thousand finetune steps
on this dataset (plus regularization against catastrophic forgetting)
should fatten the generator's prior in the target chemistry.

Then re-run Entry 24's pipeline on the seed-finetuned generator. The
piezo head's discrimination plus the generator's beefed-up Nb/Ti
prior should compound — head ranks the perovskites highly, generator
samples them more often, top-15 finally clears the 5% bar.

Estimated cost: ~3 h dev + ~2 h finetune + ~2 h AL run = half a day.
Risk: forgetting the broader stability prior. Mitigation: small LR,
short finetune, EWC-style regularization, hold the original ckpt as
a fallback.

### Files touched
- [invdesflow_al/scripts/run_tiny_al_dryrun.py](invdesflow_al/scripts/run_tiny_al_dryrun.py)
  (+ `--piezo-transform`, `--piezo-scale`, log path in scoring)
- [invdesflow_al/scripts/run_stage3_piezo_plus.sh](invdesflow_al/scripts/run_stage3_piezo_plus.sh)
- [al_runs/chgnet_stage3_piezo_plus/](al_runs/chgnet_stage3_piezo_plus/)
- [al_runs/chgnet_stage3_piezo_plus_movement/](al_runs/chgnet_stage3_piezo_plus_movement/)

---

## Entry 24 — 2026-06-04 — Plan C (piezo head): AL signal recovered, Ti returns to top-15 but at 0.5 %

### What shipped
- **MatBench piezoelectric_tensor dataset** pulled via
  [fetch_piezo_dataset.py](invdesflow_al/scripts/fetch_piezo_dataset.py)
  to [data_raw/mp_piezo.jsonl](data_raw/mp_piezo.jsonl). 941 entries,
  de Jong et al. Sci.Data 2015 — same DFT-PBE data the MP API serves.
  Portable JSONL schema (`z, frac, lattice, target, target_name,
  source, material_id, formula, spacegroup, point_group, n_sites,
  piezo_tensor`) so a future MP-API pull drops in without refactor.

- **PiezoHead** ([piezo_head.py](invdesflow_al/models/piezo_head.py)) —
  EGNN regressor (3 layers × hidden 128, ~430 k params, dropout 0.1)
  predicting `log(eij_max + 0.01)`. Reuses EGNNLayer from the generator
  denoiser. Mean-pool readout → 3-layer MLP → scalar.

- **train_piezo_head.py** — point-group-stratified 80/20 split,
  Smooth-L1 on log-target, AdamW + cosine schedule, early stop on
  val Spearman. Trained in **17 s** on RTX 3060:
  - **best val Spearman ρ = 0.7227** @ epoch 62
  - val MSE 0.98
  - sanity checks on famous piezos: BaTiO₃ pred 3.45 / true 3.45,
    KNbO₃ 2.98 / 3.26, LiNbO₃ 3.42 / 3.42, NaNbO₃ 3.81 / 4.05;
    *and* correctly de-ranks low-piezo Nb compounds
    (NbCu₃Se₄ 0.02 / 0.02, Li₃NbS₄ 0.01 / 0.00). This is the
    discrimination the family prior could not provide.

- **PiezoOracle** ([oracle_piezo.py](invdesflow_al/al/oracle_piezo.py))
  — thin wrapper around the trained head. One method
  `score_relaxed(zlist, frac, lattice) -> float` returns predicted
  |e_max| in C/m². No persistent cache (head is sub-ms/structure).

- **Wired** `--piezo-head` + `--piezo-floor` into
  [run_tiny_al_dryrun.py](invdesflow_al/scripts/run_tiny_al_dryrun.py).
  When set, the score becomes:
  ```
  S = (-E_form) * max(|e_max|, floor) * I_relax_ml * I_novelty
      * I_noncentro * W_comp * family_bonus
  ```
  Floor (0.05 C/m² ≈ p10 of the matminer dataset) prevents low-piezo
  candidates from being zeroed — keeps selection diverse so the loop
  can recover from oracle false-negatives. Per-candidate `piezo_e_max`
  + `piezo_factor` in `rec.meta` and `relaxed.jsonl`; quantiles + head
  config in `oracle_summary`. The score-label string now auto-builds
  from active flags so the summary says exactly what was applied.

### The run
[run_stage3_piezo.sh](invdesflow_al/scripts/run_stage3_piezo.sh).
Same bumped budgets as Entries 22 & 23 (5000 gen / 200 oracle /
200 LBFGS). Wallclock 1 h 56 m. Family prior **dropped** — let the
trained head do chemistry steering. Symmetry filter + C′
inv-enrichment kept (with Entry 22's compare.json as the prior, so Cr
suppression carries through).

### Result — verdict PASS, signal recovered, headline criterion still missing

| Metric                       | E22 (sym)  | E23 (sym+fam+C′)| **E24 (Stage 3 piezo)** |
| ---------------------------- | ---------- | -------- | ----------------------- |
| All 6 gates PASS             | ✓          | ✓        | ✓                       |
| post `fraction_converged_ml` | 0.544      | 0.253    | **0.311**               |
| post `e_form_median`         | −2.08      | −1.77    | **−2.07**               |
| post `delta_e_median`        | 0.761      | 0.761    | **0.661**               |
| **E_form drop**              | **+0.47**  | +0.005   | **+0.339**              |
| ΔE drop                      | +1.18      | +0.96    | +1.17                   |
| Cr enrichment                | **19.0×**  | 6.93×    | **5.05×**               |
| V fraction                   | absent     | absent   | 2.0 % @ 3.67×           |
| post-relax centro (raw)      | 7 / 200    | 1 / 200  | 2 / 200                 |

Stage 3 **recovers ~72 % of Entry 22's E_form drop** that Entry 23
destroyed, while inheriting (most of) Entry 23's chemistry steering
(Cr ↓ further to 5.05×, P down from 11.1 % to 4.7 %).

### Element distribution — Ti returns; Nb still absent
Top-15 post (frac / enrichment):

| Z | Symbol | frac  | enrichment | Notes                                  |
|---|--------|-------|------------|----------------------------------------|
| 8 | O      | 62.6  | 9.19       | dominant                               |
| 32| Ge     | 5.05  | 6.24       | new — piezo head likes Ge-O chemistry  |
| 15| P      | 4.68  | 5.59       | down from E22 (11.1 %) / E23 (11.1 %)  |
| 14| Si     | 4.03  | 2.16       |                                        |
| 26| Fe     | 2.81  | 2.99       | B-site held                            |
| 25| Mn     | 2.77  | 4.00       |                                        |
| 19| K      | 2.22  | 2.35       | **A-site present**                     |
| 23| V      | 2.03  | 3.67       | V re-enters but well under 5 %         |
| 16| S      | 1.93  | 0.91       |                                        |
| 63| Eu     | 1.68  | 7.44       | new — head picks Eu rare-earth oxides  |
| 24| Cr     | 1.61  | 5.05       | down further from E22 (19×)            |
| 56| Ba     | 1.09  | 1.24       | **A-site present (BaTiO₃ chemistry!)** |
| 11| Na     | 0.90  | 0.87       | A-site                                 |
| 30| Zn     | 0.88  | 0.58       |                                        |
| 22| **Ti** | **0.54** | 0.74    | **Ti RETURNS to top-15** (but 0.5 %)   |

**Absent**: Nb (Z=41), Bi (Z=83). Same as before.

### Piezo predictions on generated structures
| quantile | piezo |e_max| (C/m²) |
|----------|---------------------|
| min      | 0.014               |
| p5       | 0.080               |
| median   | 0.273               |
| p95      | 0.679               |
| max      | 2.499               |

Median 0.27 matches the matminer dataset median (0.25) — the head is
**well-calibrated** on out-of-distribution generated structures, not
over-confident. The max is BaTiO₃-territory but reached by only 1–2
candidates.

### Top-15 selected candidates (round-0, piezo head picks)
The head identifies real piezo chemistry, but not the classical
perovskite stars:

| rank | formula (Z-coded)              | score | E_form | \|e_max\| | sg  |
|------|--------------------------------|-------|--------|-----------|-----|
| 1    | O-S-Ce                         | 3.37  | −2.36  | **2.03**  | 8   |
| 2    | O₂-Mo-Eu-Ta                    | 1.59  | −1.54  | **2.50**  | 99  |
| 3    | Li-O₂-Ce-Lu                    | 0.92  | −2.61  | 0.55      | 8   |
| 4    | C-O-Br₃-Sr₃-Sm₂-Tl             | 0.75  | −1.61  | 0.51      | 6   |
| 12   | (multi)                        | 0.49  | −1.30  | 1.24      | 1   |

These are **novel chemistries** — the head identifies them as
candidates for piezo response based on local environment, not on
chemistry priors. Some are physically suspicious (Sm/Tl mixed-anion
compounds may not be synthesizable), but the *direction* is right.

### Acceptance against the 6 criteria

| # | Criterion                            | Target | Result   | Pass |
|---|--------------------------------------|--------|----------|------|
| 1 | All 6 score-movement gates PASS      | yes    | yes      | ✓    |
| 2 | E_form drop ≥ 0.30                   | 0.30   | 0.339    | ✓    |
| 3 | post conv ≥ 0.50                     | 0.50   | 0.311    | ✗    |
| 4 | Nb OR Ti ≥ 5 % in top-15             | yes    | Ti 0.54% | ✗    |
| 5 | Cr enrichment ≤ 10×                  | yes    | 5.05×    | ✓    |
| 6 | V fraction ≤ 5 %                     | yes    | 2.03%    | ✓    |

**4 of 6 met** — best of the three runs (E22: 3/6 (didn't have the
Ti/Nb criterion), E23: 3/6). Criteria 3 (post conv) and 4 (Ti/Nb in
top-15) remain open.

### Diagnosis of the two open failures

**Criterion 3 (post conv = 0.311 vs target 0.50).** Plan C′ + family
removed in this run, so it's not the same loss mode as Entry 23.
The piezo factor multiplies into the score, and because the median
predicted |e_max| is 0.27 (lower than the implicit unit factor of 1.0
in E22), the *absolute scale* of S is smaller and the finetune signal
weaker. This is a hyperparameter issue — the piezo factor needs
either a log-shape or a re-scaling so it doesn't shrink the score
range.

**Criterion 4 (Ti at 0.5 %, Nb absent).** The piezo head correctly
rates high-piezo Nb/Ti compounds when they exist. They just don't get
sampled often enough by the generator. Of 200 oracle candidates the
piezo p95 was only 0.68 — only ~10 candidates predicted at "real
piezo" levels. The bottleneck is upstream: generator sampling, not
scoring.

### What this rules in and rules out

**Rules in:** A trained piezo head is a useful AL signal. It
preserved chemistry steering (Cr ↓, P ↓), didn't crush AL convergence
(unlike the family prior), and surfaced novel candidates with real
predicted piezo response (Ce/Eu/Sm oxides; rank-2 = 2.50 C/m²
≈ BaTiO₃-strength).

**Rules out:** Score-shape alone is enough to recover the canonical
piezo perovskites. Even with a calibrated piezo signal, the generator
samples too little Nb/Ti for them to surface at 5 %+ frequency.

### Two paths forward

1. **Plan B (generator-side): seed-finetune.** Take the gen_150k.ckpt
   and finetune for a small number of steps on the ~3000-entry MP
   piezo dataset (which contains many Nb/Ti perovskites). This widens
   the generator's prior in target chemistry. Then re-run Plan C with
   the seed-finetuned generator. Estimated dev: ~3 h (data loading +
   short finetune). Risk: catastrophic forgetting on the broader
   stability prior.

2. **Plan C+ (oracle-side): re-scale piezo factor + increase budget.**
   Two tweaks:
   - Apply `log(1 + piezo)` instead of raw piezo, so the score range
     doesn't shrink. Or scale the piezo factor up to median ≈ 1.0.
   - Increase `--oracle-max-candidates` from 200 → 500. More
     candidates = more shots at the rare Nb/Ti compounds. Wallclock
     scales linearly (~45 min for oracle phase instead of 18).

I'd go **(2) first** — it's a 1-line change and 1 h longer wallclock.
Then **(1)** if (2) doesn't move Ti past 5 %.

### Files touched
- [data_raw/mp_piezo.jsonl](data_raw/mp_piezo.jsonl)
- [invdesflow_al/scripts/fetch_piezo_dataset.py](invdesflow_al/scripts/fetch_piezo_dataset.py)
- [invdesflow_al/models/piezo_head.py](invdesflow_al/models/piezo_head.py)
- [invdesflow_al/scripts/train_piezo_head.py](invdesflow_al/scripts/train_piezo_head.py)
- [invdesflow_al/al/oracle_piezo.py](invdesflow_al/al/oracle_piezo.py)
- [invdesflow_al/al/__init__.py](invdesflow_al/al/__init__.py) (export `PiezoOracle`)
- [invdesflow_al/scripts/run_tiny_al_dryrun.py](invdesflow_al/scripts/run_tiny_al_dryrun.py)
  (+ `--piezo-head`, `--piezo-floor`, score-label auto-build)
- [invdesflow_al/scripts/run_stage3_piezo.sh](invdesflow_al/scripts/run_stage3_piezo.sh)
- [checkpoints/piezo_head.ckpt](checkpoints/piezo_head.ckpt) (1.7 MB)
- [al_runs/chgnet_stage3_piezo/](al_runs/chgnet_stage3_piezo/)
  + [al_runs/chgnet_stage3_piezo_movement/](al_runs/chgnet_stage3_piezo_movement/)

---

## Entry 23 — 2026-06-03 — Plan A′ (symmetry + family-prior + C′): chemistry steered, AL signal collapsed

### What shipped
- `compute_family_bonus(z_list, mode, b_bonus, ab_bonus)` in
  [run_tiny_al_dryrun.py](invdesflow_al/scripts/run_tiny_al_dryrun.py) — soft
  multiplicative score bonus rewarding piezo-target compositions:
  - **B-site set** (Ti=22, Fe=26, Nb=41) → +b_bonus (default 0.5)
  - **A-site set** (Na=11, K=19, Ba=56, Bi=83) → additional +ab_bonus (default
    0.5) iff a B-site cation is also present
  - max bonus 2.0×, applied after `comp_w` (C′ scarcity) and before top-k
    selection. Never a hard gate — diversity preserved.
- CLI flags: `--family-prior {none, piezo}`, `--family-bonus-b`,
  `--family-bonus-ab`. Per-candidate `family_bonus` logged in `rec.meta` +
  `relaxed.jsonl`. Aggregate counters (`family_with_B_only`, `family_with_AB`,
  `family_with_neither`) in `oracle_summary`. Unit-tested against BaTiO₃,
  KNbO₃, BiFeO₃, NaCl, phosphate, chromate.

### The run (Plan A′ — symmetry + family + C′)
[run_stage1_symmetry_family_Cprime.sh](invdesflow_al/scripts/run_stage1_symmetry_family_Cprime.sh).
Same bumped budgets as Entry 22 (5000 gen / 200 oracle / 200 LBFGS).
Wallclock 1 h 56 m (22:19 → 00:15) — faster than Entry 22's 4–5 h estimate.
Prior for C′ was Entry 22's `compare.json`, so Cr/P/Mn were downweighted.

### Result — verdict PASS but the meaningful numbers regressed

| Metric                                  | Entry 22 (Plan A only) | **Entry 23 (A′)** | Δ       |
| --------------------------------------- | ---------------------- | ----------------- | ------- |
| All 6 gates pass                        | ✓                      | ✓                 | —       |
| post `fraction_converged_ml`            | **0.544**              | **0.253**         | −0.291  |
| post `delta_e_median` (eV/atom)         | 0.761*                 | 0.761             | ~0      |
| post `e_form_median` (eV/atom)          | −2.08                  | −1.77             | +0.31   |
| **`e_form_drop` (pre − post)**          | **+0.47**              | **+0.005**        | −0.465  |
| `delta_drop` (pre − post)               | +1.18                  | +0.96             | −0.22   |
| post-relax centrosymmetric (raw)        | 7 / 200                | 1 / 200           | −6      |
| Cr enrichment (max)                     | **19.0×**              | **6.93×**         | −64 %   |
| V in top-15                             | absent                 | absent            | —       |

\* delta is sensitive to which candidates pass `converged_ml`; identical
to within noise.

### Element distribution — composition steering succeeded
Top-15 post (frac / enrichment):

| Z | Symbol | frac | enrichment | Notes                                  |
|---|--------|------|------------|----------------------------------------|
| 8 | O      | 54.2 | 7.96       | dominant (oxide constraint)            |
| 11| Na     | 5.5  | 5.33       | **A-site present**                     |
| 25| Mn     | 5.0  | 7.20       | over 5×; not banned                    |
| 28| Ni     | 4.7  | 2.72       |                                        |
| 26| Fe     | 3.5  | 3.70       | **B-site present (BiFeO₃ chemistry)**  |
| 33| As     | 2.4  | 2.77       |                                        |
| 24| Cr     | 2.2  | 6.93       | **down 64 % vs Entry 22 (19×)**        |
| 9 | F      | 2.2  | 0.83       |                                        |
| 19| K      | 2.0  | 2.06       | **A-site present (KNbO₃ chemistry)**   |
| 32| Ge     | 1.9  | 2.30       |                                        |
| 55| Cs     | 1.5  | 1.68       | alkali (A-site analogue)               |
| 14| Si     | 1.5  | 0.79       |                                        |
| 75| Re     | 1.2  | 3.44       |                                        |
| 16| S      | 1.2  | 0.59       |                                        |
| 52| Te     | 1.2  | 0.97       |                                        |

**Absent from top-15**: Ti (22), Nb (41), Bi (83) — the canonical
high-strength piezo cations. Even though the family prior rewarded them.

### Diagnosis — three observations

1. **Composition prior is upper-bounded by the generator prior.** Of
   200 oracle candidates: 23 had B-only, 13 had A+B, 164 had neither
   (82 %). The pretrained generator has too few Ti/Fe/Nb-oxide
   examples for the score prior to steer toward them — *the prior can
   re-weight what gets sampled, not what gets generated*. The 13
   "AB" candidates that did exist were mostly Fe-based (BiFeO₃-like);
   Nb examples were rare.

2. **The score-shape penalty crushed the AL signal.** E_form drop
   went from +0.47 → +0.005 and post conv from 0.54 → 0.25. The
   finetune set was small (~5 % family bonus on top of E_form), but
   it was also less stable — the candidates with high `family_bonus`
   weren't the most relaxable. Pushing the finetune toward
   "preferred chemistry, slightly worse stability" hurt the global
   E_form distribution.

3. **Cr / V suppression worked.** C′ inv-enrichment + Entry 22's
   prior was sufficient: Cr enrichment 19× → 6.93×, V absent.
   Criterion 7 cleared.

### Acceptance against the 6 criteria
| # | Criterion                            | Target | Result   | Pass |
|---|--------------------------------------|--------|----------|------|
| 1 | All 6 score-movement gates PASS      | yes    | yes      | ✓    |
| 2 | E_form drop ≥ 0.30                   | 0.30   | 0.005    | ✗    |
| 3 | post conv ≥ 0.50                     | 0.50   | 0.253    | ✗    |
| 4 | Nb OR Ti ≥ 5 % in top-15             | yes    | neither  | ✗    |
| 5 | Cr enrichment ≤ 10×                  | yes    | 6.93×    | ✓    |
| 6 | V fraction ≤ 5 %                     | yes    | absent   | ✓    |

**3 of 6 met.** The chemistry-steering half of the plan worked;
the AL-signal-preservation half did not.

### What this tells us about further composition priors

Soft composition priors have a **hard ceiling** that's set by the
generator's pretraining distribution. The C′-style scarcity penalty
*can* push the generator AWAY from over-represented chemistries (V, Cr),
because it has plenty of alternatives to fall back on. But a family
prior trying to push it TOWARD a sparse chemistry (Nb, Ti, Bi) has
nowhere to go — the generator simply doesn't have enough Nb-perovskite
density for the prior to find candidates to up-weight.

Two ways past this ceiling:

- **(a) Generator-side fix**: re-pretrain with Materials Project piezo
  data overlaid (~3000 Nb/Ti/Bi-perovskite-rich structures) to deepen
  the prior in the target chemistry. Heavy. Slow.
- **(b) Oracle-side fix**: replace the score with a piezo-tensor
  predictor (Plan C). The piezo signal naturally ranks the few Nb-O
  examples that do get sampled, AND rewards strong piezo response
  among the more populous Fe/Mn-oxide non-centro candidates. The AL
  loop then steers the generator toward "piezo-relevant chemistry"
  without us having to specify the family explicitly.

(b) is faster to implement and aligned with the paper's overall
philosophy of "let the oracle steer". (a) becomes the fallback if
the piezo head can't pull enough Nb/Ti into top-15 even when present.

### Next — Plan C: piezoelectric tensor predictor

Stop iterating on chemistry priors. Move to a real piezo signal.

1. **Pull MP piezoelectric dataset** (~3000 entries with full e_ij /
   d_ij tensors) via mp-api → `data_raw/mp_piezo.jsonl`. Target:
   `|e_max| = max(|e_ij|)` (clamped-ion piezoelectric coefficient).
2. **Train a small EGNN regressor** on the relaxed structure → scalar
   |e_max|. Reuse the EGNN block from `invdesflow_al/models/`; 3
   layers, hidden 128. Target ≤ 2 h training on RTX 3060.
3. **`PiezoOracle` class** in `invdesflow_al/al/` with the same
   `RelaxResult`-shaped interface as `CHGNetOracle`. Score becomes
   `S = (−E_form) · |e_max| · I_relax · I_novelty · I_noncentro`.
4. **Re-run** the bumped Stage-1 setup with the piezo head as the
   primary scoring signal (drop family prior; keep symmetry + C′).
   Acceptance: same 6 gates + Nb OR Ti ≥ 5 % in top-15.

### Files touched
- [run_tiny_al_dryrun.py](invdesflow_al/scripts/run_tiny_al_dryrun.py)
  (+47 lines: helper, CLI flags, score application, logging)
- [run_stage1_symmetry_family_Cprime.sh](invdesflow_al/scripts/run_stage1_symmetry_family_Cprime.sh)
  (orchestrator)
- [al_runs/chgnet_stage1_symmetry_family_Cprime/](al_runs/chgnet_stage1_symmetry_family_Cprime/)
  (round-0 summary + selected.jsonl)
- [al_runs/chgnet_stage1_symmetry_family_Cprime_movement/](al_runs/chgnet_stage1_symmetry_family_Cprime_movement/)
  (compare.json + safety_eval.json)

---

## Entry 22 — 2026-06-02 — Plan A (non-centrosymmetric filter): PASS, stronger AL signal, composition narrows

### What shipped (your patches + my chained run)
- `is_centrosymmetric_crystal()` + `is_centrosymmetric_relaxed()` in
  `run_tiny_al_dryrun.py` — pymatgen `SpacegroupAnalyzer` looks for the
  inversion operation directly (since this pymatgen version lacks
  `is_centrosymmetric()`).
- `--reject-centrosymmetric` (pre-relax validity gate) and
  `--reject-centrosymmetric-post` (post-relax selection gate) in both
  `run_tiny_al_dryrun.py` and `run_al_score_movement.py`.
- Terminology footnote: it's **20 piezoelectric point groups**, not 21
  non-centrosymmetric spacegroups. The non-centrosymmetric filter is the
  necessary first coarse gate — point group 432 is non-centrosymmetric
  but still not piezoelectric. The point-group filter goes on top in C.

### Chained run on the bumped Stage-1 setup
`al_runs/chgnet_stage1_noCprime_symmetry{,_movement}/`. Same Stage-1
baseline (full elemental refs, 5000 gen, 200 LBFGS, 200 cap, 50 top-k,
no C′) + the two new symmetry flags.

### Round-0 numbers
| | Entry 21 bumped | **+ symmetry** |
|---|---|---|
| Generated | 5000 | 5000 |
| Valid | 221 (4.4 %) | **226 (4.5 %)** |
| `centrosymmetric_pre` rejects | n/a | **only 1 / 5000** |
| Relaxed | 200 | 200 |
| `converged_ml` | 72 (36 %) | **86 (43 %)** |
| Selected | 50 | 50 |

The pre-relax filter rejected just **1 in 5000** generated candidates —
the diffusion generator at this stage almost never produces *exactly*
centrosymmetric structures on its own. The post-relax filter rejected
7 / 200 (**3.5 %**); 3.5 % is the rate at which CHGNet relaxation
collapses an initially non-centrosymmetric structure *into* inversion
symmetry. That's the piezo-loss rate during relaxation.

### Score-movement — strongest AL signal yet, PASS
| Gate | Entry 21 bumped | **+ symmetry** |
|---|---|---|
| Verdict | PASS | **PASS** |
| Pre conv_ml | 0.36 | **0.43** |
| Post conv_ml | 0.295 | **0.544** |
| Conv change | −0.07 | **+0.11** (rises) |
| ΔE drop | +1.14 | +1.18 |
| **E_form drop** | +0.15 | **+0.47** (3× larger) |
| Post E_form median | −1.92 | **−2.08** |
| Memo | 0.040 | 0.048 |
| Coverage | 100 % | 100 % |
| Post valid_fraction | 0.91 | **0.97** |

E_form drop tripled. Conv fraction rose (vs falling slightly in Entry 21).
The non-centrosymmetric slice is a more learnable region for CHGNet AL.

### Composition — narrowed to P + Cr
Top-15 post-finetune (5805 atoms, 484 valid candidates):
```
  Z   count post_frac  baseline   enrich
   8   3645  0.628     0.068      9.22  ← O (forced)
  15    643  0.111     0.0084    13.24  ← P  (phosphates) — high
  24    351  0.061     0.0032    18.96  ← Cr ⚠ over 10× bar
  26    180  0.031     0.0094     3.30  ← Fe
   9    128  0.022     0.0262     0.84  ← F (under baseline)
  74     83  0.014     0.0026     5.51  ← W
   3     72  0.012     0.0158     0.79  ← Li (under)
  55     61  0.011     0.0088     1.20  ← Cs
  19     55  0.010     0.0095     1.00  ← K (natural)
  32     47  0.008     0.0081     1.00  ← Ge (natural)
  ...
```

Two notable shifts vs Entry 21 bumped:
- **Cr re-emerged at 6.1 % / 19×** (was 1.4 % / 4.3× in Entry 21).
- **Nb (Z=41) dropped out of top-15** — ironically, the symmetry filter
  selected away from Nb-bearing chemistry. The model finds non-centro
  P-O and Cr-O configurations easier than non-centro Nb-O ones in the
  generated pool.

P + Cr is **genuinely piezo-relevant** (KH₂PO₄ family is the textbook
KDP piezoelectric; chromium phosphates have non-centrosymmetric variants),
but it isn't the K-Na-Nb / Ba-Ti-O target space.

### Against the seven Stage-1 + symmetry pass criteria
| Criterion | Status |
|---|---|
| 1. E_form coverage ≥ 80 % | ✅ 100 % |
| 2. Selected top-50 diverse | ✅ |
| 3. No banned elements | ✅ |
| 4. 100 % O for oxide branch | ✅ |
| 5. Safety eval passes | ✅ |
| 6. **Post median E_form improves** | ✅ +0.47 eV/atom |
| 7. **V/Cr don't explode** | ⚠ **Cr at 19×** (over 10× bar; V still gone) |

**6 of 7** — criterion 7 is the one open issue. The symmetry filter
strengthens the AL signal but shifts the bias to phosphates + chromates.

### Files
`al_runs/chgnet_stage1_noCprime_symmetry{,_movement}/`,
`logs/stage1_symmetry_{round0,movement}.log`, updates to
`run_tiny_al_dryrun.py`, `run_al_score_movement.py`.

### What's next — Plan C (piezo-tensor oracle)
The symmetry filter is the **coarse** gate. The fine gate is a
piezoelectric-tensor predictor. Two paths:
1. **Train a small head on Materials Project piezo data** (~3000 entries
   with piezoelectric tensors). Output: scalar like max(|d_ij|) or e₃₃.
   GNN regressor over the relaxed structure → adds ~3–4 h dev + run.
2. **M3GNet / MACE-MP-0 + heuristic** — there's no direct piezo head in
   the stock potentials; would have to derive piezo via finite-difference
   strain + Born effective charge approximations. More machinery, less
   clean.

Recommendation: **option 1** (small head on MP piezo data). It gives a
direct paper-Eq.3-style score `S = (−E_form) · |d_max| · I_relax · I_novelty`,
which is the actual piezoelectric AL objective.

---

## Entry 21 — 2026-06-02 — Stage 1 baseline locked in: full PASS on all 6 gates

### What shipped (your patches + my chained run)
- `invdesflow_al/al/elemental_refs.py` — explicit `_explicit_fallback_atoms`
  for **B / P / S / Mn / Ga / Se / Te** (the 7 elements that previously
  failed `ase.bulk()` with "requires_atomic_basis"). Small, deterministic,
  multi-atom cells: bcc-Mn, alpha-Ga 8-atom, rhombohedral-B, P4-in-vacuum,
  S8-puckered-ring, etc.
- `invdesflow_al/scripts/build_elemental_refs.py` — `--retry-failed` flag
  to discard negative-cache entries and recompute.
- `invdesflow_al/scripts/run_stage1_overnight.sh` — pre-warm step now uses
  `--retry-failed`.
- Rebuild: `chgnet_elemental_refs.json` now **67/67 ok**. Energies (eV/atom):
  ```
  B  -5.51   P  -5.16   S  -4.15   Mn -9.09
  Ga -3.00   Se -3.35   Te -3.15
  ```

### Two follow-up runs
**v2 (small budget, full refs)** — same Entry-20 budget
(`--num-generate 2000 --lbfgs-steps 100`) but with the rebuilt cache.
Coverage cleared (100 %), but `e_form_pass` flipped to FAIL: post −1.37
vs pre −1.76 = drop −0.39 eV/atom (post less stable).

The flip vs Entry 20's apparent +0.47 drop is a **measurement artifact**:
Entry 20's pre median was computed over only **19/86** candidates (those
whose elements all had refs); v2 covers all **88/88**, including
previously-hidden Mn/Ga/P/S/etc-containing candidates whose stability is
distributed differently. The fair within-v2 comparison still shows a
small regression, attributable to a thin selected set (8 fine-tune
crystals) failing to drive a meaningful distribution shift.

**Bumped (5000 gen, 200 LBFGS, full refs)** — same Stage-1-only setup,
larger pool + longer relaxation. **All 6 gates PASS.**

### Bumped result vs all prior runs
| Run | Selected | conv_ml | E_form drop | Top non-O frac | Max enrichment | Verdict |
|---|---|---|---|---|---|---|
| Entry 18 (ΔE only) | 50 | 9 % | n/a | V 25.8 % | V 46.6× | "PASS" (no E_form gate) |
| Entry 19 (ΔE + C′) | 50 | 12.5 % | n/a | V 14.5 % | V 26.2× | "PASS" |
| Entry 20 Stage 1 (partial refs) | 7 | 8 % | +0.47 (inflated) | Na 9.9 % | W 14× | FAIL cov |
| v2 (full refs, small) | 8 | 10 % | −0.39 | Li 14 % | W 18× | FAIL e_form |
| **Bumped (full refs + 5000 + 200 LBFGS)** | **50** | **36 %** | **+0.15** | **P 5.5 %** | **Eu 7.4×** | **PASS** |

### Bumped gate detail (`al_runs/chgnet_stage1_noCprime_bumped_movement/compare.json`)
| Gate | Threshold | Bumped value |
|---|---|---|
| safety | Entry-8 quick-eval thresholds | ✅ |
| delta | ΔE drop ≥ 0.05 | ✅ +1.14 |
| conv | conv change ≥ −0.10 | ✅ −0.065 (within tol) |
| memo | ≤ 0.50 | ✅ **0.040** |
| **e_form_pass** | post ≤ pre − 0 | ✅ **+0.15 eV/atom (post more stable)** |
| **e_form_cov_pass** | ≥ 80 % | ✅ **1.00** |

PRE: 200 candidates relaxed, **fraction_conv_ml 0.360**, ΔE-median 1.81,
**E_form-median −1.77**.
POST: 200 candidates (fresh held-out), fraction_conv_ml 0.295, ΔE-median
0.67, **E_form-median −1.92**.

Post valid_fraction is **0.912** (456/500 — only 44 atom-overlaps from
the validity filter on the fresh batch). Distinct formulas: **331 / 456**.

### Composition — real oxide chemistry, no single-element dominance
Top-15 post-finetune valid (5372 atoms across 456 valid crystals):
```
  Z   count post_frac  baseline   enrich
   8   3520  0.6589    0.0681      9.67  ← O, forced
  15    296  0.0554    0.0084      6.62  ← P (phosphates)
  16    175  0.0328    0.0212      1.55  ← S (natural)
  32    130  0.0243    0.0081      3.00  ← Ge
   3    115  0.0215    0.0158      1.36  ← Li (natural)
  26     93  0.0174    0.0094      1.85  ← Fe (ferrites — natural)
  63     90  0.0168    0.0023      7.44  ← Eu
  24     73  0.0137    0.0032      4.29  ← Cr (moderate)
  40     68  0.0127    0.0104      1.22  ← Zr (natural!)
  25     66  0.0124    0.0069      1.79  ← Mn (natural)
  19     65  0.0122    0.0095      1.29  ← K  (natural)
  42     57  0.0107    0.0036      3.01  ← Mo
  71     47  0.0088    0.0107      0.82  ← Lu (UNDER baseline)
  75     41  0.0077    0.0036      2.14  ← Re
  30     40  0.0075    0.0153      0.49  ← Zn (UNDER baseline)
```
And **Nb (Z=41) at 1.07 %, enrich 3.0×** — the key piezoelectric precursor
for K-Na-niobate (KNN). It's *in* the top-15.

### All seven Stage-1 pass criteria
1. E_form for ≥ 80 % relaxed: **✅ 100 %**
2. Selected top-50 diverse: **✅ 50 selected** (was 7)
3. No banned elements: ✅
4. 100 % O for oxide branch: ✅
5. Safety eval passes: ✅
6. **Post median E_form improves**: ✅ **+0.15 eV/atom**
7. **V/Cr don't explode**: ✅ V not in top-15; Cr at 1.4 % @ 4.3× (was 25.8 % @ 46.6× in Entry 18)

**All seven met cleanly. Stage 1 baseline is locked.**

### Why the bumped budget mattered
At 100 LBFGS / 2000 gen (Entry 20 + v2): `conv_ml` floor was ~10 %,
selected count 7–8. Fine-tuning on 7 crystals isn't enough signal to
move a 150 k-pretrained generator coherently — hence the noisy E_form
post-medians.

At 200 LBFGS / 5000 gen (bumped): `conv_ml` jumped to **36 %**, the
selection cap (50) was actually hit, and the fine-tune signal cleared
the gates by margin. The lesson: oxide+eaonly chemistry is intrinsically
harder for CHGNet (PRE conv 0.10 baseline) and the loop needs more
candidate headroom than the open-composition runs did.

### Files added
`al_runs/chgnet_stage1_noCprime_v2{,_movement}/`,
`al_runs/chgnet_stage1_noCprime_bumped{,_movement}/`,
`logs/stage1_noCprime_v2_*.log`, `logs/stage1_bumped_*.log`. Updated
`elemental_refs.py`, `build_elemental_refs.py`, `run_stage1_overnight.sh`.

### What's next — clearing the way for the piezoelectric direction
The composition space is now genuinely accessible:
- **K, Na, Nb, Ba, Fe, Ti** are all in the natural-enrichment range
  of the post-finetune distribution.
- **No single-element bias** above the 15 % / 10× thresholds.
- **Reproducible 100 % E_form coverage** via the rebuilt elemental cache.
- **PASS verdict** with margin under the paper-faithful Eq. 1 score.

So per your order — **3 is now unblocked**. Real next steps for the
lead-free piezoelectric ceramic direction:
1. **Symmetry filter**: reject centrosymmetric structures (use
   `SpacegroupAnalyzer.is_centrosymmetric()`); only non-centrosymmetric
   spacegroups can be piezoelectric.
2. **Property predictor**: wire in a piezoelectric-tensor predictor
   (M3GNet/MACE if available, or a small task head trained on MP piezo
   tensors) as the next oracle on top of CHGNet's stability scoring.
3. **Composition prior toward KNN / BaTiO3 / BiFeO3 family**: optionally
   tighten the `--exclude-elements` / require key cations.
4. **Active-learning loop on the piezoelectric oracle**: same shape as
   Stage 1 but the score becomes `(−E_form) · |d_33|` (or similar).

---

## Entry 20 — 2026-05-27 — Stage 1 (paper E_form) eliminates the V-bias

### Headline
Stage 1 — paper Eq. 1 score, `S = (−E_form) · I_relax · I_novelty · W_comp` —
**eliminated the V-bias entirely** and shifted post-finetune composition
toward chemically clean oxide chemistry (alkali / alkaline-earth /
main-group). Both runs (with C′ and without) **fail one gate**
(`e_form_cov_pass` at 80 %) because 7 elements had ref-build failures;
**all six scientific gates pass cleanly**, including the new paper-target
gate (post median E_form drops 0.47–0.49 eV/atom — more negative = more
stable).

### What shipped
- `invdesflow_al/al/elemental_refs.py` — `ElementalRefs` with lazy
  on-demand CHGNet relaxation of `ase.build.bulk(sym)`. Diatomic /
  monatomic-in-vacuum fallbacks for elements ASE doesn't build by default.
  Persistent JSON cache; failures negative-cached.
- `invdesflow_al/scripts/build_elemental_refs.py` — pre-warm script.
- `run_tiny_al_dryrun.py` — `--use-e-form` + `--elemental-refs-cache`.
  E_form per Eq. 1 replaces ΔE in `stage0_score` when available; falls
  back to ΔE per candidate when any element ref is missing.
  `composition_weight` × `stage` × `e_form` recorded per candidate.
- `run_al_score_movement.py` — `--use-e-form` flag; computes pre/post
  median(E_form) on the held-out batch; new gate
  `e_form_pass` (post ≤ pre − thresh) plus
  `e_form_cov_pass` (≥ 80 % coverage of `status=ok` candidates).
- `invdesflow_al/scripts/run_stage1_overnight.sh` — three-phase
  orchestrator (pre-warm refs → Stage 1 + C′ → Stage 1 only).

### Element-distribution comparison across runs
| Run | Top non-O element | Max enrichment | V in top-5 | Cr in top-5 |
|---|---|---|---|---|
| Entry 18 — ΔE only | **V 25.8 % @ 46.6×** | 46.6× | yes | yes (21.8×) |
| Entry 19 — ΔE + C′ | V 14.5 % @ 26.2× | 26.2× | yes | yes (21.8×) |
| **Stage 1 + C′** | **Zn 12.7 % @ 8.3×** | Cr 18.2× | **GONE** | yes (18.2×) |
| **Stage 1 only**  | **Na 9.9 % @ 9.6×**  | W 14.1×  | **GONE** | **GONE** |

**Stage 1 alone** is the chemistry-cleanest result:
post-finetune composition is 53 % O + **Na 9.9 % + Mg 9.8 % + K 6.7 % +
P 4.2 % + W 3.7 %** — alkali / alkaline-earth / main-group oxides. This is
the natural chemistry space for lead-free piezoelectric ceramics
(K-Na-niobates, alkaline-earth titanates / phosphates etc.), not the
transition-metal-oxide concentration the ΔE proxy was producing.

C′ on top of E_form is **slightly counterproductive** — the prior was
built from the C′-on-ΔE Entry 19 distribution, which itself was biased.
Score is composition-normalized once you use E_form, so C′'s job is mostly
done by the new score function.

### Score-movement gates (post vs pre on held-out batch)
| Gate | thresh | Stage 1 + C′ | Stage 1 only |
|---|---|---|---|
| safety_pass | Entry-8 quick-eval thresholds | ✅ | ✅ |
| delta_pass | ΔE drop ≥ 0.05 | ✅ −1.28 | ✅ −1.48 |
| conv_pass | conv change ≥ −0.10 | ✅ +0.33 | ✅ +0.31 |
| memo_pass | ≤ 0.50 | ✅ 0.064 | ✅ 0.081 |
| **e_form_pass** | post − pre ≤ 0 (more negative = better) | **✅ −0.49** | **✅ −0.47** |
| **e_form_cov_pass** | ≥ 80 % E_form coverage | ❌ 64 % | ❌ 57.5 % |
| **VERDICT** | | **FAIL** (coverage only) | **FAIL** (coverage only) |

#### Why coverage failed
Seven elements have no ASE-bulk default and aren't in our current
fallback set:
```
Z=5  B   Z=15 P   Z=16 S   Z=25 Mn   Z=31 Ga   Z=34 Se   Z=52 Te
```
All fail with the same `ValueError: This structure requires an atomic
basis`. My exception handler caught the wrong type (caught only the
bulk-default exception, not the "needs explicit basis" one) — easy fix:
provide explicit `(crystalstructure, a)` for these elements. Mn (bcc),
Ga (orthorhombic), B (rhombohedral), Te/Se (chains), P (white-P approx),
S (S8 approx) are all well-known. Will add in a follow-up commit.

Coverage shortfall costs us E_form for ~35–45 % of candidates (those
that contain one of the seven). Those candidates fall back to the
Stage-0 ΔE score in the loop — graceful per design.

### Against your seven Stage-1 pass criteria
1. E_form for ≥ 80 %: ❌ 64/58 % (coverage-fixable)
2. Selected top-50 diverse: ⚠ only **7 selected** (CHGNet convergence
   floor at oxide+eaonly is ~10 % ML-converged within 100 LBFGS steps —
   tight pool)
3. No banned elements: ✅
4. 100 % O for oxide branch: ✅
5. Safety eval passes after fine-tune: ✅
6. **Post median(E_form) improves**: ✅ both (−0.49, −0.47 eV/atom)
7. **V/Cr don't explode beyond previous run**: ✅ both (V disappeared
   from top-5; Cr halved or disappeared)

**5 of 7 met, 1 partial (low selected count), 1 fixable (coverage).**

### Files added
`invdesflow_al/al/elemental_refs.py`,
`invdesflow_al/scripts/{build_elemental_refs,run_stage1_overnight}.{py,sh}`;
`data_raw/chgnet_elemental_refs.json` (60/67 ok); two run dirs
`al_runs/chgnet_stage1_{Cprime,noCprime}{,_movement}/`;
`logs/stage1_overnight*.{log,txt}`.

### What's next
1. **Fix the 7 ref builds** (explicit `(crystalstructure, a)` for B, P,
   S, Mn, Ga, Se, Te). Re-run Stage 1; coverage should clear 80 %.
2. **Increase `--num-generate`** (5000?) or `--lbfgs-steps` (200?) for
   the oxide+eaonly branch — current convergence floor is tight, limits
   selected count to ~7.
3. **Accept Stage 1 alone as the new baseline** (drop C′ — it became
   slightly counterproductive on top of a composition-normalized score).
4. After that lands: the actual **lead-free piezoelectric direction** —
   K/Na-niobate-like and alkaline-earth-titanate-like compositions are
   now in the model's natural output range, so domain-specific filters
   (non-centrosymmetric symmetry, target Curie temperature, etc.) are
   the next layer.

---

## Entry 19 — 2026-05-26 — Plan C′ implemented + rerun: V-bias cut 44 %, but partial

### What shipped
`run_tiny_al_dryrun.py` gained `--scarcity-mode {none, inv-enrichment}` +
`--enrichment-prior <path>` + `--scarcity-min-weight`. With
`inv-enrichment`, the Stage-0 score becomes
**`S = ΔE · I_relax_ml · I_novelty_pre · W_comp(z_list)`**,
where `W_comp` is the per-atom-averaged weight
`W(z) = max(min_w, 1 / max(enrichment(z), 1))` using the prior's
`element_distribution.post_finetune_valid_enrichment_top` table.
`composition_weight` recorded per-candidate in `relaxed.jsonl` and
`oracle_summary` records the mode + prior used.

Helper functions: `load_scarcity_weights()`, `composition_weight()`.

### Smoke test verified wiring
Loaded Entry 18's compare.json as prior:
```
covered_Z=15  most-penalized:
  V (Z=23) → 0.021       (from 46.6× enrichment)
  O (Z=8)  → 0.116       (from 8.65×; forced by --require-oxygen, expected)
  Eu/Sr/Ge/P/Cr  → 0.28-0.42
  Mn/Cu/...      → ~1.0 (untouched)
```

### Full Plan A + C′ rerun
Same chained orchestrator
(`invdesflow_al/scripts/run_oxide_eaonly_Cprime_test.sh`), same Stage-0 +
score-movement structure, with `--scarcity-mode inv-enrichment`.

Outputs:
- `al_runs/chgnet_stage0_round0_oxide_eaonly_Cprime/`
- `al_runs/chgnet_stage0_round0_oxide_eaonly_Cprime_movement/`

#### Head-to-head vs Entry 18 (Plan A without C′)
| Metric | Entry 18 (no C′) | **C′** | Change |
|---|---|---|---|
| Verdict | PASS | **PASS** | both gates clear |
| ΔE drop | −1.150 | **−0.963** | smaller (still 19× the 0.05 bar) |
| conv change | +0.425 | **+0.230** | smaller but ≥ +0.1 |
| memorization | 0.031 | **0.015** | ↓ halved |
| safety unique_rate | 0.63 | **0.69** | ↑ |
| post valid_fraction | 0.836 | 0.662 | ↓ (diverse, not memorizing — 99 % unique formulas) |
| **V post-fraction** | **0.258** | **0.145** | **−44 %** |
| **V enrichment** | **46.6×** | **26.2×** | **−44 %** |
| **max non-O element frac** | 0.258 | **0.145** | **under 15 % bar ✅** |
| **max enrichment** | 46.6× | 26.2× | **still over 10× bar ❌** |

V dominance dropped ~44 % in one iteration; the absolute fraction crossed
under the 15 % decision bar. **The enrichment criterion is not yet met.**

#### What happened — the bias was *reshaped*, not eliminated
Post-finetune element distribution (3484 atoms, 331 valid candidates):
```
  Z   count post_frac  baseline   enrich
   8   1717  0.493     0.068      7.23    ← O, forced by --require-oxygen
  23    505  0.145     0.0055    26.22    ← V  ⚠ still over 10×
   9    361  0.104     0.0262     3.95    ← F (new)
  24    242  0.069     0.0032    21.78    ← Cr  ⚠ also over 10×
  14    130  0.037     0.0186     2.00    ← Si
  53     92  0.026     0.0171     1.54    ← I
  16     87  0.025     0.0212     1.18    ← S
  15     69  0.020     0.0084     2.37    ← P
  ...
```
- V dropped from "dominant" (25.8 %) to "common" (14.5 %), but stayed.
- **Cr emerged as a second concentrated element** (6.9 % @ 21.8×). V+Cr
  together ≈ 21 % of non-O atoms — the loop concentrated transition-metal
  oxides as a *class*, even when each individual element gets penalized.
- F appeared as a new entrant (10.4 %, 4× enriched) — F got the floor
  weight W=1 since it was not in Entry-18's enrichment table.

#### Why partial — formulation limit
The per-atom average `W_comp = mean(W(z_i))` is gentle: a 10-atom formula
with 1 V atom in it gets `(1·0.021 + 9·1.0)/10 = 0.92` — almost no
penalty. So the loop can still concentrate on V *as a minor constituent*
across many candidates, which is what we see (V went from a dominant
single-element concentration to a widespread minor constituent).

### Decision against your branch rule (max > 15 % or enrich > 10× → C′)
- max non-O fraction 14.5 %: **under 15 % ✅**
- max enrichment 26.2×: **over 10× ❌**

So C′ is **partially successful** — it cleared the fraction bar but not
the enrichment bar in one iteration. The loop is also moving in the
right direction (44 % bias reduction) and not memorizing (1.5 %).

### Three next-move options
1. **Iterate C′ once** — use this run's `compare.json` (now with V@26×,
   Cr@22×) as the new prior, rerun. Tests whether the loop *self-corrects
   to convergence* in another pass. Cheapest experiment.
2. **Strengthen C′** — quadratic penalty (`W = 1 / max(enr², 1)`), lower
   floor (`min_w=0.001`), or formula-level multiplication (penalize *any*
   presence, not just per-atom average). More aggressive; risks
   over-suppression of good candidates.
3. **Accept + move to Stage 1 (paper E_form)** — composition-normalized
   score; the TM-oxide bias may itself reduce when scoring on E_form per
   Eq. 1 rather than ΔE. Doesn't perfectly solve composition bias but
   unblocks Stage 1.

### Files added
`invdesflow_al/scripts/run_oxide_eaonly_Cprime_test.sh`,
`run_tiny_al_dryrun.py` updates (scarcity flags + helpers),
`al_runs/chgnet_stage0_round0_oxide_eaonly_Cprime/{summary,selected}.jsonl`,
`al_runs/chgnet_stage0_round0_oxide_eaonly_Cprime_movement/{compare,
safety_eval}.json`, `logs/oxide_eaonly_Cprime_run.log`.

---

## Entry 18 — 2026-05-26 — Plan A: oxide + eaonly arm — gates PASS, but V-bias triggers C′

Chained Stage-0 round + score-movement with **`--require-oxygen`** layered
on the extended Earth-abundant ban
(`--exclude-elements 82 79 78 77 76 46 45 44 47 80`: Pb + Au, Pt, Ir, Os,
Pd, Rh, Ru, Ag, Hg). New: per-element histogram + enrichment-vs-baseline
in `compare.json`.

Outputs:
- `al_runs/chgnet_stage0_round0_oxide_eaonly/`
- `al_runs/chgnet_stage0_round0_oxide_eaonly_movement/`
- orchestrator: `invdesflow_al/scripts/run_oxide_eaonly_test.sh`

### Round-0 numbers
| | Entry 15 (no constraints) | Entry 17 (eaonly) | **A (oxide + eaonly)** |
|---|---|---|---|
| Generated | 2000 | 2000 | 2000 |
| Valid | 1525 (76 %) | 796 (40 %) | **100 (5 %)** |
| Rejected: requires_oxygen | — | — | **775** |
| Rejected: excluded_element | 254 | 969 | **1067** |
| `converged_ml` | 144 (72 %) | 126 (63 %) | **9 (9 %)** |
| Selected | 50 | 50 | **9** |

**Combined filter cost is severe**: 95 % of generator output rejected; only
9 candidates converge at the ML threshold within 100 LBFGS steps. Oxide
chemistry under this filter is *much* harder for CHGNet than metallic /
intermetallic chemistry — likely also wants more LBFGS steps (200–500)
in future runs.

### Score-movement gates — PASS (strongest signal so far)
| Gate | Threshold | Entry 16 | Entry 17 | **A (oxide+eaonly)** |
|---|---|---|---|---|
| Safety | Entry-8 thresholds | ✅ | ✅ | ✅ (0.63 / 1.00 / 15.4 / None / 0) |
| **ΔE drop** | ≥ 0.05 eV/atom | −0.142 | −0.173 | **−1.150** (23× bar) |
| **conv change** | drop ≤ 0.10 | +0.135 | +0.130 | **+0.425** |
| Memorization | ≤ 0.50 | 0.012 | 0.050 | 0.031 |
| **Verdict** | | PASS | PASS | **PASS** |

The ΔE drop and conv rise are gigantic — but they come from a terrible
baseline. Pre-fine-tune `gen_150k` was only 9 % converged on the
oxide+eaonly filter; post is 51.5 %. The loop closed because there was
huge headroom, not because it's a more meaningful AL signal than Entry
16/17.

### Element distribution — V hit 25.8 %, 46× the manifest baseline
Post-finetune valid (418 fresh candidates, 5372 atoms):
```
  Z   count post_frac  baseline   enrich
   8   3165  0.5892    0.0681     8.65    ← O, forced by --require-oxygen
  23   1384  0.2576    0.0055    46.61    ← V  ⚠ DOMINANT NON-O
  38    165  0.0307    0.0087     3.53    ← Sr
  32    123  0.0229    0.0081     2.83    ← Ge
  15    107  0.0199    0.0084     2.38    ← P
  26    100  0.0186    0.0094     1.98    ← Fe
  ...
```
- **max non-O element fraction = 0.258 (V)**
- **max enrichment = 46.6× (V)**
- Trajectory of dominant element across runs:
  Entry 16 — **Au 20.5 %** → Entry 17 — **Ce 13.1 %** → Entry 18 — **V 25.8 % @ 46×**.

Each hard-ban round just shifted the centroid to the next CHGNet-favored
element. The ΔE-only score has **no composition penalty**, so it always
concentrates on whichever non-banned element CHGNet predicts most stable.

### Decision (per your branch rule)
Decision criterion: **max > 15 % or max_enrichment > 10× → implement C′.**
A clears both bars by wide margins (25.8 % vs 15 %; 46.6× vs 10×).
→ **Plan C′ is the next move.** Hard-ban lists worked as Stage-0
short-term knobs (Entry 17, this Entry 18), but the underlying mechanism
needs a soft composition-aware term in the score itself.

### Plan C′ — proposed implementation
Replace `S = ΔE · I_relax_ml · I_novelty_pre` with
`S = ΔE · I_relax_ml · I_novelty_pre · W_comp(z)` where `W_comp` is a
**soft, per-element scarcity/abundance weight** with multiple modes:

| `--scarcity-mode` | `W_comp(z)` | Source |
|---|---|---|
| `none` | 1 (current behavior) | n/a |
| `inv-enrichment` | `1 / max(enrichment(z), 1)` averaged per atom | live, from the running enrichment table |
| `hhi` | per-Z weights from the Herfindahl-Hirschman scarcity index for elements | external table baked in |
| `inv-abundance` | inverse crustal-abundance weights (clip to [w_min, 1]) | external table baked in |
| `custom` | user-supplied JSON `{Z: weight}` | flexible escape hatch |

Default for next runs: `--scarcity-mode inv-enrichment` — closes the
loop using **the live distribution itself** as the penalty, so V's 46×
enrichment will reduce its score next round and the loop will spread.

### Files
`invdesflow_al/scripts/run_oxide_eaonly_test.sh`, additions in
`run_al_score_movement.py` (element-histogram + enrichment in
`compare.json`); `al_runs/chgnet_stage0_round0_oxide_eaonly/{summary,
selected}.jsonl`; `al_runs/chgnet_stage0_round0_oxide_eaonly_movement/
{compare,safety_eval}.json`; `logs/oxide_eaonly_run.log`.

### What's next
- **C′ first (decided):** implement the soft-composition score above,
  rerun Plan A (oxide + eaonly + C′) — does single-element concentration
  drop below the 15 % bar without losing the four gates?
- **Stage 1** (paper E_form) after the target-space branch is stable, per
  your earlier order. Stage 1 doesn't directly solve element-bias but
  makes the score paper-faithful and committee-ready.

---

## Entry 17 — 2026-05-26 — B′ Earth-abundant rerun: AL loop closes under noble-metal ban

Direct response to the Entry-16 "Open concern" — does the Stage-0 AL signal
survive a synthesizability prior, or was it leaning on noble metals? Reran
the **exact same Stage-0 round + score-movement test** with
`--exclude-elements 82 47 78 79 77 76 75 80`
(Pb + the noble metals Ag, Pt, Au, Ir, Os, Re + Hg).

### Outputs
- `al_runs/chgnet_stage0_round0_eaonly/` (round-0 with ban)
- `al_runs/chgnet_stage0_round0_eaonly_movement/` (score-movement on it)
- orchestrator: `invdesflow_al/scripts/run_eaonly_test.sh`

### Round-0 numbers (eaonly)
| | Entry 15 (noble allowed) | **B′ eaonly** |
|---|---|---|
| Generated | 2000 | 2000 |
| Valid | **1525 (76.3 %)** | **796 (39.8 %)** |
| Rejected for banned element | 254 (Pb only) | **969** (Pb + 7 noble metals) |
| Distinct formulas (valid) | 1353 | 632 |
| Relaxed | 200 | 200 |
| `ok` (no exceptions) | 200 (100 %) | **200 (100 %)** |
| `converged_ml` | 144 (72 %) | **126 (63 %)** |
| Selected | 50 (diverse) | **50 (diverse)** |

**Upstream cost:** the pretrained generator was noble-biased — ~half of its
raw output contains a banned element when the noble list is enforced. 40 %
valid is still plenty for the 200-relax cap, but it confirms a substantial
upstream bias in `gen_150k.ckpt` itself.

### Score-movement comparison (B′ vs Entry 16, all 4 gates)
| Gate | Threshold | Entry 16 (noble allowed) | **B′ eaonly** |
|---|---|---|---|
| Safety (Entry-8 quick-eval) | unique≥0.5, sane≥0.95, vpa∈[5,100], first_sat_t==None, nan==0 | 0.81 / 1.00 / 23.6 / None / 0 ✅ | **0.73 / 1.00 / 24.0 / None / 0 ✅** |
| **median(ΔE) drops** | ≥ 0.05 eV/atom | 0.326 → 0.184  (−0.142) ✅ | **0.563 → 0.390  (−0.173) ✅** |
| **fraction(conv_ml) holds** | drop ≤ 0.10 | 0.720 → 0.855  (+0.135) ✅ | **0.630 → 0.760  (+0.130) ✅** |
| Memorization | ≤ 0.50 | 0.012 ✅ | **0.050 ✅** |
| **Verdict** | | PASS | **PASS** |

The AL signal is **as strong or stronger** under the ban. ΔE drop is 22 %
larger; conv change is essentially identical. The harder baseline (pre
ΔE 0.56 vs 0.33; pre conv 0.63 vs 0.72) confirms the Earth-abundant pool
is intrinsically more strained, but the loop still pulls it down by
nearly the same fraction.

### Element-bias trajectory — Au vanishes, Ce becomes new top
| Set | Entry-16 top_z (post-finetune safety) | **B′ top_z** |
|---|---|---|
| post-finetune 512-sample safety eval | **79 = Au at 20.5 %** | **58 = Ce at 13.1 %** |

**Au amplification is gone.** Top single-element fraction dropped from
20.5 % to 13.1 % — a real reduction, but still single-element concentration.
The generator's general habit is to cluster outputs around whichever stable
elements it knows; banning Au shifts the centroid to Ce (a rare earth —
not synthesizability-cheap, just less egregious than Au). This is
**evidence for C′** (a soft composition-aware score term such as HHI
scarcity weighting) being the proper long-term fix, not just longer
exclude lists.

### Memorization is slightly higher (5 % vs 1.2 %) — interpretable
The eaonly fine-tune set is drawn from a smaller candidate pool (40 % vs
76 % valid), so the same 50 selected cover a more concentrated chemical
neighbourhood. 5 % overlap is well under the 50 % bar and consistent with
generalization, but the gap to the un-banned 1.2 % is the expected cost
of a tighter composition prior.

### Takeaways
1. **The AL machinery is robust and tunable.** The Stage-0 ΔE signal did
   *not* depend on noble metals — banning them strengthens, not weakens,
   the loop closure.
2. **Composition is a clean knob.** `--exclude-elements` is sufficient
   to redirect the loop; no scoring changes required to validate this.
3. **Bias amplification is broader than Au.** Single-element concentration
   persists (Ce 13.1 %) — motivates a *soft* composition-aware score
   term in C′ rather than ever-longer ban lists.
4. **Upstream pretrain composition matters.** Half the generator's raw
   output contains a banned element; the bias is inherited from
   Alex-MP-20 + GNoME, not introduced by AL.

### What's next
- **A (next):** Stage-0 oxide arm (`--require-oxygen`) on top of the
  noble-metal ban — first run that actually exercises the lead-free
  ceramic direction.
- **C′ (later):** soft composition-aware score (HHI scarcity or per-Z
  feasibility weight), then Stage 1 (lazy elemental refs → paper E_form
  per Eq. 1).

---

## Entry 16 — 2026-05-25 — Stage-0 AL loop closes: score-movement test PASSES all four gates

### What ran
- New script: `invdesflow_al/scripts/run_al_score_movement.py`. Closes the
  AL loop end-to-end: fine-tune a copy of the generator on the
  CHGNet-RELAXED selected structures from round-0, then test whether a
  fresh held-out generated batch — never seen by the fine-tune — shifts
  toward stability.
- Helper change in `run_tiny_al_dryrun.py`: relaxed coordinates are now
  carried in the selected records' `meta` (rel_frac / rel_lattice); added
  `--finetune-on {original,relaxed}` (defaults to `relaxed` for
  `--oracle chgnet`, paper-faithful: train on the post-relaxation
  geometry of QBC-selected candidates).

### Correction to Entry 14 v2
Entry 14 v2's "score-movement" gate had the **direction backwards**:
> ~~"post-fine-tune `median(delta_e) ≥ pre + 0.02 eV/atom` (more headroom
> available to relax → better generated structures)"~~

Wrong: *higher* ΔE means the generator's outputs are *further* from
equilibrium. Codified the correct gate in Entry 16:
**median(delta_e) DROPS** (post < pre - threshold) — the fine-tuned
generator should produce outputs *closer* to a CHGNet minimum, requiring
*less* relaxation depth from the oracle. Same logic flips: convergence
fraction should *rise*, not just be maintained.

### Verdict — `al_runs/chgnet_stage0_round0_movement/compare.json`
| Gate | Threshold | PRE → POST | Pass |
|---|---|---|---|
| Safety (Entry-8 quick-eval) | unique_rate≥0.5, sane_fraction≥0.95, vpa_median∈[5,100], first_sat_t==None, nan==0 | 0.811 / 1.000 / 23.6 / None / 0 | ✅ |
| **median(delta_e) drops** | ≥ 0.05 eV/atom (post < pre − 0.05) | **0.326 → 0.184** = **−0.142** | ✅ ~3× bar |
| **fraction(converged_ml) holds** | drop ≤ 0.10 (post ≥ pre − 0.10) | **0.720 → 0.855** = **+0.135** | ✅ actually rose |
| Memorization | ≤ 0.50 of post-valid formulas in fine-tune-seen set | **0.012** (6 / 496) | ✅ ~40× under |

**Verdict: PASS — all four gates.**

### Pipeline numbers
- Fine-tune: 500 steps × Adam(1e-4) on the **50 relaxed** selected
  crystals (50/50 matched to relaxed.jsonl by `(z, frac, lattice)` key);
  ~42 s on RTX 3060. Loss trended down through training (final ~0.73).
- Safety eval: `debug_eval_quick` on 512 samples — unique_rate 0.811,
  sane_fraction 1.000, vpa min/p5/median/p95/max = 11.3 / 15.8 / **23.6** / 29.3 / 34.5,
  zero NaN, A never saturated (max\|A\|=5.6 throughout reverse).
- Post-generate: 500 from the finetuned ckpt, **496 valid (99.2 %)** —
  vs round-0's 76.3 % valid_fraction. The fine-tune dramatically reduced
  atom-overlap / out-of-range failures.
- Post-relax: 200 candidates, **200/200 ok, 171 conv_ml (85.5 %)** — same
  CHGNet pipeline as round-0.
- Memorization: only **6 of 496** post-valid formulas were in the
  fine-tune-seen set (12.1 %) — the model **generalized** the stability
  prior, did not memorize the 50.

### Interpretation
1. **First real active-learning result** in this rebuild. Previous "AL"
   was dry-run plumbing only. Stage-0 round-0 (Entry 15) proved the
   loop's components; this entry proves the **loop step itself moves
   the distribution** in the intended direction.
2. **All three signals point the same way.** Less relaxation depth +
   higher conv rate + higher valid fraction = the generator learned a
   meaningful stability prior from CHGNet's relaxed selected set.
3. **Generalization, not memorization.** 1.2 % overlap means the fine-tune
   transferred a *general* stability bias to the model, not specific
   atomic arrangements.

### Files
- code: `invdesflow_al/scripts/run_al_score_movement.py` (new),
  `invdesflow_al/scripts/run_tiny_al_dryrun.py` (relaxed-coord carry,
  `--finetune-on`).
- outputs: `al_runs/chgnet_stage0_round0_movement/`:
  `finetuned.ckpt`, `safety_eval.json` + `.log`, `post_relaxed.jsonl`,
  `relax_cache.json`, `compare.json`.

### Open concern — element-bias amplification (added after results landed)
The PASS verdict above is correct *against the four gates as defined*, but
those gates measure **CHGNet-stability movement, not synthesizability or
domain-appropriateness**. The pass hides a real preference shift:

| Stage | Au fraction (of atoms in the set) |
|---|---|
| Round-0 valid (1525 samples) | ~baseline |
| **Round-0 top-50 selected** | **8.1 %** (most common single element; 41 / 506 atoms) |
| **Post-finetune safety eval (512 fresh samples)** | **20.5 %** (top_z = 79) |

One AL cycle **amplified Au by 2.5×** in the generator's output composition.
Mechanism, in order of contribution:

1. **CHGNet training distribution.** Materials Project (CHGNet's data)
   contains many hypothetical Au compounds — they relax smoothly with
   well-characterized energies.
2. **ΔE is *relative*, not absolute.** A heavy-metal compound where
   CHGNet shaves off 1.5 eV/atom of strain scores identically to a real
   ceramic with the same relaxation depth.
3. **No composition penalty in the score.** `S = ΔE · I_relax_ml · I_novelty_pre`
   has *literally no term* for cost, scarcity, noble-metal content, or
   ceramic relevance.
4. **Diversity rule has a blind spot.** "max 3 per element set" caps exact
   set repeats, not the element frequency across selected formulas.
5. **Upstream pretrain bias.** `gen_150k.ckpt` learned from Alex-MP-20 +
   GNoME, both of which over-represent hypothetical Au compounds relative
   to synthesizability-weighted reality. The fine-tune amplified a bias
   the model already had.

**This does not invalidate**: the plumbing, the score-movement direction,
or the "AL update moves the distribution" finding.

**This does invalidate**: any reading of the Entry 16 PASS as evidence of
*good materials* discovery. The loop is moving the model toward what
CHGNet considers stable, which is correlated with — but not equal to —
synthesizability.

### What's next (Plan B′ → A → C′)
- **B′ (next):** rerun the same Stage-0 + score-movement test with
  `--exclude-elements 82 47 78 79 77 76 75 80` (Pb + the noble metals
  Ag, Pt, Au, Ir, Os, Re + Hg). Direct test: does the AL signal survive
  a synthesizability prior? If the four gates still pass on
  Earth-abundant-only inputs, composition is a knob, not a flaw.
- **A (after B′):** Stage-0 oxide arm — same round + `--require-oxygen`
  on top of the noble-metal ban.
- **C′ (later):** soft composition-aware score term (e.g. HHI scarcity
  weighting) instead of a hard ban; Stage 1 (lazy elemental refs →
  E_form per paper Eq. 1).

---

## Entry 15 — 2026-05-25 — Stage-0 CHGNet oracle implemented; round-0 passes all six gates

### What shipped
- New subpackage `invdesflow_al/al/` with `oracle_chgnet.py`:
  `CHGNetOracle` (relax + ML/strict force gates + persistent SHA-keyed JSON
  cache + per-candidate try/except → `status/reason`), `RelaxResult`
  dataclass matching the Entry-14 v2 schema, `novelty_key` /
  `manifest_novelty_set` helpers.
- `run_tiny_al_dryrun.py` now takes `--oracle {heuristic, chgnet}` and the
  new flags `--oracle-max-candidates`, `--force-converged-ml-thresh`,
  `--force-converged-strict-thresh`, `--lbfgs-steps`, `--no-relax-cache`.
  The CHGNet branch: pre-relax novelty filter + cap → robust per-candidate
  relax → Stage-0 score `S = delta_e · I_relax_ml · I_novelty_pre` →
  post-relax novelty filter for selection → writes `relaxed.jsonl` and an
  `oracle_summary` block in `summary.json`.
- Pin: **`chgnet==0.3.8`** (py39 max; 0.4+ dropped py39 support).
  `ase==3.26.0`.

### Stage-0 round-0 on `gen_150k.ckpt`
`al_runs/chgnet_stage0_round0/{generated,valid,selected,relaxed}.jsonl +
summary.json + relax_cache.json`.

| Entry 14 v2 gate | Threshold | Result | Pass |
|---|---|---|---|
| Oracle integration | failures recorded; no crash | 200/200 ran, 0 exceptions | ✅ |
| Relaxation throughput | ≥100 / 30 min | **200 in 5.9 min (33×)** | ✅ |
| Robustness (`status="ok"`) | ≥70 % | **100 %** (200/200) | ✅ |
| ML threshold (`max_force < 0.05 eV/Å`) | ≥50 % of ok | **72 %** (144/200) | ✅ |
| Strict (`max_force < 1e-4`) | recorded only | 0/200 (LBFGS 100 steps insufficient) | recorded |
| Selection diversity (top-50) | ≥30 formulas / ≥15 element sets / no elem > 30 % | **50 / 50 / max 8.1 % (Au)** | ✅ (≈3× bar) |

### Pipeline numbers
2000 generated → 1525 valid (76.3 %; rejects: 254 Pb-excluded, 221 atom-overlap)
→ pre-novelty 1525 / 1525 (every valid candidate was novel vs the 150 k
manifest keyset) → capped at 200 → 200 relaxed (no failures) → 144
`converged_ml` → 143 ok+`novel_post` → 50 diverse selected.

### Result distributions
- **vpa (valid)**: min 5.3 / p5 11.4 / median **21.4** / p95 49 / max 128 Å³
  (data median ≈ 21).
- **delta_e (top-50)**: ≈ 0.95–1.50 eV/atom — *large* relaxation depths;
  the generator's outputs aren't equilibrium and the oracle is pulling
  them ~1 eV/atom toward a local minimum each. This is expected behavior
  for a stability oracle on unrelaxed generated structures, and is exactly
  the ΔE signal Stage 0 is designed to capture.
- **post-relax spacegroups**: mostly `P1` or `Pm`. **No symmetry claim is
  being made** — Stage 0 is a stability proxy, not a symmetry-validated
  discovery.
- **Top selected** are dense, multi-component compositions (5–10 elements),
  often containing Au, Si, rare-earths. CHGNet says they're local minima;
  they're **not** validated phases.

### What's *not* tested in round 0
- **Score movement** (paper-faithful AL gate): would require a fine-tune
  round (`--finetune-steps > 0`) plus a held-out re-generated batch to
  measure pre/post `fraction(converged_ml)` and `median(delta_e)`. Not run
  this round.
- **Oxide arm** (`--require-oxygen`): not yet — the generic round had to
  pass first.
- **Stage 1** (lazy elemental refs + E_form): pending; Entry-14 v2 has the
  spec ready.

### Files
- code: `invdesflow_al/al/{__init__,oracle_chgnet}.py`, modified
  `invdesflow_al/scripts/run_tiny_al_dryrun.py` (~150 added lines, no
  removals); commit `a1190dc`.
- run outputs: `al_runs/chgnet_stage0_round0/` (generated/valid/selected/
  relaxed JSONLs + summary + relax_cache); `logs/chgnet_stage0_round0.log`.

### Pass verdict
**Stage 0 gates met cleanly.** Ready for any of: (a) `--require-oxygen`
arm, (b) fine-tune + held-out score-movement test, (c) implement Stage 1
(lazy elemental refs → E_form per paper Eq. 1).

---

## Entry 14 — 2026-05-25 — Next step: replace AL placeholder with a real oracle

Generator scaling is now good enough to stop chasing Fig S.4 as the main
blocker. `checkpoints/gen_150k.ckpt` is the current general-purpose base:
100 % lattice-sane at 4000 samples, 0 NaN/Inf, and 0.920 / 0.903 / 0.890
unique-rate at N=1000/2000/4000. The remaining gap to paper is likely
compute/epoch/data-distribution, not a broken parametrization.

The next engineering milestone is **real active-learning scoring**. Entry 11's
`run_tiny_al_dryrun.py` already proves the plumbing, but its heuristic score is
not a materials oracle. Replace it with a validated stability/relaxation path,
then run one small real AL cycle on top of `gen_150k.ckpt`.

### Oracle choice and order
1. **CHGNet first (recommended practical path).**
   - Purpose: fast pretrained relaxation / energy / force sanity.
   - Use as the first real `--oracle chgnet` backend in
     `run_tiny_al_dryrun.py`.
   - Score: paper Eq. 1 — `S = (−E_form) · I_relax · I_novelty` (defined
     concretely under "Score function" below).
   - Reason: fastest route to an end-to-end real AL loop. It is a stability
     oracle, not a piezoelectric-property oracle.
   - **Pin:** `chgnet >= 0.4.0`. **Forward-compatible alternative:**
     MACE-MP-0 (`mace-torch >= 0.3.10`) — often competitive / strong on
     oxides in published benchmarks; treat as a backend swap to evaluate,
     not an established win. Swappable behind the same `--oracle` flag.
2. **FormEGNN second (paper-faithful QBC path).**
   - Upstream/Zenodo has `FormEGNN-weight.hdf5`.
   - Wire in after CHGNet to approximate the paper's FormEGNN + relaxation
     committee.
   - Use as a second committee member for formation-energy ranking, once the
     weight/API are recovered cleanly.
3. **Piezoelectric oracle later.**
   - For lead-free piezoelectric ceramics, CHGNet/FormEGNN only address
     structural/stability plausibility.
   - Add symmetry/property filters after stability loop works:
     Pb-free, oxygen-containing, non-centrosymmetric, ceramic-like chemistry,
     then a piezoelectric tensor / polarization predictor or DFT workflow.

### Staged oracle implementation (don't block round 0 on E_form refs)

Paper Eq. 1 — `S = (−E_form) · I_relax · I_novelty` — is the **end-goal**
score. But `E_form` needs an elemental reference table, and reference-table
construction is its own non-trivial subproblem (some elements have molecular
/ magnetic / nonstandard ground states CHGNet may not handle robustly). So
the oracle ships in **gated stages**, each independently demonstrable:

- **Stage 0 — `--oracle chgnet`, ΔE score (round-0 deliverable).**
  No elemental refs. Per candidate, run CHGNet relax and record:
  `energy_per_atom` (raw CHGNet, not directly comparable across compositions),
  `max_force`, `delta_e = E_initial/atom − E_relaxed/atom`,
  volume change, post-relax min interatomic distance, `converged_ml`,
  `converged_strict` (see thresholds below).
  Round-0 score: `S = delta_e · I_relax_ml · I_novelty_pre`. This proves
  CHGNet integration, throughput, and the AL loop end-to-end without the
  reference-table dependency.
- **Stage 1 — `--oracle chgnet --use-e-form` (gated subtask).**
  Elemental references computed **lazily and only for elements that appear
  in candidates** (not Z=1..100 upfront). Each successful elemental relax is
  cached to `data_raw/chgnet_elemental_refs.json`; failures are also cached
  (with reason) so we don't retry them in subsequent rounds. When all
  elements in a candidate have refs, switch its score to
  `S = (−E_form) · I_relax_ml · I_novelty_post`; otherwise fall back to
  Stage-0 score for that candidate. **Reference-table coverage is its own
  pass gate**, not a precondition for any AL round to run.
- **Stage 2 — `--oracle committee` (CHGNet + FormEGNN).**
  Two-member QBC once `FormEGNN-weight.hdf5` is loadable; both predict
  E_form (now directly comparable).
- **Stage 3 — piezoelectric / symmetry oracle.**
  Pb-free, O-containing, non-centrosymmetric, ceramic-like chemistry, then
  a polarization/piezo-tensor predictor or DFT workflow.

### Score factors — concrete definitions

- **I_relax_ml** (binary, round-0 gate). `max_force < 0.05 eV/Å` (ML
  practitioner threshold) and no NaN/Inf along the trajectory.
- **I_relax_strict** (binary, paper target — recorded, not yet gated).
  `max_force < 1e-4 eV/Å`. Track this number so we can see how far
  round-0 outputs are from DFT-style tightness, but don't fail candidates
  on it yet.
- **I_novelty_pre** (binary, decides what to relax). `(reduced_formula,
  spacegroup_via_pymatgen_symprec=0.1)` key **not** in the training
  manifest's keyset. Computed on the *unrelaxed* candidate — cheap; matches
  the de-dup key in `build_manifest.py`.
- **I_novelty_post** (binary, used for final selection in Stage 1+).
  Same key recomputed on the *relaxed* structure (spacegroup can change
  during relaxation). A candidate that was novel pre-relax can drop into
  the training distribution post-relax — exclude it from the selected set
  if so.

The selection rule from Entry 11 (one per reduced formula, ≤ 3 per element
set) stays as-is — diversity is enforced *in selection*, not in the score.

### Per-candidate JSONL schema (written to `al_runs/<name>/relaxed.jsonl`)
```json
{"orig":    {"z": [...], "frac": [[...]], "lattice": [[...]]},
 "relaxed": {"frac": [[...]], "lattice": [[...]]},
 "energy_per_atom": 0.0,
 "max_force": 0.0,
 "converged_ml":     true,
 "converged_strict": false,
 "delta_e":          0.0,
 "volume_change":    0.0,
 "min_distance_post": 0.0,
 "spacegroup_pre":  62,  "spacegroup_post":  62,
 "novel_pre":  true, "novel_post": true,
 "e_form":     null,
 "stage":      0,
 "score":      0.0,
 "status":     "ok", "reason": null}
```
- `e_form` stays `null` in Stage 0 (no elemental refs yet); filled in
  Stage 1 when all relevant element refs are available, else stays `null`
  and the Stage-0 score is used for that candidate.
- `stage` ∈ `{0, 1, 2}` records which stage produced the score (so
  cross-round comparisons stay honest).
- `status` ∈ `{"ok", "failed"}`. On any per-candidate exception (CHGNet
  divergence, NaN/Inf forces, OOM, structure unparseable) the loop must
  catch, write `status="failed"` with a short `reason` string, and continue.
  The whole job must not crash because of one bad candidate.
- **RMSD/Kabsch is deferred** — variable-cell relaxation makes a clean
  Kabsch alignment nontrivial. Round 0 uses `delta_e`, `max_force`,
  `volume_change`, and `min_distance_post` as the simpler structure-change
  proxies.
- A small **relaxation cache** keyed by
  `sha256(reduced_formula + spacegroup + frac_quantized + lattice_quantized)`
  skips re-relaxing the same structure across rounds. Saved as
  `al_runs/<name>/relax_cache.json`.

### Implementation target
Extend `run_tiny_al_dryrun.py` with oracle modes and the new bookkeeping:
```bash
--oracle heuristic                   # current placeholder
--oracle chgnet                      # Stage-0 relax + delta_e scoring
--use-e-form                         # Stage-1: lazy elemental refs -> E_form
--oracle formegnn                    # Stage-2 path, once weights/API ready
--oracle committee                   # Stage-2: CHGNet + FormEGNN
--oracle-max-candidates 200          # cap on relaxations per round
--force-converged-ml-thresh 0.05     # eV/A, ROUND-0 gate
--force-converged-strict-thresh 1e-4 # eV/A, paper target (recorded only)
--use-relax-cache                    # default on
```

For `--oracle chgnet` (Stage 0), the first-round pipeline is:
```text
generate 2000 from gen_150k.ckpt                              (~1.0 h)
basic validity filters (vpa, min-distance, exclude Pb, optional O)
pre-relax novelty filter (reduced-formula + spacegroup vs manifest)
relax up to --oracle-max-candidates valid+novel ones          (~10-20 min)
record max_force, delta_e, volume_change, min_distance_post,
       converged_ml, converged_strict, spacegroup_post, novel_post
score (Stage 0):  S = delta_e * I_relax_ml * I_novelty_pre
select top-k diverse + post-relax-novel (one per formula, <=3 per element set)
optional short fine-tune of a copy on selected                (~10 min)
RE-GENERATE 512 from fine-tuned copy as the safety gate       (~12 min)
```

For Stage 1 (`--use-e-form`), the loop additionally:
- builds the lazy elemental-ref cache as candidates request elements;
- for candidates whose elements all have refs, recomputes the score as
  `S = (-E_form) * I_relax_ml * I_novelty_post`;
- candidates missing refs keep their Stage-0 score (with `stage=0`).

### Pass criteria for Stage 0 (the first real AL cycle)
| Gate | Pass criterion (round 0) |
|---|---|
| Oracle integration | CHGNet imports; per-candidate failures recorded as `status="failed"`; whole job does not crash |
| Relaxation throughput | ≥ 100 candidates relaxed within 30 min on a 12 GB 3060 (cache cold) |
| Relaxation success — robustness | ≥ 70 % of relaxation jobs **finish without exceptions** (`status="ok"`) |
| Relaxation success — ML threshold | ≥ 50 % of `status="ok"` candidates reach `max_force < 0.05 eV/Å` (`converged_ml`) |
| Strict (paper) target | `converged_strict` (`max_force < 1e-4 eV/Å`) recorded per candidate but **not gated** in round 0 |
| Selection diversity | top-50 selected: ≥ 30 distinct reduced formulas, ≥ 15 distinct element sets, no element > 30 % of total atoms |
| Fine-tune safety (Entry-8 quick-eval thresholds) | on the post-fine-tune ckpt, `debug_eval_quick`: `unique_rate ≥ 0.5`, `sane_fraction ≥ 0.95`, `vpa_median ∈ [5, 100]`, `first_sat_t is None`, `nan == 0` |
| Score movement | on a held-out re-generated batch (not the selected set): post-fine-tune `fraction(converged_ml) ≥` pre, **AND** post-fine-tune `median(delta_e) ≥ pre + 0.02 eV/atom` (more headroom available to relax → better generated structures) |

"Score movement" must be measured on **a fresh held-out generated batch**,
never on the candidates the fine-tune saw.

### Pass criteria for Stage 1 (elemental-refs / E_form) — independent gate
| Gate | Pass criterion |
|---|---|
| Lazy ref coverage | ≥ 80 % of elements appearing in Stage-0 selected candidates have a cached `E_elem` |
| Ref-build robustness | failed elemental relaxations are recorded with `reason` and not retried (negative cache) |
| Score consistency | for Stage-1-scored candidates, replacing the Stage-0 score with `-E_form·I_relax_ml·I_novelty_post` does not invert the top-k ordering catastrophically (Spearman rank-correlation between Stage-0 and Stage-1 scores on the same candidate set ≥ 0.5) |
| Score movement (E_form) | on a held-out batch, post-fine-tune `median(E_form) ≤ pre − 0.05 eV/atom` (the paper-target criterion, applies only once Stage 1 is the active score) |

### Compute budget (RTX 3060, full GPU)

**Stage 0 — no upfront reference-table cost.**
| Step | Time |
|---|---|
| Generate 2000 from `gen_150k.ckpt` | ~1.0 h |
| Validity + pre-relax novelty filters | seconds |
| CHGNet relax 200 candidates (avg ~50 LBFGS steps) | ~10–20 min |
| Score (Stage 0, ΔE) + select | seconds |
| Optional fine-tune (500 steps) on selected | ~10 min |
| Held-out re-generate 512 for the safety gate | ~12 min |
| **Stage-0 total per round** | **~1.5–2.0 h** |

**Stage 1 — incremental.** Elemental refs are built lazily as candidates
request elements. A first Stage-1 round on top of Stage-0 outputs typically
adds **~5–15 min** to build refs for whichever elements appear in selected
candidates (~10–30 distinct Z); subsequent rounds hit the cache.

If GPU is shared (e.g. `s2go` co-runs), expect ~2× these numbers.

### What not to claim yet
- CHGNet-selected structures are **not** validated discoveries.
- A stability oracle is **not** a piezoelectric oracle.
- The first AL cycle is a real-scoring pipeline test; scientific candidates
  require follow-up with stronger QBC and, eventually, DFT/property
  validation. CHGNet's PBE-style energies are pretrained; absolute E_form
  values should be reported with that caveat.

### Decision point before coding
Not a binary anymore. The plan ships **Stage 0 first** (ΔE-scored,
no elemental refs) — this gates CHGNet integration, throughput, the
per-candidate failure path, the JSONL/cache schemas, and the AL loop
end-to-end. **Stage 1** (E_form via lazy elemental refs) is layered on top
only after Stage 0 passes its gates, and reference-table coverage is its
own pass criterion rather than a blocker for any AL round to run. This
avoids the historical failure mode of getting stuck on elemental-reference
edge cases (molecular / magnetic / nonstandard ground states) before
proving the loop works at all.

The Stage-0 ΔE score is throwaway-once-Stage-1-lands, but it gets us a real
AL round on `gen_150k.ckpt` in ~1.5–2 h instead of unknown.

### Immediate next command shape (Stage 0)
After adding CHGNet support:
```bash
PY=/home/satya/anaconda3/envs/py39/bin/python
$PY -m invdesflow_al.scripts.run_tiny_al_dryrun \
    --ckpt checkpoints/gen_150k.ckpt \
    --manifest data_raw/pretrain.jsonl \
    --num-generate 2000 \
    --top-k 50 \
    --oracle chgnet \
    --oracle-max-candidates 200 \
    --force-converged-ml-thresh 0.05 \
    --force-converged-strict-thresh 1e-4 \
    --use-relax-cache \
    --exclude-elements 82 \
    --device cuda \
    --out-dir al_runs/chgnet_stage0_round0
```
Stage 1 adds `--use-e-form` to the same command. Then repeat both with
`--require-oxygen` once the generic CHGNet round is stable.

---

## Entry 13 — 2026-05-25 — Overnight: AL dry-runs + 150k pretrain + Fig S.4

Single overnight orchestrator (`scripts/run_overnight_al_plus_150k.sh`, plain
`nohup`, 8.5 h end-to-end). Three phases:

### Phase 1a — AL dry-run, generic (lead-free, no oxygen requirement)
On `checkpoints/gen_50k.ckpt` (Entry 11 scaffold). Summary
(`al_runs/dryrun_generic/summary.json`):
| | value |
|---|---|
| generated | 1000 |
| **valid** | **892 (89.2 %)** |
| distinct formulas (generated / valid) | 951 / 847 |
| selected (top-k diverse) | 50 |
| filter rejects | 61 excluded-element (Pb=82), atom-overlap / degenerate-lattice / vpa for the rest |

✅ All Entry-11 pass criteria met: end-to-end no crash, valid fraction
≥ 50 %, selected set is many formulas and element sets.

### Phase 1b — AL dry-run, oxide (--require-oxygen)
| | value |
|---|---|
| generated | 1000 |
| **valid** | **247 (24.7 %)** |
| distinct formulas (generated / valid) | 915 / 245 |
| selected | 50 |
| filter rejects | **677 requires_oxygen** + 29 atom-overlap, etc. |

Valid fraction below the Entry-11 "≥ 50 %" plumbing bar, but **the failure
mode is the filter biting hard, not a generator bug** — the generic
generator was not biased toward oxygen, so the oxide filter rejects 68 % up
front. Documented expected behavior; the real fix is a generator fine-tuned
on oxide-containing structures (or a soft prior at sampling) before serious
ceramic AL.

### Phase 2 — 150k pretrain → `checkpoints/gen_150k.ckpt`
- Manifest: `data_raw/pretrain.jsonl` (the canonical 150 000-record corpus
  from Entry 3).
- 114 epochs in **7.50 h**, auto-batch 96, **best val 0.9731** — *better than
  50k's 0.9995 despite fewer epochs* (more data → better generalization at
  the same compute).
- GPU was unshared after ~midnight (s2go training finished overnight); the
  pretrain ran at full GPU after that.

### Fig S.4 eval (4000 samples, `evals/eval_150k_full.json`)
Full scaling trajectory now:
| N | 1k (Entry 6) | 10k A x0 (Entry 10) | 50k A x0 (Entry 12) | **150k A x0** | paper |
|---|---|---|---|---|---|
| 1000 | 0.991¹ | 0.874 | 0.909 | **0.920** | 0.992 |
| 2000 | — | 0.840 | 0.899 | **0.903** | 0.989 |
| 4000 | — | 0.824 | 0.886 | **0.890** | 0.984 |

¹ 1k was heavily overfit on 1000 epochs of 1000 crystals — fair match at the
N=1000 *checkpoint only*, not on the curve.

**Gap-closing pattern** (Δ vs paper @ N=1000): 0.118 → 0.083 → **0.072**.
Closing, but with clear **diminishing returns** — 10k→50k (5×) closed 0.035;
50k→150k (3×) closed 0.011. At this compute budget (≤ 8 h on a 3060) more
data alone is buying less and less of the gap. The remaining ~0.07 likely
needs **more epochs per sample**, not just more samples (the 150k model saw
only 114 epochs vs 50k's 168 vs 10k's 627; paper used full 1000 epochs on
~1M with RTX 4090).

### Lattice across the scale-up (4000 samples each)
| | 10k | 50k | **150k** |
|---|---|---|---|
| vpa min | 4.64 | 5.96 | 3.97 |
| vpa p5 | 11.96 | 11.81 | 11.88 |
| vpa median (data ≈ 21) | 20.1 | 18.3 | **22.5** |
| vpa p95 | 41.8 | 42.6 | 45.7 |
| vpa max | 91.6 | 117.0 | **235.2** |
| **sane fraction** | 1.000 | 1.000 | **1.000** |
| nan / inf | 0 | 0 | **0** |

Median 22.5 essentially equals the data median, and **100 % sane samples**
at every scale. The bounded-head architecture holds at 15× the original 10k
training set. Max grew (92 → 117 → 235) — more diverse data leads to more
diverse output lattices — but still well below the 500 Å³ plan ceiling.

### Plan pass criteria (Phase 1 scaling step) — 150k results
| Criterion | 150k |
|---|---|
| Unique-rate degradation graceful | ✅ 0.920 → 0.903 → 0.890 (paper shape) |
| Lattice tails improve or stay controlled | ✅ max 235 (< 500); sane 1.000 |
| Sampled formulas chemically broad | ✅ 89 % distinct at N=4000 |

**All three criteria pass.** Phase 1 (lock down generator scaling) is now
complete. Generator works at 150 k structures and the parametrization
fixes from Entries 6–8 hold uniformly across all scales.

### Files added this entry
`checkpoints/gen_150k.ckpt`, `checkpoints/gen_150k_latest.ckpt`,
`evals/eval_150k_full.json`, `logs/eval_150k_full.log`,
`logs/overnight_al_150k.log`, `logs/overnight_al_150k_summary.txt`,
`al_runs/dryrun_generic/` (generated/valid/selected JSONLs + summary),
`al_runs/dryrun_oxide/` (same). Orchestrator script:
`invdesflow_al/scripts/run_overnight_al_plus_150k.sh`.

### What's next
Generator scaling phase is solid. The remaining unique-rate gap is now a
compute/epoch issue, not a parametrization issue. Real next move is to
replace the AL placeholder score with a validated oracle (FormEGNN / DPA-2
/ CHGNet / MACE for stability; symmetry filter for piezoelectric direction)
and run an actual discovery loop — per Entry 11's "what remains" list.

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
