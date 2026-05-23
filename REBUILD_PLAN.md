# InvDesFlow-AL — Rebuild Plan (from paper + supplementary only)

> Source: `s41524-025-01830-z.pdf` (main, 14 pp) + `41524_2025_1830_MOESM1_ESM.pdf` (supplementary, 10 pp).
> This plan reconstructs the system **from the paper, equations, algorithms, and tables** — not from the repo's existing code.
> Every hyperparameter below is taken verbatim from Table S.2 / S.3 / Table 3 / Table 4 — there are **no remaining unknowns** for the ML models.

---

## 1. System overview

InvDesFlow-AL is a closed-loop active-learning inverse-design framework. Four models wired together by a Query-by-Committee (QBC) loop:

| Component | Role | Spec source |
|---|---|---|
| **Generator** | Diffusion model producing crystals `(A, F, L)` | Methods, Alg. 1 & 2, Table S.2 |
| **FormEGNN** | Formation-energy / E_hull predictor (stability) | ref 18; used in QBC |
| **DPA-2** | External pretrained interatomic potential → relaxation + force convergence | ref 28 (external) |
| **SuperconGNN** | e3nn equivariant Tc predictor | Methods, Fig 5, Table S.3 |

Three discovery tasks share this machinery (workflows fixed by **Fig S.7**):
(a) low formation energy, (b) low E_hull, (c) high-Tc superconductors (Tc > 20 K). Plus a CSP task (Perov-5 / MP-20 / MPTS-52).

---

## 2. Crystal representation

Unit cell `M = (A, F, L)`:
- `A = [a₁…a_N] ∈ ℝ^{n×h}` — chemical species, one-hot.
- `F = [f₁…f_N] ∈ ℝ^{3×N}` — **fractional** coordinates (anchored to lattice → periodicity + symmetry explicit).
- `L = [l₁,l₂,l₃] ∈ ℝ^{3×3}` — lattice basis.
- `N` ≤ 20 atoms; held **fixed** during a generation run.

**Graph construction (generator):** CrystalNN with lattice scaling; neighborhood cutoff **7.0 Å**, max **20** neighbors/atom; structure-match tolerance 0.1.

---

## 3. Generator — diffusion + EGNN

### 3.1 Architecture (Table S.2, exact)
- Decoder: **EGNN**, **6** GNN layers, hidden dim **512**, activation **SiLU**.
- Diffusion steps **T = 1000**.
- Loss weights `(coord / lattice / type) = 1.0 / 1.0 / 20.0`.
- Mixed corruption: `L`, `A` → DDPM (Ho et al.); `F` → score-matching (Song et al.) with **wrapped/periodic** noise `w(·)` into `[0,1)`.

### 3.2 Algorithm 1 — training
```
t ~ U{1,T};  ε_L, ε_A, ε_F ~ N(0,I)
compute √ᾱ_t, √(1−ᾱ_t), σ_t
L_t = √ᾱ_t·L_0 + √(1−ᾱ_t)·ε_L
A_t = √ᾱ_t·A_0 + √(1−ᾱ_t)·ε_A
F_t = w(F_0 + σ_t·ε_F)
(ε̂_L, ε̂_A, ε̂_F) = φ(L_t, A_t, F_t, N, t)
L_lattice = ‖ε_L − ε̂_L‖²
L_atom    = ‖ε_A − ε̂_A‖²
L_coord   = ‖λ_t·∇_{F_t} log q(F_t|F_0) − ε̂_F‖²
L_total   = 1.0·L_coord + 1.0·L_lattice + 20.0·L_atom   # weights from Table S.2
```

### 3.3 Algorithm 2 — sampling (predictor–corrector)
```
init L_T,A_T ~ N(0,I);  F_T ~ U(0,1)
for t = T … 1:
  ε_L,ε_A,ε_F ~ N(0,I)
  (ε̂_L,ε̂_A,ε̂_F) = φ(L_t,A_t,F_t,N,t)
  L_{t−1} = 1/√α_t·(L_t − β_t/√(1−ᾱ_t)·ε̂_L) + √(β_t·(1−ᾱ_{t−1})/(1−ᾱ_t))·ε_L
  A_{t−1} = 1/√α_t·(A_t − β_t/√(1−ᾱ_t)·ε̂_A) + √(β_t·(1−ᾱ_{t−1})/(1−ᾱ_t))·ε_A
  F_{t−½} = w( F_t + (σ_t²−σ_{t−1}²)·ε̂_F + (σ_{t−1}·√(σ_t²−σ_{t−1}²)/σ_t)·ε_F )
  (·, ·, ε̂'_F) = φ(L_{t−1}, F_{t−½}, A_{t−1}, N, t−1)
  d_t = γ·σ_{t−1}²/σ_1²                      # adaptive Langevin step
  F_{t−1} = w( F_{t−½} + d_t·ε̂'_F + √(2 d_t)·ε_F )
return (L_0, A_0, F_0)
```

### 3.4 Optimization / training protocol (Table S.2)
- Optimizer **Adam**, base LR **1e-4**, scheduler **ReduceLROnPlateau** (factor 0.6, patience 30), min LR 1e-4.
- Batch sizes train/val/test = **96/64/64**; 30 preprocessing workers.
- **1000 epochs**, single RTX 4090.

> Only residual unknown: exact noise schedule shape (`β_t`) and `λ_t`. Use a standard cosine/linear DDPM schedule + `λ_t = σ_t` (DiffCSP convention) and document the assumption.

---

## 4. SuperconGNN — Tc predictor

### 4.1 Graph (Eq 4)
- pymatgen local-environment bonding edges **⊔** all distance edges within **r_max = 5.0 Å**, merged.
- `e_ab = MLP^(e)(f_ab | μ(r_ab))`; `V_a^(0) = MLP^(v)(f_a)`.
- node feats: atom-type one-hot + lattice `(α,β,γ,a,b,c)`; edge feats: 2-D one-hot (bond present/absent) + Gaussian-expanded distance. **Input edge features = 2**.

### 4.2 Encoding (Eq 5) — Table S.3, exact
- e3nn tensor-product / TFN layers, **6** convolutional layers, residual connections.
- Scalar features **ns = 128**, vector features **nv = 10**, spherical-harmonics order **ℓ_max = 2**, **third-order representation = True**.
- Equivariant BatchNorm; activation **ReLU**.
- Final **SE(3)-invariant** linear (e3nn) → atom-aggregate → ReLU → `Tc > 0`.

### 4.3 Data + training (Table S.3 + Methods)
- Base 626 conventional-SC points (ref 9), split **0.90/0.05/0.05**; +59 hydride crystals (ref 35) into train; +12 InvDesFlow-AL discoveries into test.
- Radius graph (max 5.0); batch 32/32/32.
- Adam, LR 1e-4, **Warmup Linear Decay** (warmup = 0.5·total steps), min LR 1e-4.
- **200 epochs**, **MSE** loss, RTX 4090.
- Baselines: ALIGNN, ALIGNN-H. Accuracy = predicted Tc exceeds threshold buckets Tc-5…Tc-60.

**Validation target (Table S.1):** Li₂AuH₆ SuperconGNN≈71 (Eliashberg 140 K); K₂GaCuH₆ 48; Na₂LiAgH₆ 41; etc.

---

## 5. Active-learning loop + QBC

Strategies: **Diversity Sampling** (pretrain coverage) · **Expected Model Change** (initial fine-tune on target crystals → iterative generate-and-filter) · **Query-by-Committee** (DPA-2 + FormEGNN/SuperconGNN + DFT).

Scoring functions (`I_cond` = 1 if satisfied else 0):
- **Eq 1** low formation: `S = (−E_form) · I_relax · I_novelty`
- **Eq 2** low E_hull: `S = (−E_hull) · I_relax · I_novelty`
- **Eq 3** superconductor: `S = (T_c^DFT or T_c^SuperconGNN) · I_relax · I_novelty · I_{E_hull<50meV}`

`I_relax`: DPA-2 force convergence `‖F‖ < 1e-4 eV/Å`. `I_novelty`: composition not in Materials Project / existing DBs.

### 5.1 Workflows (Fig S.7 — exact pipeline)
**(a) Low formation energy**
`Alex-MP-20 + GNoME (DS pretrain)` + `GNoME E_form<−0.5` → fine-tune (EMC) → generate candidates → QBC: DPA-2 relax (‖F‖<1e-4) → FormEGNN E_form → novelty filter → retain top → re-fine-tune. Repeat **5 rounds** (gen counts 80707/95586/97379/136784/166663; mean E_form μ = −1.14/−2.03/−2.93/−3.56/−3.77; 577,113 total).

**(b) Low E_hull**
Same but `GNoME E_hull<50 meV` branch; QBC FormEGNN predicts E_hull; novelty filter. → ~1.6M generated, **1,598,551** with E_hull<50 meV.

**(c) High-Tc (Tc>20 K)**
`Superconductor dataset (Tc>20 K)` branch → fine-tune → generate → QBC: SuperconGNN predicts Tc>20 K → DPA-2 relax (‖F‖<1e-4) → DFT validation (superconducting stability) → retain top → re-fine-tune.

Total **10** fine-tuning rounds across tasks.

---

## 6. Datasets

| Use | Dataset |
|---|---|
| Pretraining | Alex-MP-20 (607,683, ref 2) + GNoME 381,000 (ref 1), diversity-sampled |
| Low-formation fine-tune | GNoME subset `E_form < −0.5 eV/atom` |
| Low-E_hull fine-tune | GNoME subset `E_hull < 50 meV` |
| Superconductor fine-tune | Tc > 20 K superconductor set |
| CSP | Perov-5, MP-20, MPTS-52 (test sets **excluded** from pretraining) |
| UHTC | 14 UHTC crystals (ZrB₂, HfC, TiC, …) |
| Released outputs (Zenodo) | 577,113 low-formation (15222702); low-E_hull (15221067); candidate SCs (14644273) |

CSP metric: `RMSE = √( (1/N) Σ‖X_i^pred − X_i^true‖² )` with **Kabsch** alignment. Targets (Table 1): MP-20 0.0423 / Perov-5 0.0703 / MPTS-52 0.0725; match rates 60.83 / 52.86 / 23.72 %.

---

## 7. First-principles parameters (Tables 3 & 4 — for DFT-validation stage only)

- **DFT:** QUANTUM-ESPRESSO, PBE-GGA, ONCV (optimized norm-conserving Vanderbilt) pseudopotentials, cutoff **80 Ry / 320 Ry**, k-mesh **16³ unshifted**, Methfessel-Paxton smearing **0.02 Ry**, phonon DFPT **4×4×4 q-mesh**.
- **EPC:** EPW Wannier interpolation, MLWFs on 4³ k-mesh, projected orbitals Au-5d/H-1s, fine grids e:48³ ph:16³, smearing 90 meV (e) / 0.5 meV (ph), anisotropic Eliashberg, Matsubara cutoff **1.7 eV**. `λ = (1/N_k)Σ… = 2∫α²F(ω)/ω dω`.
- **USPEX baseline (Table 4):** 20 gen/compound, pop 20, 60 % low-enthalpy selection, heredity 50 % / lattice-mut 10 % / perm 20 %, elitism 2, VASP PAW GGA-PBE.

---

## 8. Software requirements

| Layer | Packages |
|---|---|
| Core ML | Python ≥3.10, PyTorch, PyTorch-Geometric, **e3nn**, NumPy, ASE |
| Materials | **pymatgen** (CrystalNN, local env, Kabsch/StructureMatcher) |
| Potential | DPA-2 via DeePMD-kit (external pretrained) |
| Baselines | ALIGNN / ALIGNN-H; USPEX (DFT comparison only) |
| First-principles | QUANTUM-ESPRESSO + EPW + Wannier90; VASP (USPEX baseline only) |

---

## 9. Proposed code structure

```
invdesflow_al/
  data/
    representation.py     # M=(A,F,L); fractional<->cartesian; wrap w(·)
    graph.py              # CrystalNN (cutoff 7.0, 20 nb) for generator
    radius_graph.py       # radius graph (5.0) for SuperconGNN
    datasets.py           # Alex-MP-20, GNoME loaders, DS sampler, filters
  models/
    egnn.py               # 6-layer EGNN, hidden 512, SiLU
    diffusion.py          # noise schedule, Alg.1 train, Alg.2 sample
    formegnn.py           # formation-energy / E_hull predictor (ref 18)
    supercongnn.py        # e3nn TFN: ns=128 nv=10 lmax=2 6 layers
  potentials/
    dpa2.py               # DPA-2 wrapper: relax + ‖F‖<1e-4 convergence flag
  active_learning/
    scoring.py            # Eq 1,2,3 + I_relax,I_novelty,I_{Ehull<50}
    qbc.py                # committee orchestration
    loop.py               # EMC generate→filter→re-finetune (Fig S.7 a/b/c)
  tasks/
    csp.py                # Perov-5/MP-20/MPTS-52; RMSE+Kabsch eval
    low_formation.py
    low_ehull.py
    superconductor.py
  dft/
    qe_inputs.py          # Table 3 QE/EPW input generators
    uspex_config.py       # Table 4 baseline config
  configs/
    generator.yaml        # = Table S.2 verbatim
    supercongnn.yaml      # = Table S.3 verbatim
```

---

## 10. Build phases & milestones

1. **Foundation** — `data/representation.py`, `data/graph.py`, periodic wrap `w(·)`, fractional↔Cartesian, unit tests on a few CIFs. (No deps on models.)
2. **EGNN + diffusion** — `egnn.py`, `diffusion.py`; implement Alg. 1 & 2 exactly; overfit a tiny set to verify the loss/sampler. *Milestone: sample a valid periodic structure.*
3. **SuperconGNN** — `supercongnn.py` per Table S.3; train on 626+59 split. *Milestone: reproduce Table S.1 ordering (Li₂AuH₆ highest).*
4. **FormEGNN + DPA-2 wrappers** — load pretrained weights (Zenodo); expose `predict_eform/ehull` and `relax→(struct, ‖F‖)`.
5. **QBC + AL loop** — `scoring.py` (Eq 1–3), `loop.py` per Fig S.7. *Milestone: one full EMC round end-to-end on a small pool.*
6. **CSP task** — fine-tune on MP-20, RMSE+Kabsch eval. *Milestone: RMSE in the 0.04–0.07 Å range vs Table 1.*
7. **DFT bridge (optional)** — QE/EPW input writers from Table 3 for the validation stage.

---

## 11. Remaining ambiguities (small, documented assumptions)

| Item | Resolution |
|---|---|
| Noise schedule `β_t` shape | Not stated → use cosine DDPM, expose as config. |
| `λ_t` in `L_coord` | Use DiffCSP convention `λ_t = σ_t`; configurable. |
| FormEGNN architecture | Defined in ref 18 ("InvDesFlow-1.0"); plan to load released `FormEGNN-weight.hdf5` rather than retrain. |
| DPA-2 exact variant | External pretrained; pin a DeePMD-kit release. |
| Diversity-sampling selection rule | Paper: "cover different regions of distribution" — implement as composition/space-group stratified sampling, configurable. |

These do **not** block model implementation; each is isolated behind a config flag.

---

## 12. Validation targets (acceptance criteria)

- CSP RMSE: MP-20 ≈0.042 Å, Perov-5 ≈0.070 Å, MPTS-52 ≈0.073 Å (Table 1).
- Generator novelty: chemical-formula unique rate ≈0.99 at 1k samples, ≈0.79 at 256k (Fig S.4).
- SuperconGNN: Tc ranking consistent with Table S.1; Tc-20 bucket accuracy 100 % (Fig 5c).
- Low-formation AL: monotone decreasing mean E_form across 5 rounds (μ ≈ −1.14 → −3.77).
