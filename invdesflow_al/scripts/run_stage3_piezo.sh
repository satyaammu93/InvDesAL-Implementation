#!/bin/bash
# Entry 24 chain: Stage 3 (piezo head) — round-0 then score-movement
#   Plan C scoring:
#     S = (-E_form) * max(|e_max|, floor) * I_relax_ml * I_novelty
#         * I_noncentro * W_comp(inv-enrichment from Entry 22)
#   - Symmetry: --reject-centrosymmetric-post (Plan A)
#   - C':       --scarcity-mode inv-enrichment, prior = Entry 22 compare.json
#               (keeps Cr suppression that worked in Entry 23)
#   - Piezo:    --piezo-head piezo_head.ckpt (val rho 0.72)
#   - Family prior dropped — the trained head does the chemistry steering.
# Stage 1 E_form (--use-e-form), bumped budgets (5000 gen / 200 oracle / 200 LBFGS).
set -u
PY=/home/satya/anaconda3/envs/py39/bin/python
ROOT=/home/satya/projects/invdes/InvDesFlow-AL
cd "$ROOT"
mkdir -p logs

LOG="$ROOT/logs/stage3_piezo.log"
exec >>"$LOG" 2>&1
echo "=== Entry 24 (Stage 3 piezo head) start $(date) ==="

EXCLUDE="82 79 78 77 76 46 45 44 47 80"
REFS="$ROOT/data_raw/chgnet_elemental_refs.json"
PRIOR="$ROOT/al_runs/chgnet_stage1_noCprime_symmetry_movement/compare.json"
PIEZO="$ROOT/checkpoints/piezo_head.ckpt"

echo "[$(date)] Phase 1: round-0 generation + CHGNet + E_form + piezo scoring"
$PY -m invdesflow_al.scripts.run_tiny_al_dryrun \
    --ckpt checkpoints/gen_150k.ckpt \
    --manifest data_raw/pretrain.jsonl \
    --num-generate 5000 --gen-batch 256 --top-k 50 \
    --oracle chgnet --oracle-max-candidates 200 \
    --lbfgs-steps 200 \
    --force-converged-ml-thresh 0.05 --force-converged-strict-thresh 1e-4 \
    --require-oxygen --exclude-elements $EXCLUDE \
    --reject-centrosymmetric-post \
    --scarcity-mode inv-enrichment --enrichment-prior "$PRIOR" \
    --piezo-head "$PIEZO" --piezo-floor 0.05 \
    --use-e-form --elemental-refs-cache "$REFS" \
    --device cuda \
    --out-dir al_runs/chgnet_stage3_piezo \
    || { echo "[$(date)] FATAL: round-0 failed"; exit 1; }

echo "[$(date)] Phase 2: post-finetune score-movement"
$PY -m invdesflow_al.scripts.run_al_score_movement \
    --ckpt checkpoints/gen_150k.ckpt \
    --round0-dir al_runs/chgnet_stage3_piezo \
    --manifest data_raw/pretrain.jsonl \
    --finetune-steps 500 \
    --post-num-generate 500 --post-max-relax 200 \
    --require-oxygen --exclude-elements $EXCLUDE \
    --reject-centrosymmetric-post \
    --use-e-form --elemental-refs-cache "$REFS" \
    --device cuda \
    --out-dir al_runs/chgnet_stage3_piezo_movement \
    || { echo "[$(date)] FATAL: score-movement failed"; exit 1; }

echo "=== Entry 24 done $(date) ==="

# brief summary
{
  echo "Entry 24 summary  --  $(date)"
  echo
  echo "--- round-0 oracle_summary ---"
  $PY -c "
import json
d = json.load(open('al_runs/chgnet_stage3_piezo/summary.json'))
o = d.get('oracle_summary', {})
for k in ['relaxed','ok','converged_ml','converged_strict','selected',
          'post_centrosymmetric','post_non_centrosymmetric',
          'use_e_form','e_form_computed_ok','e_form_coverage_of_ok',
          'scarcity_mode','enrichment_prior','scarcity_n_weights',
          'family_prior','piezo_floor','piezo_e_max_quantiles']:
    print(f'  {k}: {o.get(k)}')"
  echo
  echo "--- score-movement compare ---"
  $PY -c "
import json
d = json.load(open('al_runs/chgnet_stage3_piezo_movement/compare.json'))
print('  verdict:', d.get('verdict'))
print('  gates  :', d.get('gates'))
print('  pre    :', {k:d['pre'][k] for k in ('fraction_converged_ml','delta_e_median','e_form_median')})
print('  post   :', {k:d['post'][k] for k in ('fraction_converged_ml','delta_e_median','e_form_median')})
ed = d.get('element_distribution', {})
print('  max_post_element_frac :', ed.get('max_element_fraction_post'))
print('  max_post_enrichment   :', ed.get('max_enrichment_post'))
print('  top-15 post:')
for e in ed.get('post_finetune_valid_enrichment_top',[])[:15]:
    print(f'    Z={e[\"z\"]:>3}  frac={e[\"fraction\"]:.4f}  enrich={e[\"enrichment\"]}')"
} > "$ROOT/logs/stage3_piezo_summary.txt"
echo "summary -> $ROOT/logs/stage3_piezo_summary.txt"
