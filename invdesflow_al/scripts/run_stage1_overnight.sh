#!/bin/bash
# Stage 1 overnight orchestrator (Entry 20):
#   Phase 1  pre-warm elemental refs (CHGNet on ase.bulk / diatomic fallbacks)
#   Phase 2  Plan A + C' + Stage 1   (oxide + eaonly + inv-enrichment + E_form)
#   Phase 3  Plan A         + Stage 1 (oxide + eaonly + E_form, no C') — ABLATION
# Each phase logs to logs/; summary to logs/stage1_overnight_summary.txt.
set -u
PY=/home/satya/anaconda3/envs/py39/bin/python
ROOT=/home/satya/projects/invdes/InvDesFlow-AL
cd "$ROOT"
mkdir -p logs
LOG="$ROOT/logs/stage1_overnight.log"
exec >>"$LOG" 2>&1
echo "=== Stage 1 overnight start $(date) ==="

EXCLUDE="82 79 78 77 76 46 45 44 47 80"
REFS="$ROOT/data_raw/chgnet_elemental_refs.json"
PRIOR="$ROOT/al_runs/chgnet_stage0_round0_oxide_eaonly_Cprime_movement/compare.json"
# noble gases + banned elements: don't waste cycles warming refs for these
SKIP_WARM="$EXCLUDE 2 10 18 36 54 86"

# Phase 1: pre-warm refs from the pretrain manifest (skip banned + noble gases)
echo "[$(date)] Phase 1: pre-warm elemental refs from manifest"
$PY -m invdesflow_al.scripts.build_elemental_refs \
    --manifest data_raw/pretrain.jsonl \
    --exclude $SKIP_WARM \
    --out "$REFS" --device cuda \
    || echo "[$(date)] WARN: pre-warm exited non-zero (continuing)"

# Phase 2: Stage-1 with C'  (E_form + inv-enrichment)
echo "[$(date)] Phase 2: Stage 1 + C' chained round-0 + score-movement"
$PY -m invdesflow_al.scripts.run_tiny_al_dryrun \
    --ckpt checkpoints/gen_150k.ckpt \
    --manifest data_raw/pretrain.jsonl \
    --num-generate 2000 --gen-batch 256 --top-k 50 \
    --oracle chgnet --oracle-max-candidates 200 \
    --lbfgs-steps 100 \
    --force-converged-ml-thresh 0.05 --force-converged-strict-thresh 1e-4 \
    --require-oxygen --exclude-elements $EXCLUDE \
    --scarcity-mode inv-enrichment --enrichment-prior "$PRIOR" \
    --use-e-form --elemental-refs-cache "$REFS" \
    --device cuda \
    --out-dir al_runs/chgnet_stage1_Cprime \
    || echo "[$(date)] WARN: stage1+C' round-0 non-zero"
$PY -m invdesflow_al.scripts.run_al_score_movement \
    --ckpt checkpoints/gen_150k.ckpt \
    --round0-dir al_runs/chgnet_stage1_Cprime \
    --manifest data_raw/pretrain.jsonl \
    --finetune-steps 500 \
    --post-num-generate 500 --post-max-relax 200 \
    --require-oxygen --exclude-elements $EXCLUDE \
    --use-e-form --elemental-refs-cache "$REFS" \
    --device cuda \
    --out-dir al_runs/chgnet_stage1_Cprime_movement \
    || echo "[$(date)] WARN: stage1+C' score-movement non-zero"

# Phase 3: Stage-1 without C' (ablation — does E_form alone suppress V/Cr?)
echo "[$(date)] Phase 3: Stage 1 only (no C') ablation"
$PY -m invdesflow_al.scripts.run_tiny_al_dryrun \
    --ckpt checkpoints/gen_150k.ckpt \
    --manifest data_raw/pretrain.jsonl \
    --num-generate 2000 --gen-batch 256 --top-k 50 \
    --oracle chgnet --oracle-max-candidates 200 \
    --lbfgs-steps 100 \
    --force-converged-ml-thresh 0.05 --force-converged-strict-thresh 1e-4 \
    --require-oxygen --exclude-elements $EXCLUDE \
    --use-e-form --elemental-refs-cache "$REFS" \
    --device cuda \
    --out-dir al_runs/chgnet_stage1_noCprime \
    || echo "[$(date)] WARN: stage1 noC' round-0 non-zero"
$PY -m invdesflow_al.scripts.run_al_score_movement \
    --ckpt checkpoints/gen_150k.ckpt \
    --round0-dir al_runs/chgnet_stage1_noCprime \
    --manifest data_raw/pretrain.jsonl \
    --finetune-steps 500 \
    --post-num-generate 500 --post-max-relax 200 \
    --require-oxygen --exclude-elements $EXCLUDE \
    --use-e-form --elemental-refs-cache "$REFS" \
    --device cuda \
    --out-dir al_runs/chgnet_stage1_noCprime_movement \
    || echo "[$(date)] WARN: stage1 noC' score-movement non-zero"

# brief summary
{
  echo "Stage 1 overnight summary  --  $(date)"
  echo
  echo "REFS cache coverage:"
  $PY - <<EOF
import json
d = json.load(open("$REFS"))
ok = sum(1 for v in d.values() if v.get("status")=="ok")
print(f"  total {len(d)}  ok {ok}  failed {len(d)-ok}")
EOF
  for tag in chgnet_stage1_Cprime chgnet_stage1_noCprime; do
    echo
    echo "--- $tag (round-0 oracle_summary) ---"
    $PY -c "
import json
d = json.load(open('al_runs/$tag/summary.json'))
o = d.get('oracle_summary', {})
keys = ['relaxed','ok','converged_ml','converged_strict','selected',
        'use_e_form','e_form_computed_ok','e_form_missing',
        'e_form_coverage_of_ok','scarcity_mode']
for k in keys:
    print(f'  {k}: {o.get(k)}')"
    echo "--- $tag (compare) ---"
    $PY -c "
import json
d = json.load(open('al_runs/${tag}_movement/compare.json'))
print('  verdict:', d.get('verdict'))
print('  gates  :', d.get('gates'))
print('  pre    :', {k:d['pre'][k] for k in ('fraction_converged_ml','delta_e_median','e_form_median')})
print('  post   :', {k:d['post'][k] for k in ('fraction_converged_ml','delta_e_median','e_form_median')})
ed = d.get('element_distribution', {})
print('  max_post_element_frac :', ed.get('max_element_fraction_post'))
print('  max_post_enrichment   :', ed.get('max_enrichment_post'))
print('  top-5 post:')
for e in ed.get('post_finetune_valid_enrichment_top',[])[:5]:
    print(f'    Z={e[\"z\"]:>3}  frac={e[\"fraction\"]:.4f}  enrich={e[\"enrichment\"]}')"
  done
} > "$ROOT/logs/stage1_overnight_summary.txt"
echo "summary -> $ROOT/logs/stage1_overnight_summary.txt"
echo "=== Stage 1 overnight done $(date) ==="
