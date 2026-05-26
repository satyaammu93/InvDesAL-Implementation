#!/bin/bash
# Plan A + C' — rerun the oxide+eaonly arm with --scarcity-mode inv-enrichment
# using Entry 18's compare.json as the per-Z enrichment prior. Tests whether
# the soft composition-aware score breaks the V-bias amplification (Entry 18
# max non-O element fraction was 25.8 %; bar is 15 %; max enrichment 46.6 %;
# bar is 10x).
set -u
PY=/home/satya/anaconda3/envs/py39/bin/python
ROOT=/home/satya/projects/invdes/InvDesFlow-AL
cd "$ROOT"
mkdir -p logs
LOG="$ROOT/logs/oxide_eaonly_Cprime_run.log"
exec >>"$LOG" 2>&1
echo "=== Plan A + C': oxide+eaonly+inv-enrichment start $(date) ==="

EXCLUDE="82 79 78 77 76 46 45 44 47 80"
PRIOR="$ROOT/al_runs/chgnet_stage0_round0_oxide_eaonly_movement/compare.json"

echo "[$(date)] Round-0 (oxide+eaonly+C'): --scarcity-mode inv-enrichment, prior=$(basename $PRIOR)"
$PY -m invdesflow_al.scripts.run_tiny_al_dryrun \
    --ckpt checkpoints/gen_150k.ckpt \
    --manifest data_raw/pretrain.jsonl \
    --num-generate 2000 --gen-batch 256 --top-k 50 \
    --oracle chgnet --oracle-max-candidates 200 \
    --lbfgs-steps 100 \
    --force-converged-ml-thresh 0.05 --force-converged-strict-thresh 1e-4 \
    --require-oxygen \
    --exclude-elements $EXCLUDE \
    --scarcity-mode inv-enrichment \
    --enrichment-prior "$PRIOR" \
    --device cuda \
    --out-dir al_runs/chgnet_stage0_round0_oxide_eaonly_Cprime \
    || echo "[$(date)] WARN: round-0 exited non-zero"

echo "[$(date)] Score-movement on the C' round-0"
$PY -m invdesflow_al.scripts.run_al_score_movement \
    --ckpt checkpoints/gen_150k.ckpt \
    --round0-dir al_runs/chgnet_stage0_round0_oxide_eaonly_Cprime \
    --manifest data_raw/pretrain.jsonl \
    --finetune-steps 500 \
    --post-num-generate 500 --post-max-relax 200 \
    --require-oxygen \
    --exclude-elements $EXCLUDE \
    --device cuda \
    --out-dir al_runs/chgnet_stage0_round0_oxide_eaonly_Cprime_movement \
    || echo "[$(date)] WARN: score-movement exited non-zero"

echo "=== Plan A + C' done $(date) ==="
