#!/bin/bash
# Plan A — Stage-0 oxide arm with extended noble-metal ban.
# Same chained shape as Plan B' (run_eaonly_test.sh) but with --require-oxygen
# and the fuller noble-metal list (adds Pt-group: Pd, Rh, Ru).
# Also: element-histogram + enrichment logging in compare.json (script update).
set -u
PY=/home/satya/anaconda3/envs/py39/bin/python
ROOT=/home/satya/projects/invdes/InvDesFlow-AL
cd "$ROOT"
mkdir -p logs
LOG="$ROOT/logs/oxide_eaonly_run.log"
exec >>"$LOG" 2>&1
echo "=== A: oxide + noble-metal ban Stage-0 + score-movement start $(date) ==="

# Pb 82 + Au 79 Pt 78 Ir 77 Os 76 + Pd 46 Rh 45 Ru 44 (Pt-group) + Ag 47 + Hg 80
EXCLUDE="82 79 78 77 76 46 45 44 47 80"

echo "[$(date)] Round-0 (oxide + eaonly): --require-oxygen --exclude-elements $EXCLUDE"
$PY -m invdesflow_al.scripts.run_tiny_al_dryrun \
    --ckpt checkpoints/gen_150k.ckpt \
    --manifest data_raw/pretrain.jsonl \
    --num-generate 2000 --gen-batch 256 --top-k 50 \
    --oracle chgnet --oracle-max-candidates 200 \
    --lbfgs-steps 100 \
    --force-converged-ml-thresh 0.05 --force-converged-strict-thresh 1e-4 \
    --require-oxygen \
    --exclude-elements $EXCLUDE \
    --device cuda \
    --out-dir al_runs/chgnet_stage0_round0_oxide_eaonly \
    || echo "[$(date)] WARN: round-0 exited non-zero"

echo "[$(date)] Score-movement (oxide + eaonly): --require-oxygen --exclude-elements $EXCLUDE"
$PY -m invdesflow_al.scripts.run_al_score_movement \
    --ckpt checkpoints/gen_150k.ckpt \
    --round0-dir al_runs/chgnet_stage0_round0_oxide_eaonly \
    --manifest data_raw/pretrain.jsonl \
    --finetune-steps 500 \
    --post-num-generate 500 --post-max-relax 200 \
    --require-oxygen \
    --exclude-elements $EXCLUDE \
    --device cuda \
    --out-dir al_runs/chgnet_stage0_round0_oxide_eaonly_movement \
    || echo "[$(date)] WARN: score-movement exited non-zero"

echo "=== A done $(date) ==="
