#!/bin/bash
# Plan B' — repeat Stage-0 round + score-movement with noble-metal ban
# (Earth-abundant-only). Tests whether the AL signal from Entry 16 survives
# a synthesizability prior, or whether it leaned entirely on Au/Pt/Ir.
set -u
PY=/home/satya/anaconda3/envs/py39/bin/python
ROOT=/home/satya/projects/invdes/InvDesFlow-AL
cd "$ROOT"
mkdir -p logs
LOG="$ROOT/logs/eaonly_run.log"
exec >>"$LOG" 2>&1
echo "=== B' Earth-abundant Stage-0 + score-movement start $(date) ==="

# Pb (82) + noble metals Ag (47) Pt (78) Au (79) Ir (77) Os (76) Re (75) + Hg (80)
EXCLUDE="82 47 78 79 77 76 75 80"

echo "[$(date)] Round-0 (eaonly): excluding $EXCLUDE"
$PY -m invdesflow_al.scripts.run_tiny_al_dryrun \
    --ckpt checkpoints/gen_150k.ckpt \
    --manifest data_raw/pretrain.jsonl \
    --num-generate 2000 --gen-batch 256 --top-k 50 \
    --oracle chgnet --oracle-max-candidates 200 \
    --lbfgs-steps 100 \
    --force-converged-ml-thresh 0.05 --force-converged-strict-thresh 1e-4 \
    --exclude-elements $EXCLUDE \
    --device cuda \
    --out-dir al_runs/chgnet_stage0_round0_eaonly \
    || echo "[$(date)] WARN: round-0 exited non-zero"

echo "[$(date)] Score-movement (eaonly): excluding $EXCLUDE"
$PY -m invdesflow_al.scripts.run_al_score_movement \
    --ckpt checkpoints/gen_150k.ckpt \
    --round0-dir al_runs/chgnet_stage0_round0_eaonly \
    --manifest data_raw/pretrain.jsonl \
    --finetune-steps 500 \
    --post-num-generate 500 --post-max-relax 200 \
    --exclude-elements $EXCLUDE \
    --device cuda \
    --out-dir al_runs/chgnet_stage0_round0_eaonly_movement \
    || echo "[$(date)] WARN: score-movement exited non-zero"

echo "=== B' done $(date) ==="
