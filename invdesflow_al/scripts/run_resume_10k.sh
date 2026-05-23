#!/bin/bash
# Option B: resume-train gen_10k.ckpt for ~3h with intra-cycle quick evals.
# 3 cycles: each = 1h resume-train + ~12min quick eval (512 samples + A diagnostics).
# Each cycle's checkpoint and eval are saved separately so we can see the trend.

set -u
PY=/home/satya/anaconda3/envs/py39/bin/python
ROOT=/home/satya/projects/invdes/InvDesFlow-AL
cd "$ROOT"
mkdir -p logs
LOG="$ROOT/logs/resume_10k.log"
exec >>"$LOG" 2>&1
echo "=== resume 10k start $(date) ==="

PREV="$ROOT/gen_10k.ckpt"
for r in 1 2 3; do
  NEXT="$ROOT/gen_10k_r${r}.ckpt"
  EVAL_OUT="$ROOT/eval_r${r}.json"
  echo "[$(date)] --- Cycle $r/3: resume train 1.0h  $(basename $PREV) -> $(basename $NEXT)"
  $PY -m invdesflow_al.scripts.train_generator \
      --manifest data_raw/pretrain_10k.jsonl --device cuda \
      --auto-batch --workers 0 --max-hours 1.0 \
      --resume "$PREV" --ckpt "$NEXT" \
      --log-every 200 --ckpt-every 1000 \
      || echo "[$(date)] WARNING: train cycle $r exited non-zero, continuing"

  echo "[$(date)] --- Cycle $r/3: quick eval (512 samples, A diagnostics)"
  $PY -m invdesflow_al.scripts.debug_eval_quick \
      --ckpt "$NEXT" --manifest data_raw/pretrain_10k.jsonl \
      --n-sample 512 --gen-batch 256 --device cuda \
      --out "$EVAL_OUT" \
      || echo "[$(date)] WARNING: eval cycle $r exited non-zero, continuing"

  PREV="$NEXT"
done
echo "=== resume 10k done $(date) ==="
