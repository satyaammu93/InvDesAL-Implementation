#!/bin/bash
# Overnight orchestrator (~9 h):
#   Phase 1 : AL dry-run experiments on gen_50k.ckpt (plumbing test)
#       1a) generic (lead-free, no oxygen requirement)
#       1b) oxide  (lead-free, --require-oxygen)
#   Phase 2 : 150k pretrain (the next scaling step after 50k)
# Each step logs to logs/; final summary to logs/overnight_al_150k_summary.txt.
# Plain nohup (not setsid) per Entry 9. May share GPU with s2go training.

set -u
PY=/home/satya/anaconda3/envs/py39/bin/python
ROOT=/home/satya/projects/invdes/InvDesFlow-AL
cd "$ROOT"
mkdir -p logs al_runs checkpoints
SUM="$ROOT/logs/overnight_al_150k_summary.txt"
LOG="$ROOT/logs/overnight_al_150k.log"
exec >>"$LOG" 2>&1
echo "=== overnight (AL dry-runs + 150k pretrain) start $(date) ==="

# ---- Phase 1a: generic AL dry-run (lead-free, no oxygen requirement) ----
echo "[$(date)] Phase 1a: AL dry-run generic (lead-free)"
$PY -m invdesflow_al.scripts.run_tiny_al_dryrun \
    --ckpt checkpoints/gen_50k.ckpt \
    --manifest data_raw/pretrain_50k.jsonl \
    --num-generate 1000 --gen-batch 256 --top-k 50 \
    --device cuda --seed 1 \
    --out-dir al_runs/dryrun_generic \
    || echo "[$(date)] WARN: dryrun_generic non-zero exit"

# ---- Phase 1b: oxide AL dry-run (lead-free, --require-oxygen) ----
echo "[$(date)] Phase 1b: AL dry-run oxide (lead-free, --require-oxygen)"
$PY -m invdesflow_al.scripts.run_tiny_al_dryrun \
    --ckpt checkpoints/gen_50k.ckpt \
    --manifest data_raw/pretrain_50k.jsonl \
    --num-generate 1000 --gen-batch 256 --top-k 50 \
    --require-oxygen --device cuda --seed 2 \
    --out-dir al_runs/dryrun_oxide \
    || echo "[$(date)] WARN: dryrun_oxide non-zero exit"

# ---- Phase 2: 150k pretrain (the next scaling step) ----
echo "[$(date)] Phase 2: 150k pretrain (budget 7.5 h)"
# 150k manifest already exists at data_raw/pretrain.jsonl (the canonical 150k
# diversity-sampled corpus built way back; 150,000 records, 7032 buckets).
$PY -m invdesflow_al.scripts.train_generator \
    --manifest data_raw/pretrain.jsonl --device cuda \
    --auto-batch --workers 0 --max-hours 7.5 \
    --ckpt checkpoints/gen_150k.ckpt \
    --log-every 200 --ckpt-every 1000 \
    || echo "[$(date)] WARN: 150k train non-zero exit"

echo "=== overnight done $(date) ==="

# brief summary file
{
  echo "overnight (AL dry-runs + 150k pretrain) summary  --  $(date)"
  echo ""
  echo "--- AL dry-run: generic ---"
  cat al_runs/dryrun_generic/summary.json 2>/dev/null | head -30
  echo ""
  echo "--- AL dry-run: oxide ---"
  cat al_runs/dryrun_oxide/summary.json 2>/dev/null | head -30
  echo ""
  echo "--- 150k training tail ---"
  tail -6 logs/overnight_al_150k.log
  ls -la checkpoints/gen_150k*.ckpt 2>/dev/null
} > "$SUM"
echo "summary -> $SUM"
