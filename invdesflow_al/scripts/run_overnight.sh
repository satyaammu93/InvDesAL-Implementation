#!/bin/bash
# Unattended overnight pipeline:
#   wait for conversions -> build diversity manifest -> pretrain (wallclock
#   budget, OOM-safe) -> evaluate chemical-formula unique rate vs Fig S.4.
# Every stage logs to logs/ and is safe to leave running.
set -u
PY=/home/satya/anaconda3/envs/py39/bin/python
ROOT=/home/satya/projects/invdes/InvDesFlow-AL
cd "$ROOT"
mkdir -p logs
LOG="$ROOT/logs/overnight.log"
exec >>"$LOG" 2>&1
echo "=== overnight start $(date) ==="

# 1. wait until the GNoME converter process is gone (Alex already finished)
while pgrep -f "convert_datasets gnome" >/dev/null 2>&1; do sleep 20; done
echo "[$(date)] conversions complete: alex=$(wc -l <data_raw/alex_mp_20.jsonl) gnome=$(wc -l <data_raw/gnome.jsonl)"

# 2. memory-safe diversity manifest (combined corpus, <=20 atoms, target 150k)
$PY -m invdesflow_al.scripts.build_manifest \
    --inputs data_raw/alex_mp_20.jsonl data_raw/gnome.jsonl \
    --max-atoms 20 --target-size 150000 --seed 0 \
    --out data_raw/pretrain.jsonl
echo "[$(date)] manifest: $(wc -l <data_raw/pretrain.jsonl) records"

# 3. real pretraining: faithful arch (hidden 512, 6 layers, T=1000),
#    OOM-safe auto batch, fixed wallclock, periodic + best-val checkpoints
$PY -m invdesflow_al.scripts.train_generator \
    --manifest data_raw/pretrain.jsonl --device cuda \
    --auto-batch --workers 0 --max-hours 7.0 \
    --ckpt "$ROOT/generator.ckpt" --log-every 200 --ckpt-every 1000
echo "[$(date)] training finished"

# 4. evaluate against Supplementary Fig. S.4
#    sampling is compute-bound & linear (~0.7s/sample at T=1000): 16k ~= 3.1h,
#    covering the 1k/2k/4k/8k/16k points. gen-batch 512 = good util, ~0.8GB.
CK="$ROOT/generator.ckpt"; [ -f "$CK" ] || CK="$ROOT/generator_latest.ckpt"
$PY -m invdesflow_al.scripts.eval_unique_rate \
    --ckpt "$CK" --manifest data_raw/pretrain.jsonl \
    --max-samples 16000 --gen-batch 512 --device cuda
echo "=== overnight done $(date) ==="
