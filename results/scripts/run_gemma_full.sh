#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
TS=$(date +%Y%m%d_%H%M%S)
echo "=== START $TS ===" | tee -a results/run_progress.log

echo "[$(date +%H:%M:%S)] -- E2B starting (300 scenarios) --" | tee -a results/run_progress.log
python -m src.eval --model gemma4:e2b --backend ollama \
    --scenarios data/scenarios_seed.jsonl \
    --out results/gemma4_e2b_300.jsonl 2>&1 | tee -a results/run_e2b.log
E2B_RC=${PIPESTATUS[0]}
echo "[$(date +%H:%M:%S)] -- E2B finished (rc=$E2B_RC) --" | tee -a results/run_progress.log

echo "[$(date +%H:%M:%S)] -- E4B starting (300 scenarios) --" | tee -a results/run_progress.log
python -m src.eval --model gemma4:e4b --backend ollama \
    --scenarios data/scenarios_seed.jsonl \
    --out results/gemma4_e4b_300.jsonl 2>&1 | tee -a results/run_e4b.log
E4B_RC=${PIPESTATUS[0]}
echo "[$(date +%H:%M:%S)] -- E4B finished (rc=$E4B_RC) --" | tee -a results/run_progress.log

echo "=== DONE E2B_rc=$E2B_RC E4B_rc=$E4B_RC at $(date +%H:%M:%S) ===" | tee -a results/run_progress.log
