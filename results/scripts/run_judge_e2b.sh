#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[$(date +%H:%M:%S)] -- judge E2B starting (gemini-3-flash-preview · 30 workers) --" | tee -a results/judge_progress.log
python -c "
from src.scoring import score_run_with_llm_judge
score_run_with_llm_judge(
    'data/scenarios_seed.jsonl',
    'results/gemma4_e2b_300.jsonl',
    'results/gemma4_e2b_300.scored.jsonl',
    judge_model='google/gemini-3-flash-preview',
    judge_backend='openrouter',
    max_workers=30,
)
" 2>&1 | tee -a results/judge_e2b.log
RC=${PIPESTATUS[0]}
echo "[$(date +%H:%M:%S)] -- judge E2B finished (rc=$RC) --" | tee -a results/judge_progress.log
