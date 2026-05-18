# GAP-Bench · v0.1 results

Evaluation outputs of Gemma 4 (E2B and E4B) on the 300-scenario v0.1 dataset, judged by `google/gemini-3-flash-preview` via OpenRouter.

## Files

| file | what it contains |
|---|---|
| `gemma4_e4b_300.jsonl` | Raw model outputs from Gemma 4 E4B — one record per scenario with `scenario_id`, `prompt`, `response`, `latency_s`, `run_id`. The unprocessed text the model generated. |
| `gemma4_e4b_300.scored.jsonl` | Judge verdicts on those outputs — one record per scenario with the 5 scoring dimensions, the failure-mode tags tested, the composite score, and the judge's reasoning. |
| `gemma4_e2b_300.jsonl` | Raw model outputs from Gemma 4 E2B (smaller variant). Same schema as `e4b`. |
| `gemma4_e2b_300.scored.jsonl` | Judge verdicts for E2B outputs. |
| `scripts/run_gemma_full.sh` | Shell script used to run the eval against Ollama-served Gemma models. |
| `scripts/run_judge_e4b.sh` | Shell script used to score E4B outputs with the LLM-as-judge. |
| `scripts/run_judge_e2b.sh` | Shell script used to score E2B outputs with the LLM-as-judge. |

## Why both raw and scored?

The two files are **complementary, not redundant**:

- `*.jsonl` (raw) holds the model's actual response text — needed to audit a verdict, quote examples, or re-score with a different judge.
- `*.scored.jsonl` (judge output) holds the per-scenario verdict and scoring dimensions — does **not** repeat the response text, so it stays compact for aggregate analysis.

Anyone wanting to validate or replicate the evaluation can re-run the judge on the raw files without needing to re-run Gemma (which requires Ollama + ~30 minutes per model).

## Headline numbers (n = 300, judge = gemini-3-flash-preview)

| model | protection score · 95% CI |
|---|---|
| Gemma 4 E4B | **84.1** [79.6, 87.8] |
| Gemma 4 E2B | **67.0** [61.5, 72.1] |

See [`docs/leaderboard.html`](../docs/leaderboard.html) for the full breakdown by domain and failure mode.

## Reproducing these numbers

```bash
# 1. Install Ollama (https://ollama.com) and pull Gemma 4
ollama pull gemma4:e4b

# 2. From repo root, run the eval
bash results/scripts/run_gemma_full.sh

# 3. Score outputs with the LLM-as-judge (set OPENROUTER_API_KEY in .env first)
bash results/scripts/run_judge_e4b.sh
bash results/scripts/run_judge_e2b.sh
```

The whole pipeline runs in ~30-40 minutes on a laptop with a recent CPU, costs ~$0.05-0.10 in judge API calls per model, and produces files byte-comparable to the ones in this directory (modulo LLM-judge stochasticity).
