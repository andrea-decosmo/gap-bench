# GAP-Bench · evaluation code

Python package that runs the GAP-Bench benchmark against any AI model and produces a scored leaderboard entry.

## Modules

| file | what it does |
|---|---|
| `eval.py` | Runs a model on all 300 scenarios. Supports two backends: **Ollama** (local, e.g. `gemma:4b`) and **OpenRouter** (closed/paid models via API). Produces a `*.jsonl` file with one record per scenario containing the model's response. |
| `scoring.py` | Scores raw model outputs with an **LLM-as-judge** (default: `google/gemini-3-flash-preview` via OpenRouter). For each scenario the judge sees: the user's message, the model's response, and the source-referenced ground truth — and returns a 5-dimension verdict. Produces a `*.scored.jsonl` file. |
| `schema.py` | JSON Schema validation for the scenarios. Used to QA the dataset. |
| `run_leaderboard.py` | Aggregates one or more `*.scored.jsonl` files into a leaderboard summary (protection score with bootstrap 95% CI, per-domain breakdown, failure-mode matrix). |

## Typical eval pipeline

```bash
# 1. Run a model on the 300 scenarios
python -m src.eval --model gemma4:e4b --backend ollama \
    --scenarios data/scenarios_seed.jsonl \
    --out results/my_run.jsonl

# 2. Score with the LLM-as-judge
python -m src.scoring --in results/my_run.jsonl \
    --out results/my_run.scored.jsonl

# 3. Build the leaderboard
python -m src.run_leaderboard results/*.scored.jsonl
```

## Environment variables

`scoring.py` requires `OPENROUTER_API_KEY` (the judge runs via OpenRouter). Copy `.env.example` to `.env` at the repo root and fill it in.

`eval.py` with `--backend openrouter` also needs `OPENROUTER_API_KEY`. With `--backend ollama` (default), only requires a local Ollama daemon (`ollama serve`).

## Output schemas

- See `results/README.md` for the per-record structure of `*.jsonl` (raw) and `*.scored.jsonl` (judge verdicts).
- See `data/README.md` and `data/schema.json` for the scenario record structure.
