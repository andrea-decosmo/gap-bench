# GAP-Bench

> **A benchmark for AI that protects vulnerable users.**
>
> Not whether a model refuses harmful requests, but whether it recognises a hidden risk in the user's situation and takes the protective action.

Existing safety benchmarks (SafetyBench, HarmBench, OR-Bench, DoNotAnswer) measure whether a model refuses harmful *requests* — the model as a **source of risk**. GAP-Bench asks something different: when a vulnerable user sends a real, ordinary message that hides a real risk, does the model **recognise it** and **act protectively**? — the model as a **shield of the user**.

📊 **Live leaderboard & docs**: <https://andrea-decosmo.github.io/gap-bench/>
🎬 **Kaggle writeup**: see Gemma 4 Good Hackathon 2026 submission

---

## v0.1 headline results (n = 300, judge = `gemini-3-flash-preview`)

| model         | size  | protection score · 95% CI | runs on  |
|---------------|-------|---------------------------|----------|
| Gemma 4 E4B   | 10 GB | **84.1** [79.6, 87.8]     | laptop via Ollama, $0 |
| Gemma 4 E2B   |  3 GB | **67.0** [61.5, 72.1]     | phone-class via Ollama, $0 |

The 17-point gap between the two sizes is dominated by **restraint on safe questions** (TNR +20) rather than catching real risk (TPR +10). The diagnostic matrix in `results/` shows exactly which failure modes drive the gap.

Full per-domain and per-failure-mode breakdown: <https://andrea-decosmo.github.io/gap-bench/leaderboard.html>

---

## Quickstart

```bash
git clone https://github.com/andrea-decosmo/gap-bench
cd gap-bench
pip install -e .

# 1. Pull Gemma 4 via Ollama (one-time, ~3 GB)
ollama pull gemma4:e4b

# 2. Run the eval on all 300 scenarios
python -m src.eval --model gemma4:e4b --backend ollama \
    --scenarios data/scenarios_seed.jsonl \
    --out results/my_run.jsonl

# 3. Score with the LLM-as-judge (set OPENROUTER_API_KEY in .env first)
cp .env.example .env       # edit and add your key
python -m src.scoring --in results/my_run.jsonl \
    --out results/my_run.scored.jsonl

# 4. Print the leaderboard summary
python -m src.run_leaderboard results/*.scored.jsonl
```

Total time on a recent laptop: **~30 minutes**. Total cost: **~$0.05 in judge API calls per model**.

Alternative: run the whole pipeline in your browser on Kaggle — see [`notebooks/gap_bench_kaggle.ipynb`](notebooks/gap_bench_kaggle.ipynb).

---

## What's in this repo

| folder | contents |
|---|---|
| [`data/`](data/) | The benchmark: 300 scenarios across 5 domains and 3 languages, plus schema + failure-mode taxonomy. See `data/README.md`. |
| [`src/`](src/) | Evaluation code: `eval.py` (runs a model), `scoring.py` (LLM-as-judge), `run_leaderboard.py` (aggregate). See `src/README.md`. |
| [`results/`](results/) | Official v0.1 results: raw outputs + judge verdicts for Gemma 4 E4B and E2B, plus the shell scripts used to produce them. See `results/README.md`. |
| [`notebooks/`](notebooks/) | A Kaggle-runnable notebook that reproduces the whole eval in the cloud. |
| [`docs/`](docs/) | Source of the live documentation site (deployed via GitHub Pages). |

---

## Design principles

- **Bidirectional scoring.** Every model is scored on both *risk recognition* and *restraint on safe questions*. Always-warn scores zero; always-reassure scores zero. Only models that distinguish earn credit.
- **Source-referenced ground truth.** Every scenario cites an authoritative document (NICE NG143, BACEN scam patterns, INPS procedures, GDPR articles, …) in its `source` field.
- **15-mode failure taxonomy.** Each scenario carries failure-mode tags. The diagnostic shows *which kind* of failure dominates a model's weakness, not just that it has one.
- **Open from day one.** Dataset, scoring code, judge prompt, methodology — all public. No hidden tricks, no held-out tricks.

---

## License

- **Code** (`src/`, `notebooks/`, `docs/`): Apache License 2.0 · see [`LICENSE`](LICENSE)
- **Dataset and taxonomy** (`data/`): Creative Commons Attribution-ShareAlike 4.0 · see [`data/LICENSE`](data/LICENSE)

---

## Citation

```bibtex
@misc{decosmo2026gap,
  title  = {GAP-Bench: A Benchmark for AI that Protects Vulnerable Users},
  author = {De Cosmo, Andrea},
  year   = {2026},
  url    = {https://github.com/andrea-decosmo/gap-bench}
}
```

---

## Author

Andrea De Cosmo · Founder, Edukai · 2026

GAP-Bench is an independent open-source benchmark, not affiliated with any commercial product. Built as a v0.1 submission to the Gemma 4 Good Hackathon 2026.
