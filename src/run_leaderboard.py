"""Aggregate scored (LLM-judged) JSONL files into a leaderboard.

Reads results/*.scored.jsonl files. Each must already contain per-scenario
dimensions + composite produced by an LLM judge (either via OpenRouter or
via the manual Claude-Code workflow in src.judge_collect).

Outputs:
- docs/data/leaderboard.json
- prints summary table

Usage:
    python -m src.run_leaderboard
    python -m src.run_leaderboard --results-dir results --pattern "*.scored.jsonl"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.scoring import aggregate_from_scored_file


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--pattern", default="*.scored.jsonl",
                   help="Glob pattern for scored files (default: *.scored.jsonl)")
    p.add_argument("--out", default="docs/data/leaderboard.json")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scored_files = sorted(results_dir.glob(args.pattern))
    if not scored_files:
        print(f"No files matching '{args.pattern}' in {results_dir}/")
        print("Run the LLM judge first:")
        print("  - via OpenRouter: src.scoring.score_run_with_llm_judge(...)")
        print("  - via Claude chat: python -m src.judge_batch + src.judge_collect")
        return

    board = {"version": "0.2.0", "models": {}}
    for jsonl in scored_files:
        model_key = jsonl.stem.removesuffix(".scored")  # filename without .scored or .jsonl
        agg = aggregate_from_scored_file(str(jsonl))
        board["models"][model_key] = agg
        print(f"\n=== {model_key} ===")
        print(f"  n_scenarios:      {agg.get('n_scenarios')}")
        print(f"  protection_score: {agg.get('protection_score')} "
              f"[{agg.get('ci_95_low')}-{agg.get('ci_95_high')}]  (Wilson 95% CI)")
        print(f"  TPR (sensitivity): {agg.get('TPR')}")
        print(f"  TNR (specificity): {agg.get('TNR')}")
        per_dom = agg.get("per_domain_score", {})
        if per_dom:
            print("  per-domain:")
            for dom, s in per_dom.items():
                print(f"    {dom:14s} {s}")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(board, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
