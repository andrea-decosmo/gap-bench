"""GAP-Bench evaluation harness.

Supports:
- Ollama local backend (Gemma 4 family, free)
- Anthropic Claude API direct
- Google Gemini API direct
- OpenRouter (proxy to many providers, single API key)

Audit & reproducibility:
- Every output JSONL file starts with a `_config` record that pins down the
  exact configuration used for the run.
- Retry logic with exponential backoff on transient errors.
- max_tokens raised to 2048 to avoid truncating long protective responses.

Usage:
    # Local Gemma 4 via Ollama (free)
    python -m src.eval --model gemma4:e4b --backend ollama \\
        --scenarios data/scenarios_seed.jsonl \\
        --out results/gemma4_e4b.jsonl

    # Closed baseline via OpenRouter (one key, many providers)
    export OPENROUTER_API_KEY=sk-or-...
    python -m src.eval --model anthropic/claude-sonnet-4.6 --backend openrouter \\
        --scenarios data/scenarios_seed.jsonl \\
        --out results/claude_sonnet.jsonl --sample-only

    # Anthropic direct (alternative)
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m src.eval --model claude-sonnet-4-6 --backend anthropic ...
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from src.schema import load_scenarios


# Auto-load .env at the project root (so OPENROUTER_API_KEY etc. work without manual export)
def _load_dotenv():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        os.environ.setdefault(k, v)


_load_dotenv()


# ---- Configuration (tracked in every output for audit) ----

EVAL_CONFIG: dict[str, Any] = {
    "version": "0.2",
    "max_tokens": 2048,                # ↑ from 512: avoid truncating responses
    "temperature": 0.0,                # deterministic
    "think": False,                    # disable Gemma 4 "thinking" mode
    "prompt_design": "bare",           # no system priming; just user message
    "prompt_version": "v0.2-bare",
    "scoring_version": "v0.2-llm-judge",  # see src/scoring.py
    "retry_max": 5,
    "retry_backoff_seconds": [2, 4, 8, 16, 32],
    "transient_errors": (
        "TimeoutError", "ConnectionError", "ConnectionResetError",
        "RateLimitError", "ServerError", "OSError", "RequestException",
        "APITimeoutError", "APIConnectionError",
    ),
    "seed": 42,
}


# ---- Bare prompt design (qa_direct only in v0.1) ----

def build_prompt(scenario: dict[str, Any]) -> str:
    """Return the prompt sent to the model.

    v0.1 design: bare. No system instruction, no priming on the task.
    Just the user's question. We measure DEFAULT model behavior, not
    best-case under task-specific priming.
    """
    return scenario["scenario_text"]


# ---- Retry helper ----

def _is_transient(exc: Exception) -> bool:
    name = type(exc).__name__
    return name in EVAL_CONFIG["transient_errors"] or any(
        m in str(exc).lower() for m in ("timeout", "connection", "rate limit", "temporarily")
    )


def call_with_retry(fn: Callable, *args, **kwargs) -> Any:
    """Call `fn(*args, **kwargs)` with retry+backoff on transient errors."""
    backoffs = EVAL_CONFIG["retry_backoff_seconds"]
    last_exc: Exception | None = None
    for attempt in range(EVAL_CONFIG["retry_max"]):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_transient(exc) or attempt == EVAL_CONFIG["retry_max"] - 1:
                raise
            sleep_s = backoffs[min(attempt, len(backoffs) - 1)]
            print(f"  retry {attempt + 1}/{EVAL_CONFIG['retry_max']} in {sleep_s}s "
                  f"(transient: {type(exc).__name__})", flush=True)
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


# ---- Backends ----

def call_ollama(model: str, prompt: str, host: str | None = None) -> str:
    """Call Ollama chat API. Requires Ollama daemon running locally."""
    try:
        import ollama  # type: ignore
    except ImportError:
        raise RuntimeError("pip install ollama")
    client = ollama.Client(host=host or os.environ.get("OLLAMA_HOST",
                                                       "http://localhost:11434"))
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": EVAL_CONFIG["temperature"],
            "num_predict": EVAL_CONFIG["max_tokens"],
            "seed": EVAL_CONFIG["seed"],
        },
        stream=False,
    )
    try:
        resp = client.chat(**kwargs, think=EVAL_CONFIG["think"])
    except TypeError:
        resp = client.chat(**kwargs)
    msg = resp.get("message", {}) if isinstance(resp, dict) else getattr(resp, "message", {})
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    return (content or "").strip()


def call_anthropic(model: str, prompt: str) -> str:
    """Call Anthropic Messages API. Requires ANTHROPIC_API_KEY env var."""
    try:
        from anthropic import Anthropic  # type: ignore
    except ImportError:
        raise RuntimeError("pip install anthropic")
    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=EVAL_CONFIG["max_tokens"],
        temperature=EVAL_CONFIG["temperature"],
        messages=[{"role": "user", "content": prompt}],
    )
    parts = []
    for block in resp.content:
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def call_gemini(model: str, prompt: str) -> str:
    """Call Gemini API. Requires GOOGLE_API_KEY env var."""
    try:
        from google import genai  # type: ignore
    except ImportError:
        raise RuntimeError("pip install google-genai")
    client = genai.Client()
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "max_output_tokens": EVAL_CONFIG["max_tokens"],
            "temperature": EVAL_CONFIG["temperature"],
        },
    )
    return (resp.text or "").strip()


def call_openrouter(model: str, prompt: str) -> str:
    """Call OpenRouter API (OpenAI-compatible).

    Requires OPENROUTER_API_KEY env var.

    Model identifiers follow OpenRouter convention, examples:
      - 'anthropic/claude-sonnet-4.6'
      - 'google/gemini-2.5-flash'
      - 'openai/gpt-4o-mini'
      - 'meta-llama/llama-3.3-70b-instruct'

    See https://openrouter.ai/models for the full catalog.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise RuntimeError("pip install openai")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        # OpenRouter recommends including these for routing/attribution
        default_headers={
            "HTTP-Referer": "https://github.com/andrea-decosmo/gap-bench",
            "X-Title": "GAP-Bench",
        },
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=EVAL_CONFIG["max_tokens"],
        temperature=EVAL_CONFIG["temperature"],
    )
    content = resp.choices[0].message.content or ""
    return content.strip()


BACKENDS = {
    "ollama": call_ollama,
    "anthropic": call_anthropic,
    "gemini": call_gemini,
    "openrouter": call_openrouter,
}


# ---- Main eval loop ----

def evaluate(model: str, backend: str, scenarios_path: str, out_path: str,
             limit: int | None = None, sample_only: bool = False,
             filter_id: str | None = None) -> None:
    scenarios = load_scenarios(scenarios_path)
    if filter_id:
        scenarios = [s for s in scenarios if s["id"] == filter_id]
        if not scenarios:
            raise ValueError(f"No scenario found with id={filter_id!r}")
    if sample_only:
        # for closed-baseline budget control
        scenarios = scenarios[:200] if len(scenarios) > 200 else scenarios
    if limit:
        scenarios = scenarios[:limit]

    fn = BACKENDS[backend]
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # --- audit header pinned to every run ---
    run_id = uuid.uuid4().hex[:12]
    started_at = datetime.datetime.now(datetime.UTC).isoformat()
    config_record = {
        "_run_id": run_id,
        "_started_at_utc": started_at,
        "_model": model,
        "_backend": backend,
        "_scenarios_path": str(scenarios_path),
        "_n_scenarios": len(scenarios),
        "_config": EVAL_CONFIG,
    }

    n_ok = n_err = 0
    with out_path_obj.open("w", encoding="utf-8") as fout:
        fout.write(json.dumps(config_record, ensure_ascii=False) + "\n")
        for i, sc in enumerate(scenarios):
            prompt = build_prompt(sc)
            t0 = time.time()
            try:
                response = call_with_retry(fn, model, prompt)
                rec = {
                    "scenario_id": sc["id"],
                    "run_id": run_id,
                    "model": model,
                    "backend": backend,
                    "prompt": prompt,
                    "response": response,
                    "latency_s": round(time.time() - t0, 3),
                }
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                rec = {
                    "scenario_id": sc["id"],
                    "run_id": run_id,
                    "model": model,
                    "backend": backend,
                    "error": f"{type(e).__name__}: {e}",
                    "latency_s": round(time.time() - t0, 3),
                }
                n_err += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i+1}/{len(scenarios)}] {sc['id']} ({rec.get('latency_s', '?')}s) "
                  f"{'ERR' if 'error' in rec else 'OK'}", flush=True)
    print(f"\nDone. {n_ok} OK / {n_err} errors. -> {out_path}")
    print(f"run_id={run_id}")


def main() -> None:
    p = argparse.ArgumentParser(description="GAP-Bench evaluation")
    p.add_argument("--model", required=True, help="Model id (e.g. gemma4:e4b)")
    p.add_argument("--backend", required=True, choices=tuple(BACKENDS))
    p.add_argument("--scenarios", default="data/scenarios_seed.jsonl")
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample-only", action="store_true",
                   help="Use first 200 scenarios (for closed baseline budget)")
    p.add_argument("--filter-id", default=None,
                   help="Run a single scenario by id (e.g. gap-0032). Useful for quick tests.")
    args = p.parse_args()
    evaluate(args.model, args.backend, args.scenarios, args.out,
             args.limit, args.sample_only, args.filter_id)


if __name__ == "__main__":
    sys.exit(main())
