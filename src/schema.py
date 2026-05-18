"""Dataclasses + helpers for GAP-Bench scenarios.

Mirrors data/schema.json. Use dict-style for simplicity (jsonl-friendly).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---- Enums ----

LANGUAGES = ("en", "it", "pt-BR")
DOMAINS = ("medical", "scam", "rights", "bureaucracy", "control")
# v0.1: only qa_direct. Multiple-choice removed by design (LLM-judge replaces deterministic MC scoring).
SCENARIO_TYPES = ("qa_direct",)
RISK_LEVELS = ("high", "medium", "low", "none")
REQUIRED_ACTIONS = (
    "warn_high", "warn_medium", "refer_emergency",
    "refer_specialist", "refer_routine", "educate",
    "explain_no_risk", "no_action_needed",
)


# ---- I/O helpers ----

def load_scenarios(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL scenarios from disk. One scenario per line."""
    path = Path(path)
    scenarios = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scenarios.append(json.loads(line))
    return scenarios


def save_scenarios(scenarios: list[dict[str, Any]], path: str | Path) -> None:
    """Save list of dict scenarios as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def validate(scenario: dict[str, Any]) -> list[str]:
    """Quick validation. Returns list of error strings (empty if valid).

    Strict rules:
    - All structural fields must be present.
    - `source` must be a dict with a VERIFIABLE URL (`url_or_doi` must look
      like http(s)://... or doi:..., not just a free-text note).
    - `required_action` must be one of the enum values.
    - `required_content` must be non-empty (else scoring is impossible).
    - v0.1 only supports qa_direct (multiple-choice removed by design).
    """
    errors = []
    required = ("id", "language", "domain", "risk_present",
                "risk_level", "scenario_text", "required_action",
                "required_content", "forbidden_content",
                "failure_modes_tested", "source")
    for k in required:
        if k not in scenario:
            errors.append(f"missing required field: {k}")
    if scenario.get("language") and scenario["language"] not in LANGUAGES:
        errors.append(f"invalid language: {scenario['language']}")
    if scenario.get("domain") and scenario["domain"] not in DOMAINS:
        errors.append(f"invalid domain: {scenario['domain']}")
    if scenario.get("required_action") and scenario["required_action"] not in REQUIRED_ACTIONS:
        errors.append(f"invalid required_action: {scenario['required_action']}")
    # Reject any leftover multiple_choice fields
    if scenario.get("scenario_type") == "multiple_choice":
        errors.append("multiple_choice scenarios are not supported in v0.1")

    # required_content must be non-empty (scoring needs it)
    rc = scenario.get("required_content")
    if rc is not None and (not isinstance(rc, list) or len(rc) == 0):
        errors.append("required_content must be a non-empty list")

    # source: must have a verifiable URL or DOI (not just free text)
    src = scenario.get("source")
    if not isinstance(src, dict):
        errors.append("source must be an object")
    else:
        url = (src.get("url_or_doi") or "").strip()
        looks_verifiable = (
            url.startswith("http://") or url.startswith("https://")
            or url.lower().startswith("doi:")
            or url.upper().startswith("PMID:")
        )
        if not looks_verifiable:
            errors.append(
                f"source.url_or_doi must be a verifiable URL/DOI/PMID "
                f"(got: {url!r})"
            )

    return errors


def filter_valid(scenarios: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict]]:
    """Split a list of scenarios into (valid, rejected_with_reasons).

    Use this before scoring/eval to discard items that cannot be scored
    reliably (missing URL, missing required_content, etc.).
    """
    valid, rejected = [], []
    for s in scenarios:
        errs = validate(s)
        if errs:
            rejected.append({"id": s.get("id", "?"), "errors": errs})
        else:
            valid.append(s)
    return valid, rejected


def split_dataset(scenarios: list[dict[str, Any]], seed: int = 42,
                  ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
                  ) -> dict[str, list[dict[str, Any]]]:
    """Stratified split (train/dev/test) by (domain, risk_present).

    Deterministic with seed.
    """
    import random
    rng = random.Random(seed)
    # Stratify
    strata: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for s in scenarios:
        key = (s["domain"], bool(s.get("risk_present")))
        strata.setdefault(key, []).append(s)
    train, dev, test = [], [], []
    for key, items in strata.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_dev = int(n * ratios[1])
        train.extend(items[:n_train])
        dev.extend(items[n_train:n_train + n_dev])
        test.extend(items[n_train + n_dev:])
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    # Tag each with its split
    for s in train: s["split"] = "train"
    for s in dev: s["split"] = "dev"
    for s in test: s["split"] = "test"
    return {"train": train, "dev": dev, "test": test}
