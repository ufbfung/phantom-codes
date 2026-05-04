#!/usr/bin/env python3
"""Diagnose why Gemini 2.5 Pro returns empty `predictions` arrays at
high rates (88% of zero-shot calls in the n=125 headline run, 2026-05-04).
Tests 5 wrapper configurations to isolate the root cause; tracked in
BACKLOG §P4 "Investigate gemini-2.5-pro empty predictions" — this
script is the operational artifact for that investigation.

Configurations tested:
  A — Baseline             : current settings (max_tokens=1024, schema on, default thinking)
  B — Higher token budget  : max_tokens=4096
  C — Much higher budget   : max_tokens=8192
  D — Disable thinking     : thinking_config=ThinkingConfig(thinking_budget=0)
  E — No schema            : drop response_schema (free-form JSON)

Captures per-call:
  - finish_reason (STOP / MAX_TOKENS / SAFETY / RECITATION / OTHER)
  - usage_metadata: prompt_token_count, candidates_token_count, thoughts_token_count
  - resp.text length
  - whether parseable as {"predictions": [...]}

Cost estimate: ~5 records × 4 modes × 5 configs = 100 calls × ~$0.005 ≈ $0.50-1.

PREREQUISITES:
  - Tier 1 daily quota for gemini-2.5-pro NOT exhausted (1,000 RPD by
    default). If a recent headline run consumed the quota, wait until
    midnight Pacific for it to reset before running this script.
    Without quota headroom, every call returns 429 RESOURCE_EXHAUSTED
    and produces no diagnostic signal.
  - GEMINI_API_KEY (or GOOGLE_API_KEY) set in environment / .env.

USAGE:
  uv run python scripts/diagnose_gemini_empty_predictions.py

OUTPUT:
  - Per-call summary printed to stdout
  - Aggregate config comparison table printed at end
  - Raw per-call results saved to
    results/diagnostics/gemini_empty_prediction_diagnosis.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from dotenv import load_dotenv
    env = Path(".env")
    if env.exists():
        load_dotenv(env)
except ImportError:
    pass

if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
    print("❌ GEMINI_API_KEY (or GOOGLE_API_KEY) not set", file=sys.stderr)
    sys.exit(1)

from google import genai
from google.genai import types

# Reuse the project's actual prompt-building + schema to ensure we're
# diagnosing the EXACT call shape the headline run used.
sys.path.insert(0, str(Path("src")))
from phantom_codes.models.llm import (  # noqa: E402
    PREDICTION_TOOL_SCHEMA,
    PromptMode,
    _strip_for_gemini,
    build_system_prompt,
    build_user_message,
)

MODEL_ID = "gemini-2.5-pro"
COHORT_PATH = Path("benchmarks/synthetic_v1/conditions.ndjson")
N_RECORDS = 5  # 5 unique resource_ids
MODES = ["D1_full", "D2_no_code", "D3_text_only", "D4_abbreviated"]


def _load_sample_records(n: int) -> list[dict]:
    """Pick the first n unique resource_ids worth of records (4 modes each)
    from the cohort. Deterministic — first n unique resource_ids."""
    seen: dict[str, list[dict]] = defaultdict(list)
    with COHORT_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            rid = rec["resource_id"]
            if rid in seen or len(seen) < n:
                seen[rid].append(rec)
            if len(seen) >= n and all(len(v) >= 4 for v in seen.values()):
                break
    out: list[dict] = []
    for rid in list(seen.keys())[:n]:
        out.extend(seen[rid])
    return out


def _make_config(
    *,
    system_prompt: str,
    max_tokens: int,
    schema_on: bool,
    thinking_budget: int | None,
) -> types.GenerateContentConfig:
    """Build a GenerateContentConfig with the variant flags applied."""
    kwargs: dict = {
        "system_instruction": system_prompt,
        "max_output_tokens": max_tokens,
        "response_mime_type": "application/json",
    }
    if schema_on:
        kwargs["response_schema"] = _strip_for_gemini(PREDICTION_TOOL_SCHEMA)
    if thinking_budget is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    return types.GenerateContentConfig(**kwargs)


def _try_parse_predictions(text: str) -> int | None:
    """Return the count of predictions parsed, or None if parse fails."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1:].rstrip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "predictions" in parsed:
            return len(parsed["predictions"]) if isinstance(parsed["predictions"], list) else 0
        return 0
    except (json.JSONDecodeError, ValueError):
        return None  # parse failed


def _summarize_response(resp) -> dict:
    """Pull the diagnostic fields out of a Gemini response."""
    out: dict = {}
    # finish_reason — find on first candidate
    candidates = getattr(resp, "candidates", None) or []
    if candidates:
        cand = candidates[0]
        finish_reason = getattr(cand, "finish_reason", None)
        out["finish_reason"] = str(finish_reason) if finish_reason is not None else "?"
        safety = getattr(cand, "safety_ratings", None)
        out["safety_blocked"] = any(
            getattr(s, "blocked", False) for s in (safety or [])
        )
    else:
        out["finish_reason"] = "(no candidates)"
        out["safety_blocked"] = False

    # usage_metadata — coerce None → 0 since Gemini sometimes returns None
    # for unset numeric fields rather than omitting them.
    meta = getattr(resp, "usage_metadata", None)
    out["prompt_tokens"] = (getattr(meta, "prompt_token_count", 0) or 0) if meta else 0
    out["candidates_tokens"] = (getattr(meta, "candidates_token_count", 0) or 0) if meta else 0
    out["thoughts_tokens"] = (getattr(meta, "thoughts_token_count", 0) or 0) if meta else 0
    out["total_tokens"] = (getattr(meta, "total_token_count", 0) or 0) if meta else 0

    # text + parse outcome
    text = resp.text or ""
    out["text_len"] = len(text)
    out["text_preview"] = text[:120].replace("\n", "\\n") if text else "<empty>"
    out["n_predictions"] = _try_parse_predictions(text) if text else 0

    return out


def main() -> int:
    print(f"Diagnosing {MODEL_ID} empty-prediction issue")
    print(f"Cohort: {COHORT_PATH}  |  Records: {N_RECORDS}  |  Modes: {len(MODES)}")
    print()

    records = _load_sample_records(N_RECORDS)
    print(f"Loaded {len(records)} (record × mode) tuples from cohort")
    print(f"  Unique resource_ids: {len(set(r['resource_id'] for r in records))}")
    print(f"  Modes seen: {sorted(set(r['mode'] for r in records))}")
    print()

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])

    # Build configs
    sys_zeroshot = build_system_prompt(mode=PromptMode.ZEROSHOT, candidates=None)
    configs = [
        ("A_baseline_1024",  dict(max_tokens=1024, schema_on=True,  thinking_budget=None)),
        ("B_4096_tokens",    dict(max_tokens=4096, schema_on=True,  thinking_budget=None)),
        ("C_8192_tokens",    dict(max_tokens=8192, schema_on=True,  thinking_budget=None)),
        ("D_no_thinking",    dict(max_tokens=1024, schema_on=True,  thinking_budget=0)),
        ("E_no_schema",      dict(max_tokens=1024, schema_on=False, thinking_budget=None)),
    ]

    # Per-config aggregate
    config_results: dict[str, list[dict]] = {name: [] for name, _ in configs}

    for ci, (cfg_name, cfg_kwargs) in enumerate(configs, 1):
        print(f"=" * 78)
        print(f"[Config {ci}/{len(configs)}: {cfg_name}]")
        print(f"=" * 78)
        cfg = _make_config(system_prompt=sys_zeroshot, **cfg_kwargs)

        for rec in records:
            user_msg = build_user_message(
                input_fhir=rec.get("input_fhir"),
                input_text=rec.get("input_text"),
            )
            tag = f"  {rec['resource_id'][:12]}.. {rec['mode']:<16}"
            try:
                resp = client.models.generate_content(
                    model=MODEL_ID,
                    contents=user_msg,
                    config=cfg,
                )
                summary = _summarize_response(resp)
            except Exception as e:
                summary = {
                    "error": f"{type(e).__name__}: {str(e)[:100]}",
                    "finish_reason": "(exception)",
                    "n_predictions": None,
                }
            print(
                f"{tag} fr={summary.get('finish_reason', '?'):<25} "
                f"thoughts={summary.get('thoughts_tokens', 0):<5} "
                f"out={summary.get('candidates_tokens', 0):<5} "
                f"text_len={summary.get('text_len', 0):<5} "
                f"n_preds={summary.get('n_predictions')}"
            )
            config_results[cfg_name].append(summary)
        print()

    # ─── Aggregate per-config summary ───────────────────────────────────
    print("=" * 78)
    print("SUMMARY (per config, aggregated across all 20 calls)")
    print("=" * 78)
    print(f"{'Config':<20} {'usable':<8} {'empty':<8} {'parse_fail':<12} {'finish_reasons'}")
    print("-" * 78)
    for cfg_name, results in config_results.items():
        usable = sum(1 for r in results if r.get("n_predictions") and r["n_predictions"] > 0)
        empty = sum(1 for r in results if r.get("n_predictions") == 0)
        parse_fail = sum(1 for r in results if r.get("n_predictions") is None)
        fr_counts = Counter(r.get("finish_reason", "?") for r in results)
        fr_str = ", ".join(f"{k}={v}" for k, v in fr_counts.most_common(3))
        print(f"{cfg_name:<20} {usable:<8} {empty:<8} {parse_fail:<12} {fr_str}")

    # Save raw per-call results for post-hoc analysis
    out_path = Path("results/diagnostics/gemini_empty_prediction_diagnosis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config_results, indent=2, default=str))
    print()
    print(f"Raw per-call results saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
