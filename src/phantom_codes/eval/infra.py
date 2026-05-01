"""Blinded infra-only validation: structural assertions without performance reveal.

Given the long-format DataFrame produced by `run_eval`, compute structural
wiring assertions per model: call counts, token totals, cache reads, latency
percentiles, plus an aggregate "did the classifier code path see all 5
outcome buckets" check.

The intentional omissions are the point. We *do not* report:
- Per-model exact_match / category_match / hallucination rates
- Cross-model rank ordering on any outcome
- Per-model bucket distribution

Those are performance signals that, if seen during scaffolding, can subtly
bias prompt tuning and methodology decisions. Per the project's research
discipline (BACKLOG: Methodology discipline), smoke-test outputs are
infrastructure validation, not findings.

What we *do* report (safe to look at during scaffolding):
- Wiring binary: did each model produce a row per (record, mode) combo
- Token totals per model: cost-validation, not performance
- Cache hit volume per model: validates caching is firing
- Latency p50/p95 per model: validates real-world deployment cost
- Aggregate bucket coverage: did the 5-way classifier exercise all branches
  *anywhere* in the run (not per-model)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from phantom_codes.eval.metrics import Outcome


@dataclass(frozen=True)
class ModelAssertions:
    """Structural assertions for one model — no performance information."""

    model_name: str
    n_calls: int
    tokens_in: int
    tokens_out: int
    cache_read_tokens: int
    cache_creation_tokens: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None


@dataclass(frozen=True)
class InfraAssertions:
    """Aggregate structural assertions for an entire eval run."""

    per_model: list[ModelAssertions]
    all_buckets_reached: bool
    missing_buckets: list[str]


_EXPECTED_BUCKETS = {
    Outcome.EXACT_MATCH.value,
    Outcome.CATEGORY_MATCH.value,
    Outcome.CHAPTER_MATCH.value,
    Outcome.OUT_OF_DOMAIN.value,
    Outcome.HALLUCINATION.value,
}


def infra_assertions(df: pd.DataFrame) -> InfraAssertions:
    """Compute structural wiring assertions from the eval DataFrame.

    Reads the per-prediction CSV schema (the one written by `run_eval`):
    `pred_rank`, `model_name`, `outcome`, `input_tokens`, `output_tokens`,
    `cache_read_tokens`, `cache_creation_tokens`, `latency_ms`.

    Token and latency aggregations operate on rank-0 rows only (the runner
    attributes per-call data to rank-0 to avoid double-counting).
    """
    if df.empty:
        return InfraAssertions(
            per_model=[],
            all_buckets_reached=False,
            missing_buckets=sorted(_EXPECTED_BUCKETS),
        )

    top1 = df[df["pred_rank"] == 0]

    per_model: list[ModelAssertions] = []
    for model_name, grp in top1.groupby("model_name", sort=True):
        latencies = grp["latency_ms"].dropna() if "latency_ms" in grp else pd.Series(dtype=float)
        per_model.append(
            ModelAssertions(
                model_name=str(model_name),
                n_calls=int(len(grp)),
                tokens_in=int(grp.get("input_tokens", pd.Series(dtype=int)).sum()),
                tokens_out=int(grp.get("output_tokens", pd.Series(dtype=int)).sum()),
                cache_read_tokens=int(grp.get("cache_read_tokens", pd.Series(dtype=int)).sum()),
                cache_creation_tokens=int(
                    grp.get("cache_creation_tokens", pd.Series(dtype=int)).sum()
                ),
                latency_p50_ms=float(latencies.quantile(0.5)) if len(latencies) else None,
                latency_p95_ms=float(latencies.quantile(0.95)) if len(latencies) else None,
            )
        )

    seen_buckets = set(df["outcome"].dropna().astype(str).unique())
    missing = sorted(_EXPECTED_BUCKETS - seen_buckets)

    return InfraAssertions(
        per_model=per_model,
        all_buckets_reached=len(missing) == 0,
        missing_buckets=missing,
    )
