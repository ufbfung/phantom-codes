"""Tests for the blinded infra-only assertions helper.

We verify two things: (1) the structural data extraction is correct given a
known DataFrame, and (2) the output deliberately omits per-model performance
signals (accuracy, bucket distribution).
"""

from __future__ import annotations

import pandas as pd
import pytest

from phantom_codes.eval.infra import (
    InfraAssertions,
    ModelAssertions,
    infra_assertions,
)
from phantom_codes.eval.metrics import Outcome


def _row(
    model: str,
    rank: int = 0,
    outcome: str = "exact_match",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    latency_ms: float | None = None,
) -> dict:
    return {
        "model_name": model,
        "pred_rank": rank,
        "outcome": outcome,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "latency_ms": latency_ms,
    }


def test_infra_assertions_aggregates_tokens_per_model() -> None:
    df = pd.DataFrame([
        _row("haiku", input_tokens=500, output_tokens=150, latency_ms=800.0),
        _row("haiku", input_tokens=500, output_tokens=150, latency_ms=900.0),
        _row("opus", input_tokens=600, output_tokens=200, latency_ms=4000.0),
    ])
    a = infra_assertions(df)
    by_name = {m.model_name: m for m in a.per_model}
    assert by_name["haiku"].n_calls == 2
    assert by_name["haiku"].tokens_in == 1000
    assert by_name["haiku"].tokens_out == 300
    assert by_name["opus"].n_calls == 1
    assert by_name["opus"].tokens_in == 600


def test_infra_assertions_attributes_only_rank0_rows() -> None:
    """Higher-rank rows should not contribute to token totals (avoid double-counting)."""
    df = pd.DataFrame([
        _row("haiku", rank=0, input_tokens=500, output_tokens=150),
        _row("haiku", rank=1, input_tokens=0, output_tokens=0),  # higher rank, zero tokens
        _row("haiku", rank=2, input_tokens=0, output_tokens=0),
    ])
    a = infra_assertions(df)
    haiku = next(m for m in a.per_model if m.model_name == "haiku")
    assert haiku.n_calls == 1  # one rank-0 row
    assert haiku.tokens_in == 500


def test_infra_assertions_computes_latency_percentiles() -> None:
    df = pd.DataFrame([
        _row("haiku", latency_ms=100.0),
        _row("haiku", latency_ms=200.0),
        _row("haiku", latency_ms=300.0),
        _row("haiku", latency_ms=400.0),
        _row("haiku", latency_ms=500.0),
    ])
    a = infra_assertions(df)
    haiku = next(m for m in a.per_model if m.model_name == "haiku")
    assert haiku.latency_p50_ms == 300.0
    # pandas quantile linear-interpolation: p95 of [100..500] = 480
    assert haiku.latency_p95_ms == 480.0


def test_infra_assertions_handles_models_without_latency() -> None:
    """Models that didn't record latency (e.g., baselines if runner couldn't time them)."""
    df = pd.DataFrame([
        _row("baseline", latency_ms=None),
        _row("baseline", latency_ms=None),
    ])
    a = infra_assertions(df)
    baseline = next(m for m in a.per_model if m.model_name == "baseline")
    assert baseline.latency_p50_ms is None
    assert baseline.latency_p95_ms is None


def test_infra_assertions_sums_cost_usd_when_column_present() -> None:
    """When the per-prediction CSV includes cost_usd (modern runs), sum it per model."""
    df = pd.DataFrame([
        {**_row("haiku"), "cost_usd": 0.001},
        {**_row("haiku"), "cost_usd": 0.002},
        {**_row("opus"), "cost_usd": 0.05},
    ])
    a = infra_assertions(df)
    by_name = {m.model_name: m for m in a.per_model}
    assert by_name["haiku"].cost_usd == pytest.approx(0.003)
    assert by_name["opus"].cost_usd == pytest.approx(0.05)


def test_infra_assertions_cost_is_none_when_column_missing() -> None:
    """Older CSVs (pre-cost-tracking) just lack the column — cost_usd should be None."""
    df = pd.DataFrame([_row("baseline")])  # _row helper doesn't include cost_usd
    a = infra_assertions(df)
    assert a.per_model[0].cost_usd is None


def test_infra_assertions_cost_is_none_when_all_rows_unpriced() -> None:
    """Model with cost_usd column but all NaN values (unpriced model) reports None."""
    import math

    df = pd.DataFrame([
        {**_row("unpriced-model"), "cost_usd": math.nan},
        {**_row("unpriced-model"), "cost_usd": math.nan},
    ])
    a = infra_assertions(df)
    assert a.per_model[0].cost_usd is None


def test_infra_assertions_counts_errors_per_model() -> None:
    """When error_type column has values for some rows, count and surface them."""
    df = pd.DataFrame([
        {**_row("haiku"), "error_type": None},
        {**_row("haiku"), "error_type": None},
        {**_row("opus"), "error_type": "ServerError"},
        {**_row("opus"), "error_type": "ServerError"},
        {**_row("opus"), "error_type": "TimeoutException"},
    ])
    a = infra_assertions(df)
    by_name = {m.model_name: m for m in a.per_model}
    assert by_name["haiku"].n_errors == 0
    assert by_name["haiku"].dominant_error_type is None
    assert by_name["opus"].n_errors == 3
    # ServerError appears 2× vs TimeoutException 1×, so it dominates.
    assert by_name["opus"].dominant_error_type == "ServerError"


def test_infra_assertions_error_columns_default_when_csv_lacks_them() -> None:
    """Older CSVs (pre-fault-tolerance) have no error_type column → 0 errors, None dominant."""
    df = pd.DataFrame([_row("baseline")])  # _row helper doesn't include error_type
    a = infra_assertions(df)
    assert a.per_model[0].n_errors == 0
    assert a.per_model[0].dominant_error_type is None


def test_infra_assertions_reports_all_buckets_reached_when_5_seen() -> None:
    df = pd.DataFrame([
        _row("m", outcome=Outcome.EXACT_MATCH.value),
        _row("m", outcome=Outcome.CATEGORY_MATCH.value),
        _row("m", outcome=Outcome.CHAPTER_MATCH.value),
        _row("m", outcome=Outcome.OUT_OF_DOMAIN.value),
        _row("m", outcome=Outcome.HALLUCINATION.value),
    ])
    a = infra_assertions(df)
    assert a.all_buckets_reached is True
    assert a.missing_buckets == []


def test_infra_assertions_reports_missing_buckets() -> None:
    df = pd.DataFrame([
        _row("m", outcome=Outcome.EXACT_MATCH.value),
        _row("m", outcome=Outcome.HALLUCINATION.value),
    ])
    a = infra_assertions(df)
    assert a.all_buckets_reached is False
    assert set(a.missing_buckets) == {
        Outcome.CATEGORY_MATCH.value,
        Outcome.CHAPTER_MATCH.value,
        Outcome.OUT_OF_DOMAIN.value,
    }


def test_infra_assertions_handles_empty_dataframe() -> None:
    df = pd.DataFrame(columns=[
        "model_name", "pred_rank", "outcome",
        "input_tokens", "output_tokens", "cache_read_tokens",
        "cache_creation_tokens", "latency_ms",
    ])
    a = infra_assertions(df)
    assert a.per_model == []
    assert a.all_buckets_reached is False
    assert len(a.missing_buckets) == 5


def test_infra_assertions_does_not_expose_per_model_outcome_distribution() -> None:
    """Crucial blinding property: per-model rows must not include outcome counts
    or accuracy. Performance signals are forbidden in infra-only output.
    """
    # The ModelAssertions dataclass exposes these fields and *no others*.
    # Cost / tokens / latency / error counts are infrastructure metrics
    # (deployment economics + reliability), not performance metrics
    # (accuracy / hallucination rate / bucket distribution).
    fields = set(ModelAssertions.__dataclass_fields__)
    assert fields == {
        "model_name",
        "n_calls",
        "tokens_in",
        "tokens_out",
        "cache_read_tokens",
        "cache_creation_tokens",
        "cost_usd",
        "latency_p50_ms",
        "latency_p95_ms",
        "n_errors",
        "dominant_error_type",
    }
    # No exact_match, hallucination, accuracy, or bucket-count fields.
    forbidden = {"exact_match", "hallucination", "accuracy", "bucket_counts"}
    assert fields & forbidden == set()


def test_infra_assertions_is_alphabetized_by_model_name_for_stability() -> None:
    """Ordering of per_model is alphabetical so output is stable across runs."""
    df = pd.DataFrame([
        _row("zebra"),
        _row("alpha"),
        _row("mike"),
    ])
    a = infra_assertions(df)
    names = [m.model_name for m in a.per_model]
    assert names == ["alpha", "mike", "zebra"]


def test_infra_assertions_returns_dataclass_instances() -> None:
    """Type-check the public surface."""
    df = pd.DataFrame([_row("m")])
    a = infra_assertions(df)
    assert isinstance(a, InfraAssertions)
    assert all(isinstance(m, ModelAssertions) for m in a.per_model)
