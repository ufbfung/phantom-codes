"""Tests for the cost-computation module."""

from __future__ import annotations

from pathlib import Path

import pytest

from phantom_codes.eval.cost import (
    ModelPricing,
    PricingTable,
    compute_call_cost,
    load_pricing,
    resolve_pricing_for_model,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PRICING_YAML = REPO_ROOT / "configs" / "pricing.yaml"


def test_compute_call_cost_per_million_token_pricing() -> None:
    """Cost is the sum of per-component (tokens × $/Mtok) divided by 1e6."""
    pricing = ModelPricing(input=5.00, output=25.00, cache_read=0.50, cache_creation=6.25)
    # 500 input + 150 output + 6000 cached + 0 cache write
    cost = compute_call_cost(
        input_tokens=500,
        output_tokens=150,
        cache_read_tokens=6000,
        cache_creation_tokens=0,
        pricing=pricing,
    )
    expected = (500 * 5.00 + 150 * 25.00 + 6000 * 0.50 + 0 * 6.25) / 1_000_000
    assert cost == pytest.approx(expected)


def test_compute_call_cost_zero_when_all_tokens_zero() -> None:
    pricing = ModelPricing(input=10.0, output=20.0)
    assert compute_call_cost(0, 0, 0, 0, pricing) == 0.0


def test_load_pricing_reads_yaml_into_dataclasses() -> None:
    table = load_pricing(PRICING_YAML)
    assert table.snapshot_date == "2026-05-03"
    # All three providers' headline models should be present.
    assert "claude-opus-4-7" in table.models
    assert "gpt-5.5" in table.models
    assert "gemini-2.5-flash" in table.models
    # Gemini 3 preview models added 2026-05-03 alongside the GA 2.5 tier.
    assert "gemini-3.1-pro-preview" in table.models
    assert "gemini-3-flash-preview" in table.models
    # Spot-check Opus 4.7 numbers from the announcement.
    opus = table.models["claude-opus-4-7"]
    assert opus.input == 5.00
    assert opus.output == 25.00


def test_resolve_pricing_strips_mode_suffix_from_model_name() -> None:
    """Model names in the eval matrix are `{model_id}:{mode}`; strip mode for lookup."""
    table = PricingTable(
        snapshot_date="test",
        models={"claude-opus-4-7": ModelPricing(input=5.0, output=25.0)},
    )
    p = resolve_pricing_for_model("claude-opus-4-7:zeroshot", table)
    assert p is not None
    assert p.input == 5.0
    p = resolve_pricing_for_model("claude-opus-4-7:rag", table)
    assert p is not None
    p = resolve_pricing_for_model("claude-opus-4-7", table)  # already stripped
    assert p is not None


def test_resolve_pricing_returns_none_for_unknown_model() -> None:
    table = PricingTable(snapshot_date="test", models={})
    assert resolve_pricing_for_model("does-not-exist", table) is None
    assert resolve_pricing_for_model("does-not-exist:zeroshot", table) is None
