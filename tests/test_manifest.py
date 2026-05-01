"""Tests for the run manifest sidecar.

Cover the four pieces independently: model serialization, total computation,
manifest assembly, and YAML round-trip. The git-info collection is exercised
opportunistically (this repo is a git repo) but the test is tolerant if it
runs outside one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from phantom_codes.data.disease_groups import CandidateCode
from phantom_codes.eval.cost import ModelPricing, PricingTable, load_pricing
from phantom_codes.eval.manifest import (
    ModelConfig,
    RunManifest,
    RunTotals,
    build_manifest,
    compute_totals,
    get_dependency_versions,
    get_git_info,
    manifest_path_for,
    serialize_models,
    write_manifest,
)
from phantom_codes.models.base import ConceptNormalizer, Prediction
from phantom_codes.models.baselines import ExactMatchBaseline
from phantom_codes.models.llm import LLMModel, LLMResponse, PromptMode
from phantom_codes.models.rag_llm import RAGLLMModel

REPO_ROOT = Path(__file__).resolve().parent.parent
PRICING_YAML = REPO_ROOT / "configs" / "pricing.yaml"


# ---------- fakes ----------


class _FakeAnthropicLikeClient:
    """Looks like AnthropicClient for serialization purposes."""

    def __init__(self, model_id: str = "claude-haiku-4-5") -> None:
        self.model_id = model_id

    def predict_structured(self, system_prompt: str, user_message: str) -> LLMResponse:
        return LLMResponse(tool_input={"predictions": []})


class _FakeOpenAILikeClient:
    def __init__(self, model_id: str = "gpt-4o-mini") -> None:
        self.model_id = model_id

    def predict_structured(self, system_prompt: str, user_message: str) -> LLMResponse:
        return LLMResponse(tool_input={"predictions": []})


class _FakeGoogleLikeClient:
    def __init__(self, model_id: str = "gemini-2.5-flash") -> None:
        self.model_id = model_id

    def predict_structured(self, system_prompt: str, user_message: str) -> LLMResponse:
        return LLMResponse(tool_input={"predictions": []})


# Use type names that match what `_serialize_one_model` looks for.
_FakeAnthropicLikeClient.__name__ = "AnthropicClient"
_FakeOpenAILikeClient.__name__ = "OpenAIClient"
_FakeGoogleLikeClient.__name__ = "GoogleClient"


class _FakeRetriever(ConceptNormalizer):
    name = "fake-retriever"

    def predict(self, *, input_fhir=None, input_text=None, top_k=5) -> list[Prediction]:
        return []


def _candidates() -> list[CandidateCode]:
    return [
        CandidateCode(code="E11.9", display="Type 2 diabetes mellitus", group="ckm"),
        CandidateCode(code="I10", display="Essential hypertension", group="eckm"),
    ]


# ---------- serialize_models ----------


def test_serialize_models_distinguishes_anthropic_openai_google() -> None:
    models = [
        LLMModel(
            name="claude-haiku-4-5:zeroshot",
            client=_FakeAnthropicLikeClient(),
            mode=PromptMode.ZEROSHOT,
        ),
        LLMModel(
            name="gpt-4o-mini:zeroshot",
            client=_FakeOpenAILikeClient(),
            mode=PromptMode.ZEROSHOT,
        ),
        LLMModel(
            name="gemini-2.5-flash:zeroshot",
            client=_FakeGoogleLikeClient(),
            mode=PromptMode.ZEROSHOT,
        ),
    ]
    configs = serialize_models(models)
    assert {c.provider for c in configs} == {"anthropic", "openai", "google"}
    assert all(c.mode == "PromptMode.ZEROSHOT" or c.mode.endswith("zeroshot") for c in configs)


def test_serialize_models_records_constrained_mode() -> None:
    candidates = _candidates()
    models = [
        LLMModel(
            name="claude-haiku-4-5:constrained",
            client=_FakeAnthropicLikeClient(),
            mode=PromptMode.CONSTRAINED,
            candidates=candidates,
        ),
    ]
    config = serialize_models(models)[0]
    assert config.mode.endswith("constrained")
    assert config.provider == "anthropic"
    assert config.model_id == "claude-haiku-4-5"


def test_serialize_models_records_rag_mode_with_retrieve_k() -> None:
    candidates = _candidates()
    models = [
        RAGLLMModel(
            name="claude-haiku-4-5:rag",
            client=_FakeAnthropicLikeClient(),
            retriever=_FakeRetriever(),
            candidates=candidates,
            retrieve_k=15,
        ),
    ]
    config = serialize_models(models)[0]
    assert config.mode == "rag"
    assert config.retrieve_k == 15
    assert config.provider == "anthropic"


def test_serialize_models_records_baselines_as_baseline_provider() -> None:
    candidates = _candidates()
    models = [ExactMatchBaseline(candidates), _FakeRetriever()]
    configs = serialize_models(models)
    assert all(c.provider == "baseline" for c in configs)
    assert all(c.mode == "baseline" for c in configs)
    assert all(c.model_id is None for c in configs)


# ---------- compute_totals ----------


def _row(
    model: str,
    rank: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    outcome: str = "exact_match",
) -> dict[str, Any]:
    return {
        "model_name": model,
        "pred_rank": rank,
        "outcome": outcome,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
    }


def test_compute_totals_sums_only_rank0_rows() -> None:
    df = pd.DataFrame([
        _row("haiku", rank=0, input_tokens=500, output_tokens=150),
        _row("haiku", rank=1, input_tokens=99999, output_tokens=99999),  # higher rank, ignored
        _row("opus", rank=0, input_tokens=600, output_tokens=200),
    ])
    totals = compute_totals(df, pricing=None)
    assert totals.input_tokens == 1100
    assert totals.output_tokens == 350
    assert totals.total_cost_usd is None  # no pricing


def test_compute_totals_with_pricing_table_returns_dollar_total() -> None:
    df = pd.DataFrame([
        _row("haiku:zeroshot", input_tokens=1_000_000, output_tokens=200_000),
    ])
    pricing = PricingTable(
        snapshot_date="test",
        models={"haiku": ModelPricing(input=1.00, output=5.00)},
    )
    totals = compute_totals(df, pricing=pricing)
    # 1M input × $1/M + 200k output × $5/M = $1 + $1 = $2
    assert totals.total_cost_usd == pytest.approx(2.0)
    assert totals.n_models_with_pricing == 1
    assert totals.n_models_without_pricing == 0


def test_compute_totals_partial_pricing_coverage() -> None:
    """If pricing is missing for some models, count them but still report cost for the rest."""
    df = pd.DataFrame([
        _row("haiku:zeroshot", input_tokens=1_000_000, output_tokens=0),
        _row("unknown-model", input_tokens=1_000_000, output_tokens=0),
    ])
    pricing = PricingTable(
        snapshot_date="test",
        models={"haiku": ModelPricing(input=1.00, output=5.00)},
    )
    totals = compute_totals(df, pricing=pricing)
    assert totals.n_models_with_pricing == 1
    assert totals.n_models_without_pricing == 1
    # Only haiku contributed to cost.
    assert totals.total_cost_usd == pytest.approx(1.0)


def test_compute_totals_empty_dataframe() -> None:
    df = pd.DataFrame()
    totals = compute_totals(df, pricing=None)
    assert totals.input_tokens == 0
    assert totals.total_cost_usd is None


# ---------- get_git_info ----------


def test_get_git_info_returns_sha_and_dirty_flag_in_git_repo() -> None:
    """This repo is a git repo; confirm we can read its state. Tolerant of CI envs without git."""
    sha, dirty = get_git_info(cwd=REPO_ROOT)
    if sha is None and dirty is None:
        pytest.skip("Not running in a git environment")
    assert isinstance(sha, str) and len(sha) >= 7
    assert isinstance(dirty, bool)


def test_get_git_info_returns_none_outside_git_repo(tmp_path: Path) -> None:
    sha, dirty = get_git_info(cwd=tmp_path)
    assert sha is None
    assert dirty is None


# ---------- get_dependency_versions ----------


def test_get_dependency_versions_resolves_known_packages() -> None:
    versions = get_dependency_versions(packages=["pandas", "pyyaml"])
    assert versions["pandas"] is not None
    assert versions["pyyaml"] is not None


def test_get_dependency_versions_missing_packages_become_none() -> None:
    versions = get_dependency_versions(packages=["this-package-definitely-does-not-exist-xyz123"])
    assert versions["this-package-definitely-does-not-exist-xyz123"] is None


# ---------- build_manifest + write_manifest round-trip ----------


def test_build_manifest_assembles_run_metadata(tmp_path: Path) -> None:
    candidates = _candidates()
    models: list[ConceptNormalizer] = [
        ExactMatchBaseline(candidates),
        LLMModel(
            name="claude-haiku-4-5:zeroshot",
            client=_FakeAnthropicLikeClient(),
            mode=PromptMode.ZEROSHOT,
        ),
    ]
    df = pd.DataFrame([
        _row("exact", input_tokens=0, output_tokens=0, outcome="exact_match"),
        _row("claude-haiku-4-5:zeroshot", input_tokens=500, output_tokens=150, outcome="hallucination"),
    ])
    pricing = load_pricing(PRICING_YAML)
    started = datetime(2026, 5, 1, 22, 52, 55, tzinfo=UTC)
    finished = datetime(2026, 5, 1, 22, 53, 5, tzinfo=UTC)

    manifest = build_manifest(
        run_id="20260501T225255Z",
        command_name="smoke-test",
        started_at=started,
        finished_at=finished,
        seed=42,
        fixtures_path="tests/fixtures/conditions.ndjson",
        n_records=24,
        n_candidates=len(candidates),
        models=models,
        df=df,
        pricing_table=pricing,
        csv_path=tmp_path / "smoke_test_20260501T225255Z.csv",
        infra_only=True,
    )
    assert manifest.run_id == "20260501T225255Z"
    assert manifest.command_name == "smoke-test"
    assert manifest.duration_seconds == 10.0
    assert manifest.seed == 42
    assert manifest.n_records == 24
    assert manifest.n_candidates == 2
    assert manifest.infra_only is True
    assert len(manifest.models) == 2
    assert manifest.pricing_snapshot_date == pricing.snapshot_date
    # We saw 2 outcome buckets out of 5; missing 3.
    assert manifest.all_buckets_reached is False
    assert len(manifest.missing_buckets) == 3


def test_write_manifest_yaml_roundtrips(tmp_path: Path) -> None:
    """Write a manifest, read it back, verify key fields survive the round-trip."""
    started = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    finished = datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC)
    manifest = RunManifest(
        run_id="test",
        command_name="smoke-test",
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        duration_seconds=5.0,
        git_sha="abc1234",
        git_dirty=False,
        phantom_codes_version="0.1.0",
        python_version="3.11.10",
        key_dependencies={"pandas": "2.2.0", "missing": None},
        seed=42,
        fixtures_path="tests/fixtures/conditions.ndjson",
        n_records=10,
        n_candidates=5,
        models=[
            ModelConfig(
                name="claude:zeroshot",
                provider="anthropic",
                model_id="claude",
                mode="zeroshot",
            ),
        ],
        pricing_snapshot_date="2026-05-01",
        csv_path="results/raw/test.csv",
        infra_only=True,
        totals=RunTotals(
            input_tokens=500,
            output_tokens=150,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            total_cost_usd=0.0125,
            n_models_with_pricing=1,
            n_models_without_pricing=0,
        ),
        all_buckets_reached=True,
        missing_buckets=[],
    )
    path = tmp_path / "test.manifest.yaml"
    write_manifest(manifest, path)

    with open(path) as f:
        loaded = yaml.safe_load(f)
    assert loaded["run_id"] == "test"
    assert loaded["git_sha"] == "abc1234"
    assert loaded["git_dirty"] is False
    assert loaded["seed"] == 42
    assert loaded["models"][0]["name"] == "claude:zeroshot"
    assert loaded["totals"]["input_tokens"] == 500
    assert loaded["totals"]["total_cost_usd"] == pytest.approx(0.0125)
    assert loaded["all_buckets_reached"] is True


def test_manifest_path_for_returns_sidecar_with_manifest_yaml_suffix() -> None:
    csv = Path("results/raw/smoke_test_20260501T225255Z.csv")
    assert manifest_path_for(csv).name == "smoke_test_20260501T225255Z.manifest.yaml"


def test_manifest_yaml_human_readable_sample(tmp_path: Path) -> None:
    """Sanity check: the YAML output is human-readable, not flow-style on a single line."""
    started = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    manifest = RunManifest(
        run_id="r",
        command_name="smoke-test",
        started_at=started.isoformat(),
        finished_at=started.isoformat(),
        duration_seconds=0.0,
        git_sha=None,
        git_dirty=None,
        phantom_codes_version="0.1.0",
        python_version="3.11.10",
        key_dependencies={"pandas": "2.2.0"},
        seed=42,
        fixtures_path=None,
        n_records=0,
        n_candidates=0,
        models=[],
        pricing_snapshot_date=None,
        csv_path="x.csv",
        infra_only=False,
        totals=RunTotals(),
        all_buckets_reached=False,
        missing_buckets=[],
    )
    path = tmp_path / "x.manifest.yaml"
    write_manifest(manifest, path)
    text = path.read_text()
    # Block style: each top-level key on its own line.
    assert "run_id: r\n" in text
    assert "totals:\n" in text  # nested dict rendered as block, not flow
    # Should NOT be all-on-one-line flow style.
    assert "{run_id" not in text
