"""Tests for the model registry — YAML → list[ConceptNormalizer] dispatch.

Uses synthetic configs (not the real configs/models.yaml) so tests
remain isolated from production config changes. Where API keys are
required (LLM providers), tests either monkeypatch them or rely on the
"skip missing API keys" behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phantom_codes.data.disease_groups import CandidateCode
from phantom_codes.eval.registry import (
    ModelRegistryError,
    load_models_from_config,
)


# Minimal candidate set — actual content doesn't matter for registry tests
@pytest.fixture
def candidates() -> list[CandidateCode]:
    return [
        CandidateCode(code="E11.9", display="Type 2 DM", group="ckm"),
        CandidateCode(code="I10", display="Essential HTN", group="eckm"),
    ]


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(content)
    return path


# ─────────────────────────────────────────────────────────────────────────
# Baseline + retrieval providers (no API keys needed)
# ─────────────────────────────────────────────────────────────────────────


def test_baselines_instantiate(tmp_path: Path, candidates: list[CandidateCode]) -> None:
    config = _write_config(
        tmp_path,
        """
test_set:
  - {name: "ex", provider: exact}
  - {name: "fz", provider: fuzzy}
  - {name: "tf", provider: tfidf}
""",
    )
    models = load_models_from_config(config, "test_set", candidates)
    assert len(models) == 3
    names = [m.name for m in models]
    assert names == ["ex", "fz", "tf"]


# ─────────────────────────────────────────────────────────────────────────
# LLM providers — skip-if-missing-key behavior
# ─────────────────────────────────────────────────────────────────────────


def test_llm_skipped_if_api_key_missing(
    tmp_path: Path,
    candidates: list[CandidateCode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
test_set:
  - {name: "ex", provider: exact}
  - {name: "haiku", provider: anthropic, model_id: claude-haiku-4-5, mode: zeroshot}
""",
    )
    warns: list[str] = []
    models = load_models_from_config(
        config, "test_set", candidates, warn=warns.append
    )
    assert len(models) == 1
    assert models[0].name == "ex"
    assert any("ANTHROPIC_API_KEY" in w for w in warns)


def test_llm_raises_when_skip_disabled(
    tmp_path: Path,
    candidates: list[CandidateCode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
test_set:
  - {name: "gpt", provider: openai, model_id: gpt-5.5, mode: zeroshot}
""",
    )
    with pytest.raises(ModelRegistryError, match="OPENAI_API_KEY"):
        load_models_from_config(
            config, "test_set", candidates, skip_missing_api_keys=False
        )


def test_llm_instantiates_with_key_present(
    tmp_path: Path,
    candidates: list[CandidateCode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    config = _write_config(
        tmp_path,
        """
test_set:
  - {name: "haiku", provider: anthropic, model_id: claude-haiku-4-5, mode: zeroshot}
""",
    )
    models = load_models_from_config(config, "test_set", candidates)
    assert len(models) == 1
    assert models[0].name == "haiku"


# ─────────────────────────────────────────────────────────────────────────
# RAG-LLM providers — require retriever
# ─────────────────────────────────────────────────────────────────────────


def test_rag_skipped_if_no_retriever(
    tmp_path: Path,
    candidates: list[CandidateCode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    config = _write_config(
        tmp_path,
        """
test_set:
  - name: "haiku-rag"
    provider: anthropic-rag
    model_id: claude-haiku-4-5
    retrieve_k: 5
""",
    )
    warns: list[str] = []
    models = load_models_from_config(
        config, "test_set", candidates, retriever=None, warn=warns.append
    )
    assert len(models) == 0
    assert any("retriever" in w.lower() for w in warns)


# ─────────────────────────────────────────────────────────────────────────
# Classifier provider — checkpoint glob resolution
# ─────────────────────────────────────────────────────────────────────────


def test_classifier_skipped_if_no_checkpoint(
    tmp_path: Path, candidates: list[CandidateCode]
) -> None:
    config = _write_config(
        tmp_path,
        f"""
test_set:
  - name: "pubmedbert:classifier"
    provider: classifier
    checkpoint_path: "{tmp_path}/no-such-file-*.pt"
""",
    )
    warns: list[str] = []
    models = load_models_from_config(
        config, "test_set", candidates, warn=warns.append
    )
    assert len(models) == 0
    assert any("checkpoint" in w.lower() or "matched" in w.lower() for w in warns)


# ─────────────────────────────────────────────────────────────────────────
# Validation of malformed configs
# ─────────────────────────────────────────────────────────────────────────


def test_unknown_provider_raises(
    tmp_path: Path, candidates: list[CandidateCode]
) -> None:
    config = _write_config(
        tmp_path,
        """
test_set:
  - {name: "bogus", provider: not-a-real-provider}
""",
    )
    with pytest.raises(ModelRegistryError, match="unknown provider"):
        load_models_from_config(config, "test_set", candidates)


def test_missing_provider_raises(
    tmp_path: Path, candidates: list[CandidateCode]
) -> None:
    config = _write_config(
        tmp_path,
        """
test_set:
  - {name: "no-provider"}
""",
    )
    with pytest.raises(ModelRegistryError, match="provider"):
        load_models_from_config(config, "test_set", candidates)


def test_unknown_set_raises(
    tmp_path: Path, candidates: list[CandidateCode]
) -> None:
    config = _write_config(
        tmp_path,
        """
real_set:
  - {name: "x", provider: exact}
""",
    )
    with pytest.raises(ModelRegistryError, match="not found"):
        load_models_from_config(config, "fake_set", candidates)


def test_missing_config_file_raises(
    tmp_path: Path, candidates: list[CandidateCode]
) -> None:
    with pytest.raises(ModelRegistryError, match="not found"):
        load_models_from_config(
            tmp_path / "does-not-exist.yaml", "any_set", candidates
        )


# ─────────────────────────────────────────────────────────────────────────
# Real configs/models.yaml integrity check
# ─────────────────────────────────────────────────────────────────────────


def test_real_models_yaml_smoke_test_set_loads(
    candidates: list[CandidateCode], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The committed configs/models.yaml's smoke_test_set must load
    cleanly (skipping any LLMs without API keys). Catches accidental
    schema breakage in the real config."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    real_config = (
        Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
    )
    if not real_config.exists():
        pytest.skip(f"{real_config} not present (registry config not yet shipped)")

    warns: list[str] = []
    # Don't pass retriever to avoid downloading sentence-transformer in CI
    # — just check non-LLM, non-RAG, non-retrieval entries instantiate.
    models = load_models_from_config(
        real_config, "smoke_test_set", candidates, warn=warns.append
    )
    # At minimum the 3 baselines should always succeed
    baseline_names = {"baseline:exact", "baseline:fuzzy", "baseline:tfidf"}
    actual_names = {m.name for m in models}
    assert baseline_names.issubset(actual_names), (
        f"baselines missing from smoke_test_set load: "
        f"{baseline_names - actual_names}"
    )


def test_real_models_yaml_headline_set_loads(
    candidates: list[CandidateCode], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same integrity check for headline_set — must parse + dispatch
    cleanly even without API keys (those entries get skipped)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    real_config = (
        Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
    )
    if not real_config.exists():
        pytest.skip(f"{real_config} not present")

    warns: list[str] = []
    models = load_models_from_config(
        real_config, "headline_set", candidates, warn=warns.append
    )
    # Without retriever / checkpoint / API keys: only the 3 baselines load.
    # That's enough to confirm config schema is valid.
    baseline_names = {"baseline:exact", "baseline:fuzzy", "baseline:tfidf"}
    actual_names = {m.name for m in models}
    assert baseline_names.issubset(actual_names)
