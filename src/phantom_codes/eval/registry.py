"""Model registry — YAML → list[ConceptNormalizer] dispatch.

Lifts the hardcoded model-instantiation logic out of `cli.py:smoke-test`
into a config-driven loader. Both `phantom-codes smoke-test` and
`phantom-codes evaluate` consume `configs/models.yaml` through this
module so the model lineup stays declarative, version-controlled, and
identical across smoke-tests and headline runs.

Provider dispatch lives in `_PROVIDER_DISPATCH` and reuses the
existing model factories. Adding a new provider type means adding one
key to that table and one branch in the call.

API-key requirements:
    LLM providers (anthropic / openai / google + RAG variants) need
    the corresponding API key in the environment. If absent, the
    model is silently skipped with a warning rather than raising —
    so contributors without all three providers' keys can still run
    partial evaluations.

Trained classifier:
    The `classifier` provider's `checkpoint_path` field accepts glob
    patterns (e.g., `models/checkpoints/pubmedbert/best_*.pt`) and
    resolves to the most recently modified matching file.
"""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from collections.abc import Callable
from typing import Any

import yaml

from phantom_codes.data.disease_groups import CandidateCode
from phantom_codes.models.base import ConceptNormalizer
from phantom_codes.models.baselines import (
    ExactMatchBaseline,
    FuzzyMatchBaseline,
    TfidfBaseline,
)
from phantom_codes.models.llm import (
    PromptMode,
    make_anthropic_model,
    make_gemini_model,
    make_openai_model,
)
from phantom_codes.models.rag_llm import (
    make_rag_anthropic_model,
    make_rag_gemini_model,
    make_rag_openai_model,
)
from phantom_codes.models.retrieval import RetrievalModel


class ModelRegistryError(Exception):
    """Raised for malformed configs (missing required fields, unknown
    provider, etc.). Configuration mistakes should be loud."""


def load_models_from_config(
    config_path: str | Path,
    set_name: str,
    candidates: list[CandidateCode],
    retriever: ConceptNormalizer | None = None,
    *,
    skip_missing_api_keys: bool = True,
    warn: Callable[[str], None] | None = None,
) -> list[ConceptNormalizer]:
    """Load and instantiate every model in the named set.

    Args:
        config_path: path to `configs/models.yaml` (or compatible).
        set_name: which named set to instantiate (e.g.,
            "smoke_test_set", "headline_set").
        candidates: ACCESS-scope candidate codes the constrained
            baselines + LLMs use as their menu.
        retriever: pre-built retrieval model (sentence-transformer)
            shared with RAG-LLM variants. If any RAG entry exists in
            the set but `retriever` is None, the entry is skipped
            with a warning. Building the retriever once and passing
            it in avoids repeatedly downloading the encoder weights.
        skip_missing_api_keys: when True (default), LLM models whose
            provider's API key isn't in the environment are skipped
            with a warning rather than raising.
        warn: optional callback for warning messages. Defaults to
            print-to-stderr-style behavior via a built-in helper.

    Returns:
        list of instantiated ConceptNormalizer, ready to feed into
        `runner.run_eval()`. Order matches the config file, with
        skipped entries omitted.
    """
    if warn is None:
        warn = _default_warn

    config = _load_config(config_path)
    if set_name not in config:
        raise ModelRegistryError(
            f"set {set_name!r} not found in {config_path}; "
            f"available: {sorted(config.keys())}"
        )

    entries = config[set_name]
    if not isinstance(entries, list):
        raise ModelRegistryError(
            f"set {set_name!r} must be a list of entries; got {type(entries).__name__}"
        )

    models: list[ConceptNormalizer] = []
    for i, entry in enumerate(entries):
        try:
            model = _build_model(entry, candidates, retriever, skip_missing_api_keys, warn)
        except _SkipEntryError as skip:
            warn(f"skipping {entry.get('name', f'entry-{i}')!r}: {skip}")
            continue
        if model is not None:
            models.append(model)

    return models


def _load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise ModelRegistryError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ModelRegistryError(
            f"config root must be a mapping of set-name → list-of-entries; "
            f"got {type(raw).__name__}"
        )
    return raw


def _build_model(
    entry: dict[str, Any],
    candidates: list[CandidateCode],
    retriever: ConceptNormalizer | None,
    skip_missing_api_keys: bool,
    warn: Callable[[str], None],
) -> ConceptNormalizer | None:
    """Dispatch a single config entry to the appropriate factory."""
    provider = entry.get("provider")
    if provider is None:
        raise ModelRegistryError(f"entry missing required 'provider' field: {entry}")

    name = entry.get("name") or f"{provider}:auto"

    if provider in ("exact", "fuzzy", "tfidf"):
        model = _BASELINE_FACTORIES[provider](candidates)
        # Baselines have a class-level `name` attr (e.g., "exact"); allow
        # the config to override it for use cases like distinguishing
        # multiple baseline configurations in the same eval run.
        if "name" in entry:
            model.name = entry["name"]
        return model

    if provider == "retrieval":
        model = RetrievalModel(candidates)
        if "name" in entry:
            model.name = entry["name"]
        return model

    if provider == "classifier":
        return _build_classifier(entry, name)

    if provider in ("anthropic", "openai", "google"):
        return _build_llm(entry, candidates, skip_missing_api_keys, warn)

    if provider in ("anthropic-rag", "openai-rag", "google-rag"):
        if retriever is None:
            raise _SkipEntryError(
                "RAG provider requires a pre-built retriever; "
                "rebuild the call site to pass `retriever=...`"
            )
        return _build_rag_llm(entry, candidates, retriever, skip_missing_api_keys, warn)

    raise ModelRegistryError(
        f"unknown provider {provider!r} for entry {name!r}; "
        f"valid providers: {sorted(_VALID_PROVIDERS)}"
    )


def _build_classifier(entry: dict[str, Any], name: str) -> ConceptNormalizer:
    """Resolve checkpoint glob → most-recent file, then instantiate."""
    # Lazy import to keep torch out of the registry's import path
    # for users running CPU-only smoke tests with no classifier.
    from phantom_codes.models.classifier import ClassifierModel

    glob_pattern = entry.get("checkpoint_path")
    if not glob_pattern:
        raise ModelRegistryError(
            f"classifier entry {name!r} missing 'checkpoint_path' field"
        )
    matches = sorted(glob(glob_pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        raise _SkipEntryError(
            f"no checkpoint matched {glob_pattern!r}; "
            "train one via `phantom-codes train` first"
        )
    return ClassifierModel(checkpoint_path=matches[0], name=name)


def _build_llm(
    entry: dict[str, Any],
    candidates: list[CandidateCode],
    skip_missing_api_keys: bool,
    warn: Callable[[str], None],
) -> ConceptNormalizer:
    provider = entry["provider"]
    model_id = entry.get("model_id")
    mode_str = entry.get("mode")
    name = entry.get("name") or f"{provider}:{model_id}:{mode_str}"

    if not model_id:
        raise ModelRegistryError(f"LLM entry {name!r} missing 'model_id'")
    if not mode_str:
        raise ModelRegistryError(f"LLM entry {name!r} missing 'mode'")

    api_key_env = _API_KEY_ENV[provider]
    if not os.environ.get(api_key_env):
        if skip_missing_api_keys:
            raise _SkipEntryError(f"{api_key_env} not set in environment")
        raise ModelRegistryError(f"{api_key_env} not set in environment")

    try:
        mode = PromptMode(mode_str)
    except ValueError as e:
        raise ModelRegistryError(
            f"entry {name!r} has invalid mode {mode_str!r}: {e}"
        ) from e

    factory = _LLM_FACTORIES[provider]
    # Constrained mode receives candidates; zero-shot doesn't need them.
    candidate_arg = candidates if mode == PromptMode.CONSTRAINED else None
    return factory(name=name, model_id=model_id, mode=mode, candidates=candidate_arg)


def _build_rag_llm(
    entry: dict[str, Any],
    candidates: list[CandidateCode],
    retriever: ConceptNormalizer,
    skip_missing_api_keys: bool,
    warn: Callable[[str], None],
) -> ConceptNormalizer:
    provider = entry["provider"]
    model_id = entry.get("model_id")
    retrieve_k = entry.get("retrieve_k", 20)
    name = entry.get("name") or f"{provider}:{model_id}"

    if not model_id:
        raise ModelRegistryError(f"RAG-LLM entry {name!r} missing 'model_id'")

    base_provider = provider.removesuffix("-rag")  # anthropic-rag → anthropic
    api_key_env = _API_KEY_ENV[base_provider]
    if not os.environ.get(api_key_env):
        if skip_missing_api_keys:
            raise _SkipEntryError(f"{api_key_env} not set in environment")
        raise ModelRegistryError(f"{api_key_env} not set in environment")

    factory = _RAG_LLM_FACTORIES[provider]
    return factory(
        name=name,
        model_id=model_id,
        retriever=retriever,
        candidates=candidates,
        retrieve_k=retrieve_k,
    )


# ─────────────────────────────────────────────────────────────────────────
# Provider dispatch tables
# ─────────────────────────────────────────────────────────────────────────

_BASELINE_FACTORIES: dict[str, Callable[[list[CandidateCode]], ConceptNormalizer]] = {
    "exact": ExactMatchBaseline,
    "fuzzy": FuzzyMatchBaseline,
    "tfidf": TfidfBaseline,
}

_LLM_FACTORIES: dict[str, Callable[..., ConceptNormalizer]] = {
    "anthropic": make_anthropic_model,
    "openai": make_openai_model,
    "google": make_gemini_model,
}

_RAG_LLM_FACTORIES: dict[str, Callable[..., ConceptNormalizer]] = {
    "anthropic-rag": make_rag_anthropic_model,
    "openai-rag": make_rag_openai_model,
    "google-rag": make_rag_gemini_model,
}

_API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
}

_VALID_PROVIDERS = (
    set(_BASELINE_FACTORIES)
    | {"retrieval", "classifier"}
    | set(_LLM_FACTORIES)
    | set(_RAG_LLM_FACTORIES)
)


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────


class _SkipEntryError(Exception):
    """Raised internally to signal a config entry should be skipped
    (rather than fail the whole load) — e.g., missing API key, missing
    checkpoint, missing retriever."""


def _default_warn(message: str) -> None:
    """Default warning callback — prints to stderr with a [registry]
    prefix. Tests can override via the `warn` parameter."""
    import sys

    print(f"[registry] WARN: {message}", file=sys.stderr)
