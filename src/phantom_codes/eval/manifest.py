"""Run manifest sidecar — provenance metadata written next to each per-prediction CSV.

Goal: future-you (or a paper reviewer) can take any per-prediction CSV and
reconstruct the exact state of the world that produced it. The CSV stores
*labels* (which model name produced which prediction); the manifest stores
the *configuration* that gave those labels their behavior at run time.

Naming convention (sidecar):
    results/raw/smoke_test_20260501T225255Z.csv          ← per-prediction rows
    results/raw/smoke_test_20260501T225255Z.manifest.yaml ← run-level provenance

Why a separate file (not row-level metadata in the CSV):
- Some metadata is structured (full candidate list, dependency versions);
  awkward as repeated columns
- Run-level data shouldn't be repeated on every row (bloat)
- Sidecar is easy to find via path convention; easy to scan in YAML

Volume: one manifest per run — handful of dev smoke tests during scaffolding,
one or two real-MIMIC runs, occasional re-runs. Each manifest is a few KB.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import phantom_codes
from phantom_codes.eval.cost import (
    PricingTable,
    compute_call_cost,
    resolve_pricing_for_model,
)
from phantom_codes.eval.metrics import Outcome
from phantom_codes.models.base import ConceptNormalizer

# Dependencies whose version meaningfully affects model outputs. Pinned in
# the manifest so a future replication can install the same versions.
_KEY_DEPENDENCIES = [
    "anthropic",
    "openai",
    "google-genai",
    "sentence-transformers",
    "transformers",
    "torch",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "rapidfuzz",
]


@dataclass(frozen=True)
class ModelConfig:
    """Configuration of one model in the eval matrix."""

    name: str  # e.g., "claude-haiku-4-5:zeroshot" — the eval-matrix label
    provider: str  # "anthropic" | "openai" | "google" | "baseline"
    model_id: str | None  # e.g., "claude-haiku-4-5" — None for baselines
    mode: str  # "zeroshot" | "constrained" | "rag" | "baseline"
    retrieve_k: int | None = None  # only set for RAG models


@dataclass(frozen=True)
class RunTotals:
    """Aggregate token usage and (optional) cost for the run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    # None if no model in the run had a pricing entry; partial otherwise.
    total_cost_usd: float | None = None
    n_models_with_pricing: int = 0
    n_models_without_pricing: int = 0


@dataclass(frozen=True)
class RunManifest:
    """Run-level provenance metadata. One per CSV."""

    # Identity
    run_id: str
    command_name: str
    started_at: str  # ISO 8601
    finished_at: str
    duration_seconds: float

    # Code provenance
    git_sha: str | None
    git_dirty: bool | None  # None when not in a git repo
    phantom_codes_version: str
    python_version: str
    key_dependencies: dict[str, str | None]

    # Configuration of what was run
    seed: int | None
    fixtures_path: str | None
    n_records: int
    n_candidates: int
    models: list[ModelConfig]

    # Pricing snapshot used for cost computation
    pricing_snapshot_date: str | None  # None if pricing.yaml wasn't loaded

    # Output references
    csv_path: str
    infra_only: bool

    # Aggregates (run-level, not per-model performance)
    totals: RunTotals
    all_buckets_reached: bool
    missing_buckets: list[str]


def get_git_info(cwd: Path | None = None) -> tuple[str | None, bool | None]:
    """Return (sha, dirty). Both None when the directory isn't a git repo
    or git is unavailable.

    `dirty` reflects working-tree changes (`git status --porcelain` non-empty).
    A dirty run flag is important: if you re-ran with uncommitted changes,
    the git_sha alone won't reproduce the code that ran.
    """
    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if sha_result.returncode != 0:
            return None, None
        sha = sha_result.stdout.strip() or None
        if sha is None:
            return None, None

        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        dirty = bool(dirty_result.stdout.strip())
        return sha, dirty
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, None


def get_dependency_versions(packages: list[str] | None = None) -> dict[str, str | None]:
    """Look up installed versions for `packages`. Missing packages → None.

    Defaults to the curated list of packages whose version affects outputs.
    """
    pkgs = packages if packages is not None else _KEY_DEPENDENCIES
    out: dict[str, str | None] = {}
    for pkg in pkgs:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def serialize_models(models: list[ConceptNormalizer]) -> list[ModelConfig]:
    """Inspect each model and produce a ModelConfig record.

    Recognizes LLMModel and RAGLLMModel via duck-typed attributes (`_client`,
    `_mode`, `_retrieve_k`). Anything else is recorded as a generic baseline.
    """
    return [_serialize_one_model(m) for m in models]


def _serialize_one_model(model: ConceptNormalizer) -> ModelConfig:
    name = model.name
    client = getattr(model, "_client", None)

    if client is None:
        # Baseline (exact, fuzzy, tfidf) or retrieval — no provider client.
        return ModelConfig(
            name=name,
            provider="baseline",
            model_id=None,
            mode="baseline",
            retrieve_k=None,
        )

    provider = {
        "AnthropicClient": "anthropic",
        "OpenAIClient": "openai",
        "GoogleClient": "google",
    }.get(type(client).__name__, "unknown")

    model_id = getattr(client, "model_id", None)
    retrieve_k = getattr(model, "_retrieve_k", None)

    # LLMModel exposes ._mode (PromptMode); RAGLLMModel doesn't have one — it
    # always operates in a constrained-with-per-record-candidates mode, which
    # we record as "rag".
    mode_attr = getattr(model, "_mode", None)
    if mode_attr is not None:
        mode = str(mode_attr)
    elif retrieve_k is not None:
        mode = "rag"
    else:
        mode = "unknown"

    return ModelConfig(
        name=name,
        provider=provider,
        model_id=model_id,
        mode=mode,
        retrieve_k=retrieve_k,
    )


def compute_totals(df: pd.DataFrame, pricing: PricingTable | None) -> RunTotals:
    """Aggregate token counts and (optional) cost from the per-prediction CSV.

    Operates on rank-0 rows only (the runner attributes per-call usage to
    rank 0 to avoid double-counting).

    Cost resolution order:
    1. If the CSV has a `cost_usd` column (modern runs), sum it directly —
       cost was already computed by the runner using a snapshot pricing
       table, so re-passing pricing here would double-derive it.
    2. Otherwise, if `pricing` is provided, recompute per-model cost from
       token totals × pricing — used for backward compat with older CSVs
       that predate the cost column.
    3. Otherwise, `total_cost_usd = None`.
    """
    if df.empty:
        return RunTotals()

    top1 = df[df["pred_rank"] == 0]

    def _sum(col: str) -> int:
        if col not in top1.columns:
            return 0
        return int(top1[col].fillna(0).sum())

    total_in = _sum("input_tokens")
    total_out = _sum("output_tokens")
    total_cr = _sum("cache_read_tokens")
    total_cw = _sum("cache_creation_tokens")

    # Path 1: per-row cost is already in the CSV.
    if "cost_usd" in top1.columns:
        cost_series = top1["cost_usd"].dropna()
        total_cost = float(cost_series.sum()) if len(cost_series) else None
        # Count distinct models that had at least one priced row vs. unpriced.
        priced_models = top1[top1["cost_usd"].notna()]["model_name"].unique()
        unpriced_models = top1[top1["cost_usd"].isna()]["model_name"].unique()
        n_with_pricing = len(set(priced_models))
        n_without = len(set(unpriced_models) - set(priced_models))
        return RunTotals(
            input_tokens=total_in,
            output_tokens=total_out,
            cache_read_tokens=total_cr,
            cache_creation_tokens=total_cw,
            total_cost_usd=total_cost,
            n_models_with_pricing=n_with_pricing,
            n_models_without_pricing=n_without,
        )

    # Path 2 (backward compat): older CSVs without cost_usd — recompute from
    # tokens × pricing if a pricing table is provided.
    if pricing is None:
        return RunTotals(
            input_tokens=total_in,
            output_tokens=total_out,
            cache_read_tokens=total_cr,
            cache_creation_tokens=total_cw,
            total_cost_usd=None,
        )

    total_cost = 0.0
    n_with_pricing = 0
    n_without = 0
    for model_name, grp in top1.groupby("model_name"):
        m_pricing = resolve_pricing_for_model(str(model_name), pricing)
        if m_pricing is None:
            n_without += 1
            continue
        n_with_pricing += 1
        total_cost += compute_call_cost(
            input_tokens=int(grp["input_tokens"].fillna(0).sum()),
            output_tokens=int(grp["output_tokens"].fillna(0).sum()),
            cache_read_tokens=int(grp["cache_read_tokens"].fillna(0).sum()),
            cache_creation_tokens=int(grp["cache_creation_tokens"].fillna(0).sum()),
            pricing=m_pricing,
        )

    return RunTotals(
        input_tokens=total_in,
        output_tokens=total_out,
        cache_read_tokens=total_cr,
        cache_creation_tokens=total_cw,
        total_cost_usd=total_cost if n_with_pricing > 0 else None,
        n_models_with_pricing=n_with_pricing,
        n_models_without_pricing=n_without,
    )


def build_manifest(
    *,
    run_id: str,
    command_name: str,
    started_at: datetime,
    finished_at: datetime,
    seed: int | None,
    fixtures_path: str | None,
    n_records: int,
    n_candidates: int,
    models: list[ConceptNormalizer],
    df: pd.DataFrame,
    pricing_table: PricingTable | None,
    csv_path: Path,
    infra_only: bool,
) -> RunManifest:
    """Assemble a RunManifest from runtime state. Pure function — testable."""
    git_sha, git_dirty = get_git_info()

    expected_buckets = {o.value for o in Outcome}
    if df.empty or "outcome" not in df.columns:
        seen: set[str] = set()
    else:
        seen = set(df["outcome"].dropna().astype(str).unique())
    missing = sorted(expected_buckets - seen)

    return RunManifest(
        run_id=run_id,
        command_name=command_name,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        duration_seconds=(finished_at - started_at).total_seconds(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        phantom_codes_version=phantom_codes.__version__,
        python_version=platform.python_version(),
        key_dependencies=get_dependency_versions(),
        seed=seed,
        fixtures_path=fixtures_path,
        n_records=n_records,
        n_candidates=n_candidates,
        models=serialize_models(models),
        pricing_snapshot_date=pricing_table.snapshot_date if pricing_table else None,
        csv_path=str(csv_path),
        infra_only=infra_only,
        totals=compute_totals(df, pricing_table),
        all_buckets_reached=len(missing) == 0,
        missing_buckets=missing,
    )


def write_manifest(manifest: RunManifest, path: Path) -> None:
    """Serialize a RunManifest to YAML at `path`. Creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _to_yaml_dict(manifest)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, indent=2)


def _to_yaml_dict(manifest: RunManifest) -> dict[str, Any]:
    """Convert nested dataclasses to a plain dict for yaml.safe_dump.

    `dataclasses.asdict` handles nested dataclasses recursively, but yaml
    needs primitive types — keep this as a single seam in case we add
    non-serializable fields later.
    """
    return asdict(manifest)


def manifest_path_for(csv_path: Path) -> Path:
    """Convention: sidecar at `<csv_stem>.manifest.yaml`."""
    return csv_path.with_suffix(".manifest.yaml")
