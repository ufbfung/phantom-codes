"""Cost computation from token-usage data + a pricing config.

Intentionally lightweight: token usage is captured at the model layer
(`LLMResponse.input_tokens` / `output_tokens` / `cache_read_tokens` /
`cache_creation_tokens`) and persisted in the per-prediction CSV. Cost is
computed at *aggregation* time from those columns plus a pricing snapshot
loaded from `configs/pricing.yaml`. Splitting it this way means pricing
updates (or alternative pricing scenarios) don't require re-running the
evaluation.

Per-row cost contract: a model-prediction row gets its full cost attributed
to it. For top-k expansion (multiple prediction rows per API call), only
the rank-0 row should carry cost — the runner enforces this so we don't
double-count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens for one model."""

    input: float
    output: float
    cache_read: float = 0.0
    cache_creation: float = 0.0
    notes: str = ""


@dataclass(frozen=True)
class PricingTable:
    """Loaded snapshot of model pricing."""

    snapshot_date: str
    models: dict[str, ModelPricing]

    def lookup(self, model_id: str) -> ModelPricing | None:
        """Look up pricing by model ID. Returns None if not in the table."""
        return self.models.get(model_id)


def load_pricing(path: str | Path) -> PricingTable:
    """Load `configs/pricing.yaml` into a PricingTable."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    snapshot_date = str(raw.get("snapshot_date", "unknown"))
    models = {
        name: ModelPricing(
            input=float(entry.get("input", 0.0)),
            output=float(entry.get("output", 0.0)),
            cache_read=float(entry.get("cache_read", 0.0)),
            cache_creation=float(entry.get("cache_creation", 0.0)),
            notes=str(entry.get("notes", "")),
        )
        for name, entry in (raw.get("models") or {}).items()
    }
    return PricingTable(snapshot_date=snapshot_date, models=models)


def compute_call_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    pricing: ModelPricing,
) -> float:
    """Return the USD cost for one API call given its token counts and pricing."""
    return (
        input_tokens * pricing.input
        + output_tokens * pricing.output
        + cache_read_tokens * pricing.cache_read
        + cache_creation_tokens * pricing.cache_creation
    ) / 1_000_000


def resolve_pricing_for_model(
    model_name: str,
    pricing_table: PricingTable,
) -> ModelPricing | None:
    """Look up pricing for a model name from the eval matrix.

    Model names in the eval matrix follow the pattern `{model_id}:{mode}`
    (e.g., "claude-opus-4-7:zeroshot"). Pricing is keyed by `model_id` only
    (e.g., "claude-opus-4-7"). This helper strips the `:mode` suffix so
    aggregation code can pass the raw `model_name` from the CSV.
    """
    base_id = model_name.split(":", 1)[0]
    return pricing_table.lookup(base_id)
