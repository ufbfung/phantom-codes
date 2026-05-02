"""Orchestrate (model × record) predictions and outcome classification.

Reads degraded records (from the parquet produced by `prepare`), runs every model on each
record, classifies each prediction with the 5-way outcome taxonomy, and emits a long-format
DataFrame ready for aggregation.

Output schema (one row per top-k prediction slot):

    model_name              str
    resource_id             str
    mode                    str    # D1_full | D2_no_code | D3_text_only | D4_abbreviated
    gt_system               str
    gt_code                 str
    gt_group                str    # ckm | eckm
    pred_rank               int    # 0 = top-1, 1 = top-2, ...
    pred_system             str | None
    pred_code               str | None
    pred_display            str | None
    pred_score              float | None
    outcome                 str    # one of Outcome enum values
    best_top1               str    # best Outcome among top-1 predictions
    best_top5               str    # best Outcome among top-5 predictions
    input_tokens            int    # rank-0 row only; 0 elsewhere (avoid double-counting)
    output_tokens           int
    cache_read_tokens       int
    cache_creation_tokens   int
    latency_ms              float | None  # measured per-call at the runner level
    cost_usd                float | None  # rank-0 row only; None if no pricing entry
    error_type              str | None    # rank-0 row only; e.g., "ServerError" if predict() raised
    error_msg               str | None    # rank-0 row only; capped at 500 chars
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from phantom_codes.eval.cost import (
    PricingTable,
    compute_call_cost,
    resolve_pricing_for_model,
)
from phantom_codes.eval.metrics import (
    IcdValidator,
    Outcome,
    Truth,
    best_outcome_in_topk,
    classify,
)
from phantom_codes.models.base import ConceptNormalizer, Prediction


@dataclass
class EvalRecord:
    """One row from the prepared parquet — a single degraded condition + ground truth."""

    resource_id: str
    mode: str
    input_fhir: dict[str, Any] | None
    input_text: str | None
    gt_system: str
    gt_code: str
    gt_group: str | None

    @property
    def truth(self) -> Truth:
        return Truth(system=self.gt_system, code=self.gt_code)


def load_records(parquet_path: str) -> list[EvalRecord]:
    """Load degraded records from a parquet file written by `prepare`."""
    df = pd.read_parquet(parquet_path)
    out: list[EvalRecord] = []
    for row in df.itertuples(index=False):
        fhir_raw = getattr(row, "input_fhir", None)
        fhir = json.loads(fhir_raw) if isinstance(fhir_raw, str) and fhir_raw else None
        text = getattr(row, "input_text", None)
        out.append(
            EvalRecord(
                resource_id=str(row.resource_id),
                mode=str(row.mode),
                input_fhir=fhir,
                input_text=str(text) if isinstance(text, str) and text else None,
                gt_system=str(row.gt_system),
                gt_code=str(row.gt_code),
                gt_group=getattr(row, "gt_group", None),
            )
        )
    return out


def evaluate_one(
    model: ConceptNormalizer,
    record: EvalRecord,
    validator: IcdValidator,
    top_k: int = 5,
    pricing: PricingTable | None = None,
) -> list[dict[str, Any]]:
    """Run one model on one record; emit one row per top-k prediction slot.

    If the model returns nothing, emit a single row with null predictions and
    HALLUCINATION outcome (the empty-prediction failure mode).

    Fault tolerance: if `model.predict()` raises (e.g., transient API errors
    that exhausted the SDK's internal retries), record the error in the
    rank-0 row's `error_type` / `error_msg` columns and treat predictions as
    empty. This prevents one transient API failure from killing the entire
    eval matrix — the run continues, the failure is visible in the CSV, and
    the user can decide what to do.

    Token counts, latency, and cost are attributed to the rank-0 row only (zero
    on higher-rank rows) to avoid double-counting at aggregation time. Models
    without a `last_usage` attribute (baselines, retrieval) get zero tokens
    but still get a measured latency. `cost_usd` is computed inline using the
    optional `pricing` table; it's None if `pricing` is absent or the model
    isn't in the pricing table.
    """
    started_at = time.perf_counter()
    error_type: str | None = None
    error_msg: str | None = None
    try:
        preds: list[Prediction] = model.predict(
            input_fhir=record.input_fhir,
            input_text=record.input_text,
            top_k=top_k,
        )
    except Exception as e:  # noqa: BLE001 — intentional broad catch at runner boundary
        # Record the failure on the row; downstream sees empty predictions
        # (which the existing path emits as a single HALLUCINATION row).
        error_type = type(e).__name__
        msg = str(e)
        error_msg = msg if len(msg) <= 500 else msg[:497] + "..."
        preds = []
    latency_ms = (time.perf_counter() - started_at) * 1000.0

    # Token usage if the model exposed it (LLMModel, RAGLLMModel). When the
    # call errored, last_usage may be stale from a prior call OR fresh from
    # this one if the error happened post-response — both cases mean "don't
    # bill the user for tokens on a failed call."
    if error_type is not None:
        input_tokens = output_tokens = cache_read_tokens = cache_creation_tokens = 0
    else:
        usage = getattr(model, "last_usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage is not None else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage is not None else 0
        cache_read_tokens = (
            getattr(usage, "cache_read_tokens", 0) if usage is not None else 0
        )
        cache_creation_tokens = (
            getattr(usage, "cache_creation_tokens", 0) if usage is not None else 0
        )

    # Per-call cost — None if pricing table absent or model not priced.
    cost_usd: float | None = None
    if pricing is not None and (
        input_tokens or output_tokens or cache_read_tokens or cache_creation_tokens
    ):
        model_pricing = resolve_pricing_for_model(model.name, pricing)
        if model_pricing is not None:
            cost_usd = compute_call_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_creation_tokens=cache_creation_tokens,
                pricing=model_pricing,
            )

    best_top1 = best_outcome_in_topk(preds, record.truth, validator, k=1)
    best_top5 = best_outcome_in_topk(preds, record.truth, validator, k=5)

    base = {
        "model_name": model.name,
        "resource_id": record.resource_id,
        "mode": record.mode,
        "gt_system": record.gt_system,
        "gt_code": record.gt_code,
        "gt_group": record.gt_group,
        "best_top1": str(best_top1),
        "best_top5": str(best_top5),
    }

    def _row(
        rank: int,
        pred: Prediction | None,
        outcome: Outcome,
    ) -> dict[str, Any]:
        # Token + latency + cost + error metadata only on the rank-0 row to
        # avoid double-counting and duplicate error reporting.
        is_first = rank == 0
        return {
            **base,
            "pred_rank": rank,
            "pred_system": pred.system if pred else None,
            "pred_code": pred.code if pred else None,
            "pred_display": pred.display if pred else None,
            "pred_score": pred.score if pred else None,
            "outcome": str(outcome),
            "input_tokens": input_tokens if is_first else 0,
            "output_tokens": output_tokens if is_first else 0,
            "cache_read_tokens": cache_read_tokens if is_first else 0,
            "cache_creation_tokens": cache_creation_tokens if is_first else 0,
            "latency_ms": latency_ms if is_first else None,
            "cost_usd": cost_usd if is_first else None,
            "error_type": error_type if is_first else None,
            "error_msg": error_msg if is_first else None,
        }

    if not preds:
        return [_row(rank=0, pred=None, outcome=Outcome.HALLUCINATION)]

    return [
        _row(rank=rank, pred=pred, outcome=classify(pred, record.truth, validator))
        for rank, pred in enumerate(preds)
    ]


def run_eval(
    models: Iterable[ConceptNormalizer],
    records: Sequence[EvalRecord],
    validator: IcdValidator,
    top_k: int = 5,
    pricing: PricingTable | None = None,
) -> pd.DataFrame:
    """Run the full (model × record) matrix; return a long-format DataFrame."""
    rows: list[dict[str, Any]] = []
    for model in models:
        for record in records:
            rows.extend(
                evaluate_one(model, record, validator, top_k=top_k, pricing=pricing)
            )
    return pd.DataFrame(rows)


def summarize_by_model_and_mode(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-prediction rows into per-(model, mode) outcome rates.

    Uses the `best_top1` column (one row per record's first prediction) to avoid
    double-counting when each record has top-k rows. The output has one row per
    (model_name, mode) combo with columns for each Outcome rate plus n.
    """
    # Take only the top-1 row per (model, record) to count records, not slots.
    top1 = predictions_df[predictions_df["pred_rank"] == 0].copy()

    summary_rows: list[dict[str, Any]] = []
    for (model, mode), grp in top1.groupby(["model_name", "mode"]):
        n = len(grp)
        rates = {
            outcome.value: float((grp["best_top1"] == outcome.value).sum() / n if n else 0.0)
            for outcome in Outcome
        }
        top5_exact = float(
            (grp["best_top5"] == Outcome.EXACT_MATCH.value).sum() / n if n else 0.0
        )
        summary_rows.append({
            "model_name": model,
            "mode": mode,
            "n": n,
            **{f"top1_{k}": v for k, v in rates.items()},
            "top5_exact_match": top5_exact,
        })
    return pd.DataFrame(summary_rows).sort_values(["model_name", "mode"]).reset_index(drop=True)
