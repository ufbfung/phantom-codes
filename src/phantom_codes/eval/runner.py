"""Orchestrate (model × record) predictions and outcome classification.

Reads degraded records (from the parquet produced by `prepare`), runs every model on each
record, classifies each prediction with the 5-way outcome taxonomy, and emits a long-format
DataFrame ready for aggregation.

Output schema (one row per top-k prediction slot):

    model_name    str
    resource_id   str
    mode          str            # D1_full | D2_no_code | D3_text_only | D4_abbreviated
    gt_system     str
    gt_code       str
    gt_group      str            # ckm | eckm
    pred_rank     int            # 0 = top-1, 1 = top-2, ...
    pred_system   str | None
    pred_code     str | None
    pred_display  str | None
    pred_score    float | None
    outcome       str            # one of Outcome enum values
    best_top1     str            # best Outcome among top-1 predictions
    best_top5     str            # best Outcome among top-5 predictions
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

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
) -> list[dict[str, Any]]:
    """Run one model on one record; emit one row per top-k prediction slot.

    If the model returns nothing, emit a single row with null predictions and
    HALLUCINATION outcome (the empty-prediction failure mode).
    """
    preds: list[Prediction] = model.predict(
        input_fhir=record.input_fhir,
        input_text=record.input_text,
        top_k=top_k,
    )

    best_top1 = best_outcome_in_topk(preds, record.truth, validator, k=1)
    best_top5 = best_outcome_in_topk(preds, record.truth, validator, k=5)

    rows: list[dict[str, Any]] = []
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
    if not preds:
        rows.append({
            **base,
            "pred_rank": 0,
            "pred_system": None,
            "pred_code": None,
            "pred_display": None,
            "pred_score": None,
            "outcome": str(Outcome.HALLUCINATION),
        })
        return rows

    for rank, pred in enumerate(preds):
        outcome = classify(pred, record.truth, validator)
        rows.append({
            **base,
            "pred_rank": rank,
            "pred_system": pred.system,
            "pred_code": pred.code,
            "pred_display": pred.display,
            "pred_score": pred.score,
            "outcome": str(outcome),
        })
    return rows


def run_eval(
    models: Iterable[ConceptNormalizer],
    records: Sequence[EvalRecord],
    validator: IcdValidator,
    top_k: int = 5,
) -> pd.DataFrame:
    """Run the full (model × record) matrix; return a long-format DataFrame."""
    rows: list[dict[str, Any]] = []
    for model in models:
        for record in records:
            rows.extend(evaluate_one(model, record, validator, top_k=top_k))
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
