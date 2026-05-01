"""Tests for the eval runner — orchestration + aggregation."""

from __future__ import annotations

from pathlib import Path

from phantom_codes.config import DataConfig
from phantom_codes.data.degrade import ICD10_SYSTEM
from phantom_codes.data.icd10cm.validator import load as load_validator
from phantom_codes.data.prepare import prepare
from phantom_codes.eval.metrics import Outcome
from phantom_codes.eval.runner import (
    EvalRecord,
    evaluate_one,
    load_records,
    run_eval,
    summarize_by_model_and_mode,
)
from phantom_codes.models.base import ConceptNormalizer, Prediction

FIXTURE = Path(__file__).parent / "fixtures" / "conditions.ndjson"


class _OracleModel(ConceptNormalizer):
    """Returns the ground-truth code when it can read it from input_fhir.

    On D2 (code/system stripped) and D3/D4 (no FHIR at all) the oracle returns nothing,
    which lets us verify the empty-prediction → HALLUCINATION path.
    """

    def __init__(self, name: str = "oracle") -> None:
        self.name = name

    def predict(self, *, input_fhir=None, input_text=None, top_k=5) -> list[Prediction]:
        if input_fhir is None:
            return []
        coding = (input_fhir.get("code") or {}).get("coding") or []
        if not coding:
            return []
        first = coding[0]
        if "system" not in first or "code" not in first:
            return []
        return [
            Prediction(
                system=first["system"],
                code=first["code"],
                display=first.get("display"),
                score=1.0,
            )
        ]


class _FixedCodeModel(ConceptNormalizer):
    """Always returns the same single prediction. Used to verify outcome classification."""

    def __init__(self, name: str, code: str) -> None:
        self.name = name
        self._code = code

    def predict(self, *, input_fhir=None, input_text=None, top_k=5) -> list[Prediction]:
        return [Prediction(system=ICD10_SYSTEM, code=self._code, display=None, score=0.5)]


def _build_records(tmp_path: Path) -> list[EvalRecord]:
    """Helper: prepare the fixture conditions and load the resulting parquet."""
    cfg = DataConfig.model_validate({
        "derived_bucket": "gs://throwaway/phantom-codes",
        "resources": ["MimicCondition"],
        "top_n_codes": 50,
        "seed": 42,
        "splits": {"train": 1.0, "val": 0.0, "test": 0.0},
    })
    out_dir = tmp_path / "splits"
    written = prepare(cfg, FIXTURE, local_out=out_dir)
    return load_records(written["train"])


def test_load_records_returns_one_per_mode_per_in_scope_fixture(tmp_path: Path) -> None:
    records = _build_records(tmp_path)
    # 6 in-scope fixtures × 4 degradation modes = 24
    assert len(records) == 24
    assert all(r.mode in {"D1_full", "D2_no_code", "D3_text_only", "D4_abbreviated"} for r in records)


def test_evaluate_one_oracle_gets_exact_match_when_fhir_present(tmp_path: Path) -> None:
    records = _build_records(tmp_path)
    d1 = next(r for r in records if r.mode == "D1_full" and r.resource_id == "fixture-001")
    rows = evaluate_one(_OracleModel(), d1, load_validator(), top_k=5)
    assert len(rows) == 1
    assert rows[0]["outcome"] == Outcome.EXACT_MATCH.value
    assert rows[0]["best_top1"] == Outcome.EXACT_MATCH.value


def test_evaluate_one_empty_predictions_yield_hallucination_row(tmp_path: Path) -> None:
    records = _build_records(tmp_path)
    d4 = next(r for r in records if r.mode == "D4_abbreviated" and r.resource_id == "fixture-001")
    rows = evaluate_one(_OracleModel(), d4, load_validator(), top_k=5)
    assert len(rows) == 1
    # Oracle returns [] when input_fhir is None (D4 case) → hallucination row.
    assert rows[0]["pred_code"] is None
    assert rows[0]["outcome"] == Outcome.HALLUCINATION.value


def test_evaluate_one_classifies_outcome_taxonomy(tmp_path: Path) -> None:
    records = _build_records(tmp_path)
    # Ground truth for fixture-001 is E11.9.
    rec = next(r for r in records if r.resource_id == "fixture-001" and r.mode == "D2_no_code")
    val = load_validator()

    # Same code → exact
    out = evaluate_one(_FixedCodeModel("m_exact", "E11.9"), rec, val)[0]
    assert out["outcome"] == Outcome.EXACT_MATCH.value

    # Different code, same E11 category → category match
    # (E11.65 is DM2 with hyperglycemia — real billable code, same E11 prefix)
    out = evaluate_one(_FixedCodeModel("m_cat", "E11.65"), rec, val)[0]
    assert out["outcome"] == Outcome.CATEGORY_MATCH.value

    # Different category, same E chapter → chapter match
    out = evaluate_one(_FixedCodeModel("m_chap", "E78.5"), rec, val)[0]
    assert out["outcome"] == Outcome.CHAPTER_MATCH.value

    # Different chapter, real code → out of domain
    out = evaluate_one(_FixedCodeModel("m_ood", "I10"), rec, val)[0]
    assert out["outcome"] == Outcome.OUT_OF_DOMAIN.value

    # Made-up code → hallucination
    out = evaluate_one(_FixedCodeModel("m_hall", "ZZ9.99"), rec, val)[0]
    assert out["outcome"] == Outcome.HALLUCINATION.value


def test_run_eval_produces_long_format_dataframe(tmp_path: Path) -> None:
    records = _build_records(tmp_path)
    models = [_OracleModel(), _FixedCodeModel("fixed_e119", "E11.9")]
    df = run_eval(models, records, load_validator(), top_k=5)

    expected_cols = {
        "model_name", "resource_id", "mode",
        "gt_system", "gt_code", "gt_group",
        "pred_rank", "pred_system", "pred_code", "pred_display", "pred_score",
        "outcome", "best_top1", "best_top5",
    }
    assert expected_cols.issubset(df.columns)
    # Each model evaluated against each record at least once.
    assert df["model_name"].nunique() == 2
    assert df["resource_id"].nunique() == 6


def test_summarize_by_model_and_mode_collapses_topk_rows(tmp_path: Path) -> None:
    records = _build_records(tmp_path)
    df = run_eval([_FixedCodeModel("fixed_e119", "E11.9")], records, load_validator())
    summary = summarize_by_model_and_mode(df)

    # 1 model × 4 degradation modes
    assert len(summary) == 4
    assert {"top1_exact_match", "top1_hallucination", "top5_exact_match", "n"}.issubset(
        summary.columns
    )
    # n is the number of records per mode (6 in-scope fixtures, all modes).
    assert (summary["n"] == 6).all()
    # Fixed E11.9 model gets exact match only on fixture-001 (1 of 6 = ~0.167) per mode.
    diabetes_mode = summary[summary["mode"] == "D1_full"].iloc[0]
    assert abs(diabetes_mode["top1_exact_match"] - 1 / 6) < 1e-9


def test_evaluate_one_top_k_propagates(tmp_path: Path) -> None:
    """If a model returns 3 ranked predictions, we get 3 rows with monotone ranks."""
    records = _build_records(tmp_path)
    rec = next(r for r in records if r.mode == "D2_no_code")

    class _Top3Model(ConceptNormalizer):
        name = "top3"

        def predict(self, *, input_fhir=None, input_text=None, top_k=5):
            return [
                Prediction(system=ICD10_SYSTEM, code="E11.9", display=None, score=0.9),
                Prediction(system=ICD10_SYSTEM, code="E11.0", display=None, score=0.5),
                Prediction(system=ICD10_SYSTEM, code="ZZ9.99", display=None, score=0.1),
            ][:top_k]

    rows = evaluate_one(_Top3Model(), rec, load_validator(), top_k=5)
    assert len(rows) == 3
    assert [r["pred_rank"] for r in rows] == [0, 1, 2]
