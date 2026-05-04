"""Tests for the 5-way outcome taxonomy and aggregate metrics."""

from __future__ import annotations

from phantom_codes.data.degrade import ICD10_SYSTEM
from phantom_codes.eval.metrics import (
    Outcome,
    Truth,
    best_outcome_in_topk,
    category,
    chapter,
    classify,
    summarize,
)
from phantom_codes.models.base import Prediction


class _StubValidator:
    """In-memory ICD-10-CM existence check for tests."""

    def __init__(self, real_codes: set[str]) -> None:
        self.real_codes = real_codes

    def exists(self, system: str, code: str) -> bool:
        return system == ICD10_SYSTEM and code in self.real_codes


def _pred(code: str, system: str = ICD10_SYSTEM, score: float = 1.0) -> Prediction:
    return Prediction(system=system, code=code, display=None, score=score)


def _truth(code: str, system: str = ICD10_SYSTEM) -> Truth:
    return Truth(system=system, code=code)


# A reasonably realistic universe of "real" ICD-10-CM codes for tests.
REAL = _StubValidator(
    {
        "E11.9", "E11.65", "E11.0", "E11.1",
        "E78.5", "E78.0",
        "E66.9",
        "I10", "I11.9", "I20.0", "I25.10",
        "N18.31", "N18.32",
        "J18.9",
    }
)


def test_category_extracts_3_char_prefix() -> None:
    assert category("E11.9") == "E11"
    assert category("E11") == "E11"
    assert category("I10") == "I10"
    assert category("N18.31") == "N18"


def test_chapter_extracts_first_letter() -> None:
    assert chapter("E11.9") == "E"
    assert chapter("I10") == "I"
    assert chapter("") == ""


def test_classify_exact_match() -> None:
    pred = _pred("E11.9")
    truth = _truth("E11.9")
    assert classify(pred, truth, REAL) == Outcome.EXACT_MATCH


def test_classify_category_match_same_3char_prefix() -> None:
    """E11.0 vs truth E11.9 — both Type 2 DM."""
    pred = _pred("E11.0")
    truth = _truth("E11.9")
    assert classify(pred, truth, REAL) == Outcome.CATEGORY_MATCH


def test_classify_chapter_match_different_category() -> None:
    """E78.5 (lipidemia) vs truth E11.9 (DM2) — both endocrine 'E' chapter."""
    pred = _pred("E78.5")
    truth = _truth("E11.9")
    assert classify(pred, truth, REAL) == Outcome.CHAPTER_MATCH


def test_classify_out_of_domain_real_code_unrelated_chapter() -> None:
    """J18.9 (pneumonia, J chapter) vs truth E11.9 (DM2, E chapter) — real code, unrelated."""
    pred = _pred("J18.9")
    truth = _truth("E11.9")
    assert classify(pred, truth, REAL) == Outcome.OUT_OF_DOMAIN


def test_classify_hallucination_nonexistent_code() -> None:
    """E11.99 doesn't exist in our validator's real-codes set."""
    pred = _pred("E11.99")
    truth = _truth("E11.9")
    assert classify(pred, truth, REAL) == Outcome.HALLUCINATION


def test_classify_hallucination_wrong_system() -> None:
    """Predicting in SNOMED instead of ICD-10-CM is a hallucination of the wrong domain."""
    pred = _pred("44054006", system="http://snomed.info/sct")
    truth = _truth("E11.9")
    assert classify(pred, truth, REAL) == Outcome.HALLUCINATION


def test_classify_exact_takes_priority_over_other_buckets() -> None:
    """Sanity: an exact match is never miscategorized as category-match etc."""
    pred = _pred("E11.9")
    truth = _truth("E11.9")
    assert classify(pred, truth, REAL) == Outcome.EXACT_MATCH


def test_best_outcome_in_topk_picks_best_among_first_k() -> None:
    truth = _truth("E11.9")
    preds = [
        _pred("E78.5", score=0.9),  # chapter match
        _pred("E11.0", score=0.8),  # category match
        _pred("E11.9", score=0.7),  # exact match — at rank 3
    ]
    # Top-1: only chapter match.
    assert best_outcome_in_topk(preds, truth, REAL, k=1) == Outcome.CHAPTER_MATCH
    # Top-2: chapter + category — best is category.
    assert best_outcome_in_topk(preds, truth, REAL, k=2) == Outcome.CATEGORY_MATCH
    # Top-3: includes exact — best is exact.
    assert best_outcome_in_topk(preds, truth, REAL, k=3) == Outcome.EXACT_MATCH


def test_best_outcome_in_topk_empty_predictions_is_no_prediction() -> None:
    """Model returning nothing usable maps to NO_PREDICTION (abstention),
    distinct from HALLUCINATION (fabrication of a non-existent code).
    Refined 2026-05-04 from the original 5-bucket taxonomy that lumped
    these together — see metrics.py module docstring for the safety
    rationale (abstention is preferable to fabrication)."""
    truth = _truth("E11.9")
    assert best_outcome_in_topk([], truth, REAL, k=5) == Outcome.NO_PREDICTION


def test_summarize_counts_and_rates() -> None:
    outcomes = [
        Outcome.EXACT_MATCH,
        Outcome.EXACT_MATCH,
        Outcome.CATEGORY_MATCH,
        Outcome.HALLUCINATION,
    ]
    summary = summarize(outcomes)
    assert summary.n == 4
    assert summary.counts[Outcome.EXACT_MATCH] == 2
    assert summary.exact_match_rate == 0.5
    assert summary.hallucination_rate == 0.25
    # All 5 outcomes appear in to_dict, including zeros.
    d = summary.to_dict()
    assert d["chapter_match"] == 0.0
    assert d["out_of_domain"] == 0.0


def test_summarize_empty_input_no_div_by_zero() -> None:
    summary = summarize([])
    assert summary.n == 0
    assert summary.exact_match_rate == 0.0
    assert summary.hallucination_rate == 0.0
