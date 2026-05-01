"""Tests for the constrained baselines (exact, fuzzy, tfidf)."""

from __future__ import annotations

import pytest

from phantom_codes.data.disease_groups import CandidateCode
from phantom_codes.models.baselines import (
    ExactMatchBaseline,
    FuzzyMatchBaseline,
    TfidfBaseline,
)


@pytest.fixture
def candidates() -> list[CandidateCode]:
    return [
        CandidateCode(code="E11.9", display="Type 2 diabetes mellitus without complications", group="ckm"),
        CandidateCode(code="I10", display="Essential (primary) hypertension", group="eckm"),
        CandidateCode(code="E78.5", display="Hyperlipidemia, unspecified", group="eckm"),
        CandidateCode(code="E66.9", display="Obesity, unspecified", group="eckm"),
        CandidateCode(code="N18.31", display="Chronic kidney disease, stage 3a", group="ckm"),
    ]


# ---------- ExactMatchBaseline ----------


def test_exact_match_finds_identical_display(candidates) -> None:
    model = ExactMatchBaseline(candidates)
    preds = model.predict(input_text="Type 2 diabetes mellitus without complications")
    assert len(preds) == 1
    assert preds[0].code == "E11.9"
    assert preds[0].score == 1.0


def test_exact_match_is_case_insensitive(candidates) -> None:
    model = ExactMatchBaseline(candidates)
    preds = model.predict(input_text="essential (PRIMARY) HYPERTENSION")
    assert len(preds) == 1
    assert preds[0].code == "I10"


def test_exact_match_returns_nothing_on_partial_match(candidates) -> None:
    model = ExactMatchBaseline(candidates)
    preds = model.predict(input_text="diabetes")
    assert preds == []


def test_exact_match_handles_empty_input(candidates) -> None:
    model = ExactMatchBaseline(candidates)
    assert model.predict(input_text="") == []
    assert model.predict(input_fhir={}) == []


def test_exact_match_uses_fhir_text_field(candidates) -> None:
    model = ExactMatchBaseline(candidates)
    fhir = {"code": {"text": "Obesity, unspecified"}}
    preds = model.predict(input_fhir=fhir)
    assert len(preds) == 1
    assert preds[0].code == "E66.9"


# ---------- FuzzyMatchBaseline ----------


def test_fuzzy_match_handles_close_matches(candidates) -> None:
    model = FuzzyMatchBaseline(candidates)
    # Reordered tokens — token_set_ratio should still rank E11.9 first.
    preds = model.predict(input_text="diabetes mellitus type 2 without complications", top_k=2)
    assert preds[0].code == "E11.9"
    assert preds[0].score > 0.5


def test_fuzzy_match_returns_top_k(candidates) -> None:
    model = FuzzyMatchBaseline(candidates)
    preds = model.predict(input_text="hypertension obesity", top_k=3)
    assert len(preds) == 3
    # Scores monotonically decreasing.
    assert all(preds[i].score >= preds[i + 1].score for i in range(len(preds) - 1))


def test_fuzzy_match_empty_input(candidates) -> None:
    model = FuzzyMatchBaseline(candidates)
    assert model.predict(input_text="") == []


# ---------- TfidfBaseline ----------


def test_tfidf_ranks_obviously_correct_candidate_first(candidates) -> None:
    model = TfidfBaseline(candidates)
    preds = model.predict(input_text="hypertension primary essential", top_k=3)
    assert preds[0].code == "I10"


def test_tfidf_returns_top_k_with_descending_scores(candidates) -> None:
    model = TfidfBaseline(candidates)
    preds = model.predict(input_text="diabetes obesity hypertension", top_k=5)
    assert len(preds) >= 1
    assert all(preds[i].score >= preds[i + 1].score for i in range(len(preds) - 1))


def test_tfidf_unrelated_query_yields_no_high_score(candidates) -> None:
    model = TfidfBaseline(candidates)
    # Token has zero overlap with any candidate display.
    preds = model.predict(input_text="xyzzyqwerty")
    # Could be empty (0.0 sims filtered) — just verify no spurious high score.
    for p in preds:
        assert p.score < 0.5
