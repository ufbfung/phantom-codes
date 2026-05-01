"""Tests for the bi-encoder retrieval baseline.

Unit tests use a fake encoder so the suite stays fast and offline. The fake encoder
returns deterministic fixed-dimension embeddings so we can verify ranking logic
without downloading sentence-transformers weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from phantom_codes.data.degrade import ICD10_SYSTEM
from phantom_codes.data.disease_groups import CandidateCode
from phantom_codes.models.retrieval import RetrievalModel


class _FakeEncoder:
    """Returns a hardcoded embedding per text. Lets tests pin ranking deterministically."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        out = np.array([self.vectors[t] for t in texts], dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out


def _candidates() -> list[CandidateCode]:
    return [
        CandidateCode(code="E11.9", display="Type 2 diabetes mellitus", group="ckm"),
        CandidateCode(code="I10", display="Essential hypertension", group="eckm"),
        CandidateCode(code="N18.31", display="Chronic kidney disease, stage 3a", group="ckm"),
    ]


def test_retrieval_returns_topk_in_descending_similarity() -> None:
    """Query embedding aligned with E11.9 → E11.9 ranks first.

    Query text matches the E11.9 display verbatim, so they share the same fake-encoder
    embedding (one entry in `vectors`). The other candidates get orthogonal vectors,
    so cosine similarity ranks them strictly below.
    """
    encoder = _FakeEncoder({
        "Type 2 diabetes mellitus": [1.0, 0.0, 0.0],
        "Essential hypertension": [0.0, 1.0, 0.0],
        "Chronic kidney disease, stage 3a": [0.0, 0.0, 1.0],
    })
    model = RetrievalModel(_candidates(), encoder=encoder)
    preds = model.predict(input_text="Type 2 diabetes mellitus", top_k=3)
    assert [p.code for p in preds] == ["E11.9", "I10", "N18.31"]
    assert all(p.system == ICD10_SYSTEM for p in preds)
    # Top-1 score should be 1.0 (exact alignment); others < 1.
    assert preds[0].score == pytest.approx(1.0)
    assert preds[1].score < 1.0


def test_retrieval_topk_truncates_results() -> None:
    encoder = _FakeEncoder({
        "diabetes": [1.0, 0.0, 0.0],
        "Type 2 diabetes mellitus": [0.9, 0.1, 0.0],
        "Essential hypertension": [0.1, 0.9, 0.0],
        "Chronic kidney disease, stage 3a": [0.1, 0.0, 0.9],
    })
    model = RetrievalModel(_candidates(), encoder=encoder)
    preds = model.predict(input_text="diabetes", top_k=2)
    assert len(preds) == 2


def test_retrieval_handles_empty_candidate_list() -> None:
    encoder = _FakeEncoder({})
    model = RetrievalModel([], encoder=encoder)
    assert model.predict(input_text="anything", top_k=5) == []


def test_retrieval_handles_empty_query() -> None:
    encoder = _FakeEncoder({
        "Type 2 diabetes mellitus": [1.0, 0.0, 0.0],
        "Essential hypertension": [0.0, 1.0, 0.0],
        "Chronic kidney disease, stage 3a": [0.0, 0.0, 1.0],
    })
    model = RetrievalModel(_candidates(), encoder=encoder)
    # No input_fhir, no input_text → empty query → empty predictions.
    assert model.predict(top_k=5) == []


def test_retrieval_uses_fhir_text_field_when_no_input_text() -> None:
    encoder = _FakeEncoder({
        "diabetes management": [1.0, 0.0, 0.0],
        "Type 2 diabetes mellitus": [0.95, 0.05, 0.0],
        "Essential hypertension": [0.0, 1.0, 0.0],
        "Chronic kidney disease, stage 3a": [0.0, 0.0, 1.0],
    })
    model = RetrievalModel(_candidates(), encoder=encoder)
    fhir = {"resourceType": "Condition", "code": {"text": "diabetes management"}}
    preds = model.predict(input_fhir=fhir, top_k=1)
    assert preds[0].code == "E11.9"


def test_retrieval_falls_back_to_first_coding_display() -> None:
    encoder = _FakeEncoder({
        "Type 2 diabetes mellitus": [1.0, 0.0, 0.0],
        "Essential hypertension": [0.0, 1.0, 0.0],
        "Chronic kidney disease, stage 3a": [0.0, 0.0, 1.0],
    })
    model = RetrievalModel(_candidates(), encoder=encoder)
    fhir = {
        "resourceType": "Condition",
        "code": {
            "coding": [{"display": "Type 2 diabetes mellitus"}],
        },
    }
    preds = model.predict(input_fhir=fhir, top_k=1)
    assert preds[0].code == "E11.9"
