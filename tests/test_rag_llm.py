"""Tests for the RAG-LLM composition.

We test the wiring: does the retriever's top-K get correctly passed to the LLM as
a per-record candidate list, and does the LLM's response flow back as predictions?
Both retriever and LLM client are stubbed so the suite stays fast and offline.
"""

from __future__ import annotations

from typing import Any

from phantom_codes.data.degrade import ICD10_SYSTEM
from phantom_codes.data.disease_groups import CandidateCode
from phantom_codes.models.base import ConceptNormalizer, Prediction
from phantom_codes.models.rag_llm import (
    RAGLLMModel,
    make_rag_anthropic_model,
    make_rag_gemini_model,
    make_rag_openai_model,
)


class _FakeRetriever(ConceptNormalizer):
    """Retriever that returns a predetermined set of predictions."""

    name = "fake-retriever"

    def __init__(self, predictions: list[Prediction]) -> None:
        self._predictions = predictions
        self.last_top_k: int | None = None

    def predict(self, *, input_fhir=None, input_text=None, top_k=5) -> list[Prediction]:
        self.last_top_k = top_k
        return list(self._predictions[:top_k])


class _CapturingLLMClient:
    """LLM client that captures prompts and returns canned predictions."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.received_system: str | None = None
        self.received_user: str | None = None

    def predict_structured(self, system_prompt: str, user_message: str) -> dict[str, Any]:
        self.received_system = system_prompt
        self.received_user = user_message
        return self.response


def _candidates() -> list[CandidateCode]:
    return [
        CandidateCode(code="E11.9", display="Type 2 diabetes mellitus", group="ckm"),
        CandidateCode(code="I10", display="Essential hypertension", group="eckm"),
        CandidateCode(code="N18.31", display="Chronic kidney disease, stage 3a", group="ckm"),
        CandidateCode(code="E78.5", display="Hyperlipidemia, unspecified", group="eckm"),
    ]


def _retrieved(codes: list[str]) -> list[Prediction]:
    """Helper: build retriever predictions for a list of codes."""
    return [
        Prediction(system=ICD10_SYSTEM, code=c, display=f"display-{c}", score=1.0 - i * 0.1)
        for i, c in enumerate(codes)
    ]


def test_rag_llm_passes_retrieved_candidates_into_system_prompt() -> None:
    retriever = _FakeRetriever(_retrieved(["E11.9", "I10"]))
    client = _CapturingLLMClient(
        response={"predictions": [{"code": "E11.9", "display": "T2DM", "confidence": 0.9}]}
    )
    model = RAGLLMModel(
        name="test-rag",
        client=client,
        retriever=retriever,
        candidates=_candidates(),
        retrieve_k=2,
    )
    model.predict(input_text="Patient with T2DM")

    # The two retrieved codes should be in the constrained-mode system prompt
    # (candidate-list format: `- CODE: display [GROUP]`).
    assert "- E11.9: Type 2 diabetes mellitus" in (client.received_system or "")
    assert "- I10: Essential hypertension" in (client.received_system or "")
    # The codes that were NOT retrieved should NOT appear in candidate-list format.
    # (The bare strings might appear elsewhere in the prompt as examples — e.g.,
    # the system instructions show "'E11.9', 'I10', 'N18.31'" as format hints —
    # so we check the candidate-block format specifically.)
    assert "- N18.31:" not in (client.received_system or "")
    assert "- E78.5:" not in (client.received_system or "")
    # MUST-select language confirms it's constrained mode.
    assert "MUST select" in (client.received_system or "")


def test_rag_llm_calls_retriever_with_retrieve_k_not_top_k() -> None:
    """The retriever should see retrieve_k (e.g., 20), not the user's top_k (e.g., 5)."""
    retriever = _FakeRetriever(_retrieved(["E11.9"]))
    client = _CapturingLLMClient(response={"predictions": []})
    model = RAGLLMModel(
        name="test-rag",
        client=client,
        retriever=retriever,
        candidates=_candidates(),
        retrieve_k=20,
    )
    model.predict(input_text="anything", top_k=3)
    assert retriever.last_top_k == 20


def test_rag_llm_returns_at_most_top_k_predictions_from_llm() -> None:
    retriever = _FakeRetriever(_retrieved(["E11.9", "I10", "E78.5"]))
    client = _CapturingLLMClient(
        response={
            "predictions": [
                {"code": "E11.9", "display": "T2DM", "confidence": 0.9},
                {"code": "I10", "display": "HTN", "confidence": 0.5},
                {"code": "E78.5", "display": "HLD", "confidence": 0.2},
            ]
        }
    )
    model = RAGLLMModel(
        name="test-rag",
        client=client,
        retriever=retriever,
        candidates=_candidates(),
        retrieve_k=3,
    )
    preds = model.predict(input_text="anything", top_k=2)
    assert len(preds) == 2
    assert [p.code for p in preds] == ["E11.9", "I10"]


def test_rag_llm_returns_empty_when_retriever_returns_nothing() -> None:
    retriever = _FakeRetriever([])
    client = _CapturingLLMClient(response={"predictions": []})
    model = RAGLLMModel(
        name="test-rag",
        client=client,
        retriever=retriever,
        candidates=_candidates(),
    )
    assert model.predict(input_text="anything") == []
    # LLM should not have been called when nothing was retrieved.
    assert client.received_system is None


def test_rag_llm_drops_retrieved_codes_not_in_candidate_pool() -> None:
    """Defensive: if retriever returns a code we don't have metadata for, skip it."""
    retriever = _FakeRetriever(_retrieved(["E11.9", "Z99.999"]))  # second is bogus
    client = _CapturingLLMClient(
        response={"predictions": [{"code": "E11.9", "display": "T2DM", "confidence": 0.9}]}
    )
    model = RAGLLMModel(
        name="test-rag",
        client=client,
        retriever=retriever,
        candidates=_candidates(),
    )
    model.predict(input_text="anything")
    # Only the in-pool code makes it into the prompt.
    assert "- E11.9: Type 2 diabetes mellitus" in (client.received_system or "")
    assert "- Z99.999:" not in (client.received_system or "")


def test_rag_llm_supports_fhir_input() -> None:
    retriever = _FakeRetriever(_retrieved(["E11.9"]))
    client = _CapturingLLMClient(response={"predictions": []})
    model = RAGLLMModel(
        name="test-rag",
        client=client,
        retriever=retriever,
        candidates=_candidates(),
    )
    fhir = {"resourceType": "Condition", "code": {"text": "diabetes"}}
    model.predict(input_fhir=fhir)
    assert "diabetes" in (client.received_user or "")
    assert "FHIR" in (client.received_user or "")


# ---------- factory smoke tests (no real API calls) ----------


def test_make_rag_anthropic_model_wires_anthropic_client() -> None:
    retriever = _FakeRetriever([])
    model = make_rag_anthropic_model(
        name="claude-haiku-4-5:rag",
        model_id="claude-haiku-4-5",
        retriever=retriever,
        candidates=_candidates(),
        api_key="fake-key",
    )
    assert model.name == "claude-haiku-4-5:rag"
    assert type(model._client).__name__ == "AnthropicClient"


def test_make_rag_openai_model_wires_openai_client() -> None:
    retriever = _FakeRetriever([])
    model = make_rag_openai_model(
        name="gpt-4o-mini:rag",
        model_id="gpt-4o-mini",
        retriever=retriever,
        candidates=_candidates(),
        api_key="fake-key",
    )
    assert type(model._client).__name__ == "OpenAIClient"


def test_make_rag_gemini_model_wires_google_client() -> None:
    retriever = _FakeRetriever([])
    model = make_rag_gemini_model(
        name="gemini-2.5-flash:rag",
        model_id="gemini-2.5-flash",
        retriever=retriever,
        candidates=_candidates(),
        api_key="fake-key",
    )
    assert type(model._client).__name__ == "GoogleClient"
