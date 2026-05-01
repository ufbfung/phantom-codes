"""Tests for the LLM module — pure functions + mock-client integration.

We do NOT make real API calls here. Real-API smoke tests live elsewhere
(out of the unit-test path so CI stays fast and free).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from phantom_codes.data.degrade import ICD10_SYSTEM
from phantom_codes.data.disease_groups import CandidateCode
from phantom_codes.models.llm import (
    LLMModel,
    PromptMode,
    build_system_prompt,
    build_user_message,
    parse_predictions,
)

# ---------- build_system_prompt ----------


def test_zeroshot_prompt_does_not_list_candidates() -> None:
    prompt = build_system_prompt(PromptMode.ZEROSHOT)
    assert "Candidate codes" not in prompt
    assert "MUST select" not in prompt
    assert "ICD-10-CM" in prompt


def test_constrained_prompt_includes_every_candidate_code() -> None:
    candidates = [
        CandidateCode(code="E11.9", display="Type 2 DM uncomplicated", group="ckm"),
        CandidateCode(code="I10", display="Essential hypertension", group="eckm"),
    ]
    prompt = build_system_prompt(PromptMode.CONSTRAINED, candidates)
    assert "MUST select" in prompt
    assert "E11.9: Type 2 DM uncomplicated" in prompt
    assert "I10: Essential hypertension" in prompt
    assert "[CKM]" in prompt
    assert "[ECKM]" in prompt


def test_constrained_prompt_requires_candidates() -> None:
    with pytest.raises(ValueError, match="non-empty candidate list"):
        build_system_prompt(PromptMode.CONSTRAINED, candidates=[])


# ---------- build_user_message ----------


def test_user_message_with_fhir_includes_json() -> None:
    fhir = {"resourceType": "Condition", "code": {"text": "diabetes"}}
    msg = build_user_message(input_fhir=fhir, input_text=None)
    assert "FHIR Condition" in msg
    assert "diabetes" in msg
    # The JSON should be embedded in a fenced block.
    assert "```json" in msg


def test_user_message_with_text_only() -> None:
    msg = build_user_message(input_fhir=None, input_text="Patient has hypertension.")
    assert "Patient has hypertension." in msg
    assert "FHIR" not in msg


def test_user_message_requires_one_of_two_inputs() -> None:
    with pytest.raises(ValueError):
        build_user_message(input_fhir=None, input_text=None)


# ---------- parse_predictions ----------


def test_parse_predictions_sorts_by_confidence_descending() -> None:
    tool_input = {
        "predictions": [
            {"code": "E11.0", "display": "DM2 with hyperosmolarity", "confidence": 0.4},
            {"code": "E11.9", "display": "DM2 uncomplicated", "confidence": 0.95},
            {"code": "E11.65", "display": "DM2 with hyperglycemia", "confidence": 0.6},
        ]
    }
    preds = parse_predictions(tool_input)
    codes = [p.code for p in preds]
    assert codes == ["E11.9", "E11.65", "E11.0"]
    assert all(p.system == ICD10_SYSTEM for p in preds)


def test_parse_predictions_skips_blank_codes() -> None:
    tool_input = {
        "predictions": [
            {"code": "", "display": "blank", "confidence": 0.9},
            {"code": "I10", "display": "HTN", "confidence": 0.8},
        ]
    }
    preds = parse_predictions(tool_input)
    assert len(preds) == 1
    assert preds[0].code == "I10"


def test_parse_predictions_strips_whitespace_from_codes() -> None:
    tool_input = {
        "predictions": [{"code": "  E11.9  ", "display": "x", "confidence": 0.5}]
    }
    preds = parse_predictions(tool_input)
    assert preds[0].code == "E11.9"


def test_parse_predictions_handles_empty_predictions_array() -> None:
    assert parse_predictions({"predictions": []}) == []


# ---------- LLMModel with a fake client ----------


class _FakeClient:
    """Captures inputs and returns a canned response. Verifies the model wiring."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.received_system: str | None = None
        self.received_user: str | None = None

    def predict_structured(self, system_prompt: str, user_message: str) -> dict[str, Any]:
        self.received_system = system_prompt
        self.received_user = user_message
        return self.response


def test_llm_model_predict_returns_top_k_predictions() -> None:
    fake = _FakeClient(
        response={
            "predictions": [
                {"code": "E11.9", "display": "DM2 uncomplicated", "confidence": 0.9},
                {"code": "E11.65", "display": "DM2 hyperglycemia", "confidence": 0.5},
                {"code": "E11.0", "display": "DM2 hyperosmolarity", "confidence": 0.2},
            ]
        }
    )
    model = LLMModel(name="claude-test:zeroshot", client=fake, mode=PromptMode.ZEROSHOT)
    preds = model.predict(input_text="Type 2 diabetes mellitus", top_k=2)
    assert len(preds) == 2
    assert [p.code for p in preds] == ["E11.9", "E11.65"]


def test_llm_model_passes_system_and_user_messages_to_client() -> None:
    fake = _FakeClient(response={"predictions": []})
    candidates = [CandidateCode(code="I10", display="HTN", group="eckm")]
    model = LLMModel(
        name="claude-test:constrained",
        client=fake,
        mode=PromptMode.CONSTRAINED,
        candidates=candidates,
    )
    model.predict(input_text="Hypertension")
    # Constrained mode → candidate list embedded in system prompt.
    assert "I10: HTN" in (fake.received_system or "")
    assert "Hypertension" in (fake.received_user or "")


def test_llm_model_system_prompt_is_built_once_at_construction() -> None:
    """Caching depends on the system prompt being identical across calls."""
    fake = _FakeClient(response={"predictions": []})
    model = LLMModel(name="claude-test", client=fake, mode=PromptMode.ZEROSHOT)
    model.predict(input_text="A")
    sys_first = fake.received_system
    model.predict(input_text="B")
    sys_second = fake.received_system
    assert sys_first == sys_second


def test_llm_model_supports_fhir_input() -> None:
    fake = _FakeClient(response={"predictions": []})
    model = LLMModel(name="t", client=fake, mode=PromptMode.ZEROSHOT)
    fhir = {"resourceType": "Condition", "code": {"text": "obesity"}}
    model.predict(input_fhir=fhir)
    assert "obesity" in (fake.received_user or "")
    assert "FHIR" in (fake.received_user or "")
    # Verify it's actually JSON-embedded (sanity).
    assert json.dumps(fhir, indent=2) in (fake.received_user or "")
