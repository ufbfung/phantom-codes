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
    LLMResponse,
    PromptMode,
    _parse_gemini_json,
    build_system_prompt,
    build_user_message,
    make_anthropic_model,
    make_gemini_model,
    make_openai_model,
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


def test_parse_predictions_skips_malformed_string_items() -> None:
    """Real-world Opus failure: a `predictions` array contained a bare string
    instead of `{"code": ..., "display": ..., "confidence": ...}`. Used to
    crash the whole record with AttributeError; should now skip the bad item
    and salvage the well-formed ones."""
    tool_input = {
        "predictions": [
            "E11.9",  # malformed — bare string
            {"code": "I10", "display": "HTN", "confidence": 0.7},
        ]
    }
    preds = parse_predictions(tool_input)
    assert len(preds) == 1
    assert preds[0].code == "I10"


def test_parse_predictions_skips_none_items() -> None:
    tool_input = {
        "predictions": [
            None,
            {"code": "E11.9", "display": "DM2", "confidence": 0.9},
        ]
    }
    preds = parse_predictions(tool_input)
    assert len(preds) == 1
    assert preds[0].code == "E11.9"


def test_parse_predictions_returns_empty_when_predictions_not_a_list() -> None:
    """Defensive: model may return `predictions` as a bare dict / string /
    null instead of the prescribed list shape."""
    assert parse_predictions({"predictions": {"code": "E11.9"}}) == []
    assert parse_predictions({"predictions": "E11.9"}) == []
    assert parse_predictions({"predictions": None}) == []


def test_parse_predictions_tolerates_non_numeric_confidence() -> None:
    """Confidence might come back as a string ('0.9') or null; coerce or fall
    back to 0.0 rather than crash."""
    tool_input = {
        "predictions": [
            {"code": "E11.9", "display": "DM2", "confidence": "not-a-number"},
            {"code": "I10", "display": "HTN", "confidence": None},
        ]
    }
    preds = parse_predictions(tool_input)
    assert {p.code for p in preds} == {"E11.9", "I10"}
    assert all(p.score == 0.0 for p in preds)


# ---------- LLMModel with a fake client ----------


class _FakeClient:
    """Captures inputs and returns a canned response. Verifies the model wiring."""

    def __init__(
        self,
        response: dict[str, Any],
        usage: dict[str, int] | None = None,
    ) -> None:
        self.response = response
        self.usage = usage or {}
        self.received_system: str | None = None
        self.received_user: str | None = None

    def predict_structured(self, system_prompt: str, user_message: str) -> LLMResponse:
        self.received_system = system_prompt
        self.received_user = user_message
        return LLMResponse(
            tool_input=self.response,
            input_tokens=self.usage.get("input_tokens", 0),
            output_tokens=self.usage.get("output_tokens", 0),
            cache_read_tokens=self.usage.get("cache_read_tokens", 0),
            cache_creation_tokens=self.usage.get("cache_creation_tokens", 0),
        )


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


def test_llm_model_last_usage_captures_token_counts() -> None:
    """After predict(), last_usage should hold the LLMResponse from the most recent call."""
    fake = _FakeClient(
        response={"predictions": [{"code": "E11.9", "display": "T2DM", "confidence": 0.9}]},
        usage={
            "input_tokens": 500,
            "output_tokens": 150,
            "cache_read_tokens": 6000,
            "cache_creation_tokens": 0,
        },
    )
    model = LLMModel(name="claude-haiku-4-5:zeroshot", client=fake, mode=PromptMode.ZEROSHOT)
    assert model.last_usage is None
    model.predict(input_text="diabetes")
    assert model.last_usage is not None
    assert model.last_usage.input_tokens == 500
    assert model.last_usage.output_tokens == 150
    assert model.last_usage.cache_read_tokens == 6000
    assert model.last_usage.cache_creation_tokens == 0


def test_llm_model_last_usage_overwrites_on_subsequent_calls() -> None:
    """last_usage reflects the most recent call only, not a cumulative sum."""
    fake = _FakeClient(
        response={"predictions": []},
        usage={"input_tokens": 100, "output_tokens": 50},
    )
    model = LLMModel(name="t", client=fake, mode=PromptMode.ZEROSHOT)
    model.predict(input_text="A")
    assert model.last_usage is not None
    assert model.last_usage.input_tokens == 100
    fake.usage = {"input_tokens": 200, "output_tokens": 75}
    model.predict(input_text="B")
    assert model.last_usage.input_tokens == 200  # not 300


def test_llm_response_default_token_fields_are_zero() -> None:
    """Tests fakes can construct LLMResponse with just tool_input."""
    resp = LLMResponse(tool_input={"predictions": []})
    assert resp.input_tokens == 0
    assert resp.output_tokens == 0
    assert resp.cache_read_tokens == 0
    assert resp.cache_creation_tokens == 0


# ---------- provider-factory smoke tests (no real API calls) ----------


def test_make_anthropic_model_returns_llmmodel_with_anthropic_client() -> None:
    """Constructor wiring: name, mode, and provider client type are correct."""
    model = make_anthropic_model(
        name="claude-haiku-4-5:zeroshot",
        model_id="claude-haiku-4-5",
        mode=PromptMode.ZEROSHOT,
        api_key="fake-anthropic-key",
    )
    assert model.name == "claude-haiku-4-5:zeroshot"
    assert model.mode == PromptMode.ZEROSHOT
    assert type(model._client).__name__ == "AnthropicClient"


def test_make_openai_model_returns_llmmodel_with_openai_client() -> None:
    model = make_openai_model(
        name="gpt-4o-mini:zeroshot",
        model_id="gpt-4o-mini",
        mode=PromptMode.ZEROSHOT,
        api_key="fake-openai-key",
    )
    assert model.name == "gpt-4o-mini:zeroshot"
    assert type(model._client).__name__ == "OpenAIClient"


def test_make_gemini_model_returns_llmmodel_with_google_client() -> None:
    model = make_gemini_model(
        name="gemini-2.5-flash:zeroshot",
        model_id="gemini-2.5-flash",
        mode=PromptMode.ZEROSHOT,
        api_key="fake-gemini-key",
    )
    assert model.name == "gemini-2.5-flash:zeroshot"
    assert type(model._client).__name__ == "GoogleClient"
    # The model_id is what the SDK will route to.
    assert model._client.model_id == "gemini-2.5-flash"


# ---------- _parse_gemini_json defensive parsing ----------


def test_parse_gemini_json_handles_clean_json() -> None:
    parsed, err = _parse_gemini_json('{"predictions": [{"code": "E11.9"}]}')
    assert err is None
    assert parsed == {"predictions": [{"code": "E11.9"}]}


def test_parse_gemini_json_strips_markdown_code_fences() -> None:
    """Gemini sometimes wraps JSON in ```json ... ``` even with application/json mime."""
    text = '```json\n{"predictions": [{"code": "E11.9"}]}\n```'
    parsed, err = _parse_gemini_json(text)
    assert err is None
    assert parsed == {"predictions": [{"code": "E11.9"}]}


def test_parse_gemini_json_strips_unlabeled_code_fences() -> None:
    text = '```\n{"predictions": []}\n```'
    parsed, err = _parse_gemini_json(text)
    assert err is None
    assert parsed == {"predictions": []}


def test_parse_gemini_json_returns_empty_predictions_on_malformed_json() -> None:
    """Malformed responses don't crash the run — they return empty + error."""
    parsed, err = _parse_gemini_json('{"predictions": [{"code":\n')
    assert parsed == {"predictions": []}
    assert err is not None
    assert "JSONDecodeError" in err


def test_parse_gemini_json_rejects_non_dict_top_level() -> None:
    """A bare array or string at the top level is invalid for our schema."""
    parsed, err = _parse_gemini_json('["not a dict"]')
    assert parsed == {"predictions": []}
    assert err is not None
    assert "list" in err
