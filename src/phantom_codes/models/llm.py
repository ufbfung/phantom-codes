"""LLM-based code recovery: Claude (Anthropic) and GPT (OpenAI), with two prompting modes.

Two modes per LLM:

- `zeroshot`     — primary experiment. The LLM predicts ICD-10-CM codes from the degraded
                   input with no candidate list. Tests raw out-of-the-box ability and
                   exposes hallucination.
- `constrained`  — control experiment. Same input, same LLM, but the prompt contains an
                   explicit candidate list. Comparing zero-shot vs constrained isolates
                   whether failures are due to hallucination (wandering off the menu)
                   or genuine ignorance.

Prompt caching:
- Anthropic: explicit `cache_control: ephemeral` on the system prompt block.
- OpenAI: automatic for prompts >= 1024 tokens; we just keep the system prompt static.

Structured output:
- Both providers are pinned to a JSON object with a `predictions` array of ranked
  candidates. We use Anthropic tool-use (forced tool choice) and OpenAI JSON mode.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from phantom_codes.data.degrade import ICD10_SYSTEM
from phantom_codes.data.disease_groups import CandidateCode
from phantom_codes.models.base import ConceptNormalizer, Prediction


class PromptMode(StrEnum):
    ZEROSHOT = "zeroshot"
    CONSTRAINED = "constrained"


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


# Tool / JSON schema for structured output.
PREDICTION_TOOL_NAME = "record_predictions"
PREDICTION_TOOL_DESCRIPTION = (
    "Record the model's ranked ICD-10-CM code predictions for the given clinical input."
)
PREDICTION_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "predictions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "ICD-10-CM code in dotted format, e.g. 'E11.9' or 'I10'.",
                    },
                    "display": {
                        "type": "string",
                        "description": "Human-readable description of the code.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Model's confidence, higher = more confident.",
                    },
                },
                "required": ["code", "display", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["predictions"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Pure functions: prompt construction and response parsing. Unit-testable.
# ---------------------------------------------------------------------------


def build_system_prompt(
    mode: PromptMode,
    candidates: list[CandidateCode] | None = None,
    top_k: int = 5,
) -> str:
    """Static portion of the prompt — same across every prediction in a run.

    This is the chunk we cache. The LLM sees this once and reuses it cheaply.
    """
    base = (
        "You are a clinical informatics assistant. Your task is to recover the "
        "standardized ICD-10-CM diagnosis code for a clinical condition described "
        "by the user.\n\n"
        f"Return up to {top_k} candidate codes, ranked by your confidence (highest "
        "first), via the `record_predictions` tool. Use ICD-10-CM dotted format "
        "(e.g., 'E11.9', 'I10', 'N18.31'). Provide the official display text for "
        "each code. Confidence values should reflect your true belief about each "
        "candidate.\n\n"
        "If you are uncertain, still return your best guess(es) — the eval framework "
        "will distinguish hallucinations (non-existent codes) from real-but-wrong codes."
    )

    if mode == PromptMode.ZEROSHOT:
        return base

    if mode == PromptMode.CONSTRAINED:
        if not candidates:
            raise ValueError("constrained mode requires a non-empty candidate list")
        candidate_block = "\n".join(
            f"- {c.code}: {c.display}" + (f" [{c.group.upper()}]" if c.group else "")
            for c in candidates
        )
        return (
            base
            + "\n\n"
            + "You MUST select codes from the following candidate list. Predicting any "
            "code not in this list is incorrect.\n\n"
            f"Candidate codes ({len(candidates)} total):\n{candidate_block}"
        )

    raise ValueError(f"unknown PromptMode: {mode!r}")


def build_user_message(
    input_fhir: dict[str, Any] | None,
    input_text: str | None,
) -> str:
    """Per-record portion of the prompt — varies per call, NOT cached."""
    if input_fhir is not None:
        return (
            "The clinical condition is described by this (degraded) FHIR Condition "
            "resource. Recover the ICD-10-CM code:\n\n"
            f"```json\n{json.dumps(input_fhir, indent=2)}\n```"
        )
    if input_text is not None:
        return (
            "The clinical condition is described by this text. Recover the "
            f"ICD-10-CM code:\n\n{input_text}"
        )
    raise ValueError("either input_fhir or input_text must be provided")


def parse_predictions(tool_input: dict[str, Any]) -> list[Prediction]:
    """Parse the model's structured response into our Prediction format.

    Sorts by confidence (descending). Codes are normalized (stripped, uppercased
    where applicable, but ICD-10-CM is letter+digits with optional dot — we just strip).
    """
    raw = tool_input.get("predictions", [])
    out: list[Prediction] = []
    for item in raw:
        code = (item.get("code") or "").strip()
        display = item.get("display")
        confidence = float(item.get("confidence", 0.0))
        if not code:
            continue
        out.append(
            Prediction(
                system=ICD10_SYSTEM,
                code=code,
                display=display,
                score=confidence,
            )
        )
    out.sort(key=lambda p: p.score, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Provider clients. Thin wrappers; the real testing target is the pure functions.
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    """Protocol for an LLM that returns structured `predictions` via tool/JSON output."""

    def predict_structured(
        self, system_prompt: str, user_message: str
    ) -> dict[str, Any]: ...


@dataclass
class AnthropicClient:
    """Anthropic Claude with prompt caching on the system block + forced tool use."""

    model_id: str  # e.g. "claude-sonnet-4-6", "claude-haiku-4-5"
    api_key: str | None = None
    max_tokens: int = 1024

    def predict_structured(self, system_prompt: str, user_message: str) -> dict[str, Any]:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key or os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[
                {
                    "name": PREDICTION_TOOL_NAME,
                    "description": PREDICTION_TOOL_DESCRIPTION,
                    "input_schema": PREDICTION_TOOL_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": PREDICTION_TOOL_NAME},
            messages=[{"role": "user", "content": user_message}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == PREDICTION_TOOL_NAME:
                return dict(block.input)
        raise RuntimeError(f"Anthropic response missing expected tool use: {resp}")


@dataclass
class OpenAIClient:
    """OpenAI GPT with JSON-mode structured output. Prompt caching is automatic."""

    model_id: str  # e.g. "gpt-4o", "gpt-4o-mini"
    api_key: str | None = None
    max_tokens: int = 1024

    def predict_structured(self, system_prompt: str, user_message: str) -> dict[str, Any]:
        import openai

        client = openai.OpenAI(api_key=self.api_key or os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------


class LLMModel(ConceptNormalizer):
    """LLM-backed `ConceptNormalizer`. Provider + mode are configured at construction."""

    def __init__(
        self,
        *,
        name: str,
        client: LLMClient,
        mode: PromptMode,
        candidates: list[CandidateCode] | None = None,
    ) -> None:
        self.name = name
        self._client = client
        self._mode = mode
        self._candidates = candidates
        # Eagerly build the system prompt; it's static per model instance.
        self._system_prompt = build_system_prompt(mode, candidates)

    @property
    def mode(self) -> PromptMode:
        return self._mode

    def predict(
        self,
        *,
        input_fhir: dict[str, Any] | None = None,
        input_text: str | None = None,
        top_k: int = 5,
    ) -> list[Prediction]:
        user = build_user_message(input_fhir=input_fhir, input_text=input_text)
        tool_input = self._client.predict_structured(self._system_prompt, user)
        predictions = parse_predictions(tool_input)
        return predictions[:top_k]


def make_anthropic_model(
    *,
    name: str,
    model_id: str,
    mode: PromptMode,
    candidates: list[CandidateCode] | None = None,
    api_key: str | None = None,
) -> LLMModel:
    """Convenience constructor for a Claude-backed model."""
    return LLMModel(
        name=name,
        client=AnthropicClient(model_id=model_id, api_key=api_key),
        mode=mode,
        candidates=candidates,
    )


def make_openai_model(
    *,
    name: str,
    model_id: str,
    mode: PromptMode,
    candidates: list[CandidateCode] | None = None,
    api_key: str | None = None,
) -> LLMModel:
    """Convenience constructor for a GPT-backed model."""
    return LLMModel(
        name=name,
        client=OpenAIClient(model_id=model_id, api_key=api_key),
        mode=mode,
        candidates=candidates,
    )
