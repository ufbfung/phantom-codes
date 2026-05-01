"""LLM-based code recovery: Claude (Anthropic), GPT (OpenAI), Gemini (Google), with
two prompting modes.

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
- Google: implicit caching on stable prefixes; explicit `cachedContent` is a TODO
  (separate API resource, worth the wiring once eval volumes justify it).

Structured output:
- Anthropic: forced tool use with the `record_predictions` tool.
- OpenAI: JSON mode (`response_format=json_object`).
- Google: `response_schema` with `response_mime_type=application/json`.
All three return the same `{"predictions": [...]}` shape, parsed by `parse_predictions`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from phantom_codes.data.degrade import ICD10_SYSTEM
from phantom_codes.data.disease_groups import CandidateCode
from phantom_codes.models.base import ConceptNormalizer, Prediction


@dataclass(frozen=True)
class LLMResponse:
    """Structured output from an LLM call, plus token-usage metadata.

    Token fields default to 0 so test fakes that don't care about usage can be
    constructed minimally as `LLMResponse(tool_input={...})`. Real provider
    clients should populate all relevant fields from the response object.
    """

    tool_input: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    raw_provider_metadata: dict[str, Any] = field(default_factory=dict)


class PromptMode(StrEnum):
    ZEROSHOT = "zeroshot"
    CONSTRAINED = "constrained"


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


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
    ) -> LLMResponse: ...


@dataclass
class AnthropicClient:
    """Anthropic Claude with prompt caching on the system block + forced tool use."""

    model_id: str  # e.g. "claude-sonnet-4-6", "claude-haiku-4-5"
    api_key: str | None = None
    max_tokens: int = 1024

    def predict_structured(self, system_prompt: str, user_message: str) -> LLMResponse:
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
        tool_input: dict[str, Any] | None = None
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == PREDICTION_TOOL_NAME:
                tool_input = dict(block.input)
                break
        if tool_input is None:
            raise RuntimeError(f"Anthropic response missing expected tool use: {resp}")
        usage = resp.usage
        return LLMResponse(
            tool_input=tool_input,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            raw_provider_metadata={"model": resp.model, "stop_reason": resp.stop_reason},
        )


@dataclass
class OpenAIClient:
    """OpenAI GPT with JSON-mode structured output. Prompt caching is automatic."""

    model_id: str  # e.g. "gpt-4o", "gpt-4o-mini", "gpt-5.5"
    api_key: str | None = None
    max_tokens: int = 1024

    def predict_structured(self, system_prompt: str, user_message: str) -> LLMResponse:
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
        tool_input = json.loads(content)
        usage = resp.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        # OpenAI auto-caches; cached tokens are reported as a sub-count of prompt_tokens.
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        # Report input_tokens as the *uncached* portion so the math is consistent
        # across providers: input + cache_read = total billable input.
        return LLMResponse(
            tool_input=tool_input,
            input_tokens=max(prompt_tokens - cached, 0),
            output_tokens=completion_tokens,
            cache_read_tokens=cached,
            cache_creation_tokens=0,  # OpenAI doesn't surface cache writes separately
            raw_provider_metadata={"model": resp.model, "finish_reason": resp.choices[0].finish_reason},
        )


@dataclass
class GoogleClient:
    """Google Gemini with `response_schema` structured output.

    Uses the unified `google-genai` SDK (`from google import genai`). API key resolution
    order: explicit `api_key` arg → `GEMINI_API_KEY` → `GOOGLE_API_KEY` (the SDK accepts
    either env var, but we surface both for parity with how Brian sets things in `.env`).
    """

    model_id: str  # e.g. "gemini-2.5-pro", "gemini-2.5-flash"
    api_key: str | None = None
    max_tokens: int = 1024

    def predict_structured(self, system_prompt: str, user_message: str) -> LLMResponse:
        from google import genai
        from google.genai import types

        key = (
            self.api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ["GOOGLE_API_KEY"]
        )
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=self.model_id,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
                response_schema=PREDICTION_TOOL_SCHEMA,
            ),
        )
        text = resp.text or "{}"
        tool_input = json.loads(text)
        meta = getattr(resp, "usage_metadata", None)
        prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0 if meta else 0
        completion_tokens = getattr(meta, "candidates_token_count", 0) or 0 if meta else 0
        cached = getattr(meta, "cached_content_token_count", 0) or 0 if meta else 0
        return LLMResponse(
            tool_input=tool_input,
            input_tokens=max(prompt_tokens - cached, 0),
            output_tokens=completion_tokens,
            cache_read_tokens=cached,
            cache_creation_tokens=0,  # Gemini explicit caching uses separate cachedContent resources
            raw_provider_metadata={"model": self.model_id},
        )


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------


class LLMModel(ConceptNormalizer):
    """LLM-backed `ConceptNormalizer`. Provider + mode are configured at construction.

    After each `predict()` call, `last_usage` holds the `LLMResponse` from the most
    recent API call (tokens-in, tokens-out, cache reads/writes). The eval runner
    reads this to populate per-prediction cost columns.
    """

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
        self.last_usage: LLMResponse | None = None

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
        response = self._client.predict_structured(self._system_prompt, user)
        self.last_usage = response
        predictions = parse_predictions(response.tool_input)
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


def make_gemini_model(
    *,
    name: str,
    model_id: str,
    mode: PromptMode,
    candidates: list[CandidateCode] | None = None,
    api_key: str | None = None,
) -> LLMModel:
    """Convenience constructor for a Gemini-backed model."""
    return LLMModel(
        name=name,
        client=GoogleClient(model_id=model_id, api_key=api_key),
        mode=mode,
        candidates=candidates,
    )
