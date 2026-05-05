# S1 — Prompt templates and provider-specific structured-output configurations

This supplement reproduces the verbatim system + user prompts and
provider-specific structured-output configurations used in the
headline matrix. Three prompting modes (`zeroshot`, `constrained`,
`rag`) are evaluated; each is reproduced below per provider
(Anthropic, OpenAI, Google) for full reviewer auditability.

> **Status:** Skeleton — to be filled from
> `src/phantom_codes/models/llm.py` build_system_prompt /
> build_user_message functions and the per-provider tool-use /
> response_schema definitions before submission. Verbatim
> reproduction (not paraphrase) is the goal so reviewers can
> evaluate prompt sensitivity.

## S1.1 — Zero-shot system prompt (provider-agnostic body)

[TODO: paste the verbatim string returned by
`build_system_prompt(mode=PromptMode.ZEROSHOT, candidates=None)`
from `src/phantom_codes/models/llm.py`. Approximately 200-300 words
of system instruction.]

## S1.2 — Constrained system prompt (with candidate menu)

[TODO: paste the verbatim string returned by
`build_system_prompt(mode=PromptMode.CONSTRAINED, candidates=...)`,
with the full ACCESS-scope candidate list inlined. Approximately
2-3k tokens including the menu.]

## S1.3 — RAG system prompt template

[TODO: paste the verbatim template used by
`make_rag_anthropic_model` / `make_rag_openai_model` /
`make_rag_gemini_model`. Per-record retrieval populates the menu
slot at call time with the top-k retrieved candidates.]

## S1.4 — User-message template

[TODO: paste the verbatim string returned by
`build_user_message()` for each input shape (D1_full FHIR JSON
mode and D3_text_only / D4_abbreviated text-only mode).]

## S1.5 — Provider-specific structured-output configuration

### S1.5.1 Anthropic — forced tool use

[TODO: paste the `tools` and `tool_choice` configuration from
`AnthropicClient.predict_structured()`. The schema enforces a
`predictions` array of `{code, display, confidence}` objects.]

### S1.5.2 OpenAI — structured outputs (response_format)

[TODO: paste the `response_format` JSON-schema configuration from
`OpenAIClient.predict_structured()`. Equivalent semantics to
Anthropic's tool-forced output but expressed as a JSON schema.]

### S1.5.3 Google — response_schema

[TODO: paste the `response_schema` configuration from
`GoogleClient.predict_structured()`. Equivalent semantics; the
schema-stripping helper `_strip_for_gemini` is applied to remove
fields the Gemini SDK doesn't accept.]
