# S1 — Prompt templates and provider-specific structured-output configurations

This supplement reproduces the verbatim system + user prompts and
provider-specific structured-output configurations used in the
headline matrix. Three prompting modes (`zeroshot`, `constrained`,
`rag`) are evaluated; each is reproduced below per provider
(Anthropic, OpenAI, Google) for full reviewer auditability. All
content is extracted verbatim from
[`src/phantom_codes/models/llm.py`](https://github.com/ufbfung/phantom-codes/blob/main/src/phantom_codes/models/llm.py)
and
[`src/phantom_codes/models/rag_llm.py`](https://github.com/ufbfung/phantom-codes/blob/main/src/phantom_codes/models/rag_llm.py).

## S1.1 — Zero-shot system prompt

The zero-shot prompt is the static "base" string returned by
`build_system_prompt(mode=PromptMode.ZEROSHOT)`. The model receives
this once per record (it is the cacheable prefix on Anthropic; OpenAI
and Google also automatically cache long prefixes). `top_k` defaults
to 5 in the headline run.

```
You are a clinical informatics assistant. Your task is to recover the
standardized ICD-10-CM diagnosis code for a clinical condition described
by the user.

Return up to 5 candidate codes, ranked by your confidence (highest
first), via the `record_predictions` tool. Use ICD-10-CM dotted format
(e.g., 'E11.9', 'I10', 'N18.31'). Provide the official display text for
each code. Confidence values should reflect your true belief about each
candidate.

If you are uncertain, still return your best guess(es) — the eval framework
will distinguish hallucinations (non-existent codes) from real-but-wrong codes.
```

## S1.2 — Constrained system prompt

The constrained prompt appends the full CMS ACCESS-scope candidate
list to the zero-shot base. The candidate block is rendered one
candidate per line as `- <code>: <display> [<GROUP>]` (where
`<GROUP>` is `CKM` or `ECKM`). The headline run uses ~85 candidate
codes drawn from the CMS ACCESS Model FHIR Implementation Guide
v0.9.6 ValueSets bundled at
[`src/phantom_codes/data/access_valuesets/`](https://github.com/ufbfung/phantom-codes/tree/main/src/phantom_codes/data/access_valuesets/).

```
You are a clinical informatics assistant. Your task is to recover the
standardized ICD-10-CM diagnosis code for a clinical condition described
by the user.

Return up to 5 candidate codes, ranked by your confidence (highest
first), via the `record_predictions` tool. Use ICD-10-CM dotted format
(e.g., 'E11.9', 'I10', 'N18.31'). Provide the official display text for
each code. Confidence values should reflect your true belief about each
candidate.

If you are uncertain, still return your best guess(es) — the eval framework
will distinguish hallucinations (non-existent codes) from real-but-wrong codes.

You MUST select codes from the following candidate list. Predicting any
code not in this list is incorrect.

Candidate codes (<N> total):
- E08.00: Diabetes mellitus due to underlying condition with hyperosmolarity without nonketotic hyperglycemic-hyperosmolar coma (NKHHC) [CKM]
- E08.01: Diabetes mellitus due to underlying condition with hyperosmolarity with coma [CKM]
- ...
- E11.9: Type 2 diabetes mellitus without complications [CKM]
- ...
- I10: Essential (primary) hypertension [ECKM]
- ...
- N18.31: Chronic kidney disease, stage 3a [CKM]
- ...
```

## S1.3 — RAG system prompt template

The RAG prompt reuses the constrained-mode template, but the
candidate list is populated per-record by a frozen sentence-
transformer retriever (default `all-MiniLM-L6-v2`) returning the
top-`retrieve_k = 20` ACCESS-scope candidates most similar to the
input. The structural form is therefore identical to S1.2; only the
size and identity of the candidate block change. From
`RAGLLMModel.predict` in
[`src/phantom_codes/models/rag_llm.py`](https://github.com/ufbfung/phantom-codes/blob/main/src/phantom_codes/models/rag_llm.py):

```python
# Stage 1 — retrieve top-K candidates for this specific input.
retrieved = self._retriever.predict(
    input_fhir=input_fhir,
    input_text=input_text,
    top_k=self._retrieve_k,  # default 20
)

# Stage 2 — build a constrained-mode prompt with these per-record candidates
# and call the LLM.
system_prompt = build_system_prompt(
    PromptMode.CONSTRAINED,
    candidates=candidates,
    top_k=top_k,
)
```

## S1.4 — User-message template

The user message varies by input mode. For D1\_full and D2\_no\_code
inputs the FHIR Condition resource is serialized to indented JSON;
for D3\_text\_only and D4\_abbreviated inputs the text is sent
directly. Returned by `build_user_message(input_fhir=…, input_text=…)`:

```
The clinical condition is described by this (degraded) FHIR Condition
resource. Recover the ICD-10-CM code:

```json
{
  "resourceType": "Condition",
  "code": { ... }
  ...
}
```
```

For text-only modes:

```
The clinical condition is described by this text. Recover the
ICD-10-CM code:

<input_text>
```

## S1.5 — Provider-specific structured-output configuration

All three providers are constrained to emit the same
`PREDICTION_TOOL_SCHEMA` shape. The schema is defined once in
`src/phantom_codes/models/llm.py` and adapted per-provider:

```json
{
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
            "description": "ICD-10-CM code in dotted format, e.g. 'E11.9' or 'I10'."
          },
          "display": {
            "type": "string",
            "description": "Human-readable description of the code."
          },
          "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Model's confidence, higher = more confident."
          }
        },
        "required": ["code", "display", "confidence"],
        "additionalProperties": false
      }
    }
  },
  "required": ["predictions"],
  "additionalProperties": false
}
```

### S1.5.1 Anthropic — forced tool use

The schema is wrapped in a single tool with `tool_choice` forced to
that tool, so the model must invoke it. The system block is marked
`cache_control: ephemeral` for prompt caching.

```python
client.messages.create(
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
            "name": "record_predictions",
            "description": (
                "Record the model's ranked ICD-10-CM code predictions "
                "for the given clinical input."
            ),
            "input_schema": PREDICTION_TOOL_SCHEMA,
        }
    ],
    tool_choice={"type": "tool", "name": "record_predictions"},
    messages=[{"role": "user", "content": user_message}],
)
```

### S1.5.2 OpenAI — structured outputs (`response_format`)

OpenAI's strict `json_schema` mode requires a slightly trimmed schema
(strict-mode rejects `minItems`, `maxItems`, `minimum`, `maximum`,
`pattern`, `format`); a deep-copy helper `_strip_for_openai_strict`
removes those fields. GPT-5+ and reasoning models (o1, o3, o4)
require `max_completion_tokens` instead of the legacy `max_tokens`.

```python
strict_schema = _strip_for_openai_strict(PREDICTION_TOOL_SCHEMA)
client.chat.completions.create(
    model=self.model_id,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "icd10_predictions",
            "strict": True,
            "schema": strict_schema,
        },
    },
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ],
    **{token_kwarg: self.max_tokens},  # token_kwarg: "max_completion_tokens" for GPT-5+/o-series, else "max_tokens"
)
```

### S1.5.3 Google — `response_schema`

Gemini's `response_schema` is an OpenAPI 3.0 subset that rejects JSON
Schema fields the other providers accept — most notably
`additionalProperties` and `$schema`. A deep-copy helper
`_strip_for_gemini` removes those fields. The system instruction is
passed as a separate `system_instruction` config field; output is
constrained to `application/json` with the trimmed schema.

```python
config = types.GenerateContentConfig(
    system_instruction=system_prompt,
    max_output_tokens=self.max_tokens,
    response_mime_type="application/json",
    response_schema=_strip_for_gemini(PREDICTION_TOOL_SCHEMA),
)
client.models.generate_content(
    model=self.model_id,
    contents=user_message,
    config=config,
)
```

The Google client wraps the call in a 429-only retry loop (one
outer attempt) that respects the server's `retryDelay` hint; 5xx
retries are handled by the SDK's internal backoff (5 attempts,
exponential 1→16s with jitter).
