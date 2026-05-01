"""RAG-LLM: per-record retrieve-then-prompt baseline.

Composes a retriever (any `ConceptNormalizer` returning ranked candidates) with an
`LLMClient`, building a *per-record* constrained-mode prompt from the retrieved
candidates. This is distinct from the standalone `LLMModel(mode=CONSTRAINED, ...)`,
which uses a *fixed* candidate list across the whole cohort.

Why this matters for the benchmark:
- `LLMModel(mode=CONSTRAINED)` answers: "given a fixed menu of codes that might apply
  across the cohort, can the LLM pick the right one?"
- `RAGLLMModel` answers: "given the K codes most semantically similar to *this*
  input, can the LLM pick the right one?"

The two are not interchangeable. Constrained tests menu-following; RAG tests the
combination of retrieval quality and LLM-as-reranker.

Caching note:
    The system prompt changes per record (because the retrieved candidate list does),
    so Anthropic prompt caching does not help here — each call pays the full
    system-prompt token cost. This is expected and inherent to RAG. For headline-run
    cost accounting, RAG-LLM will be more expensive per call than its
    fixed-constrained counterpart. Captured in cost-economics analysis.
"""

from __future__ import annotations

from typing import Any

from phantom_codes.data.disease_groups import CandidateCode
from phantom_codes.models.base import ConceptNormalizer, Prediction
from phantom_codes.models.llm import (
    AnthropicClient,
    GoogleClient,
    LLMClient,
    OpenAIClient,
    PromptMode,
    build_system_prompt,
    build_user_message,
    parse_predictions,
)


class RAGLLMModel(ConceptNormalizer):
    """Retrieve-top-K-then-constrained-prompt LLM model."""

    def __init__(
        self,
        *,
        name: str,
        client: LLMClient,
        retriever: ConceptNormalizer,
        candidates: list[CandidateCode],
        retrieve_k: int = 20,
    ) -> None:
        """Construct the RAG-LLM.

        Parameters
        ----------
        name
            Model name for the eval matrix (e.g., "claude-haiku-4-5:rag").
        client
            An `LLMClient` (Anthropic, OpenAI, Google).
        retriever
            Any `ConceptNormalizer`; called with `top_k=retrieve_k` per record.
            Typical choice is `RetrievalModel` over the same `candidates` list.
        candidates
            The full candidate vocabulary. Used to look up `CandidateCode` metadata
            (group, display) for codes the retriever returns.
        retrieve_k
            Number of candidates to retrieve per record and embed in the prompt.
            Default 20 — enough to give the LLM real choice without inflating
            prompt tokens.
        """
        self.name = name
        self._client = client
        self._retriever = retriever
        self._retrieve_k = retrieve_k
        self._candidate_lookup: dict[str, CandidateCode] = {c.code: c for c in candidates}

    def predict(
        self,
        *,
        input_fhir: dict[str, Any] | None = None,
        input_text: str | None = None,
        top_k: int = 5,
    ) -> list[Prediction]:
        # Stage 1 — retrieve top-K candidates for this specific input.
        retrieved = self._retriever.predict(
            input_fhir=input_fhir,
            input_text=input_text,
            top_k=self._retrieve_k,
        )
        if not retrieved:
            return []

        # Map retrieved codes back to full CandidateCode metadata (group, display).
        # Drop any retrieved codes not in the candidate pool — shouldn't happen if
        # retriever and RAG share the same `candidates` list, but be defensive.
        candidates = [
            self._candidate_lookup[p.code]
            for p in retrieved
            if p.code in self._candidate_lookup
        ]
        if not candidates:
            return []

        # Stage 2 — build a constrained-mode prompt with these per-record candidates
        # and call the LLM.
        system_prompt = build_system_prompt(
            PromptMode.CONSTRAINED,
            candidates=candidates,
            top_k=top_k,
        )
        user_message = build_user_message(input_fhir=input_fhir, input_text=input_text)
        tool_input = self._client.predict_structured(system_prompt, user_message)
        predictions = parse_predictions(tool_input)
        return predictions[:top_k]


def make_rag_anthropic_model(
    *,
    name: str,
    model_id: str,
    retriever: ConceptNormalizer,
    candidates: list[CandidateCode],
    retrieve_k: int = 20,
    api_key: str | None = None,
) -> RAGLLMModel:
    """Convenience constructor for a Claude-backed RAG-LLM."""
    return RAGLLMModel(
        name=name,
        client=AnthropicClient(model_id=model_id, api_key=api_key),
        retriever=retriever,
        candidates=candidates,
        retrieve_k=retrieve_k,
    )


def make_rag_openai_model(
    *,
    name: str,
    model_id: str,
    retriever: ConceptNormalizer,
    candidates: list[CandidateCode],
    retrieve_k: int = 20,
    api_key: str | None = None,
) -> RAGLLMModel:
    """Convenience constructor for a GPT-backed RAG-LLM."""
    return RAGLLMModel(
        name=name,
        client=OpenAIClient(model_id=model_id, api_key=api_key),
        retriever=retriever,
        candidates=candidates,
        retrieve_k=retrieve_k,
    )


def make_rag_gemini_model(
    *,
    name: str,
    model_id: str,
    retriever: ConceptNormalizer,
    candidates: list[CandidateCode],
    retrieve_k: int = 20,
    api_key: str | None = None,
) -> RAGLLMModel:
    """Convenience constructor for a Gemini-backed RAG-LLM."""
    return RAGLLMModel(
        name=name,
        client=GoogleClient(model_id=model_id, api_key=api_key),
        retriever=retriever,
        candidates=candidates,
        retrieve_k=retrieve_k,
    )
