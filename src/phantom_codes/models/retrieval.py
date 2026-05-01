"""Bi-encoder retrieval baseline: frozen sentence-transformer + cosine similarity.

A `ConceptNormalizer` that ranks the candidate list by semantic similarity to the input.
Used both as a standalone baseline in the eval matrix and as the retriever component of
`RAGLLMModel`.

Why frozen, not fine-tuned: v1 is a baseline. Fine-tuning the encoder is a P3 stretch.
This implementation uses a small, fast, well-tested encoder by default
(`sentence-transformers/all-MiniLM-L6-v2`, ~22M params). Swap in a biomedical encoder
(e.g., `pritamdeka/S-PubMedBert-MS-MARCO`) by passing `encoder_name`.

Why numpy (not FAISS) for similarity: our v1 candidate pool is the ACCESS scope
(low-hundreds of codes), where numpy `@` is fast enough and avoids FAISS as a hard
dependency at runtime. FAISS becomes worthwhile when the candidate space grows to
the full ICD-10-CM vocabulary (~75k codes); easy swap when needed.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from phantom_codes.data.degrade import ICD10_SYSTEM
from phantom_codes.data.disease_groups import CandidateCode
from phantom_codes.models.base import ConceptNormalizer, Prediction


def _query_text(
    input_fhir: dict[str, Any] | None,
    input_text: str | None,
) -> str:
    """Reduce a degraded input to a single string for embedding.

    Mirrors the logic in `baselines._query_text` so the retriever sees the same
    surface form a string-matching baseline would. The point of comparison between
    retrieval and TF-IDF is what the *encoder* does with that string, not whether
    the input was preprocessed differently.
    """
    if input_text is not None:
        return input_text
    if input_fhir is None:
        return ""
    code = input_fhir.get("code") or {}
    text = code.get("text")
    if text:
        return str(text)
    for coding in code.get("coding") or []:
        display = coding.get("display")
        if display:
            return str(display)
    return json.dumps(input_fhir)


class RetrievalModel(ConceptNormalizer):
    """Bi-encoder retrieval over candidate displays.

    Encodes all candidate displays once at construction; encodes the query at
    inference and returns top-k by cosine similarity. Candidate embeddings are
    L2-normalized so cosine similarity reduces to a dot product.
    """

    name = "retrieval"

    def __init__(
        self,
        candidates: list[CandidateCode],
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder: Any | None = None,
    ) -> None:
        """Construct the retriever.

        Parameters
        ----------
        candidates
            The candidate vocabulary to rank against.
        encoder_name
            Model id passed to `SentenceTransformer(...)`. Ignored if `encoder` is
            provided directly (used in tests to inject a fake).
        encoder
            Pre-constructed encoder for tests. Must implement
            `encode(texts, normalize_embeddings=True) -> np.ndarray`.
        """
        self._candidates = candidates
        self._encoder = encoder if encoder is not None else self._load_encoder(encoder_name)

        displays = [c.display or c.code for c in candidates]
        if displays:
            self._candidate_emb = self._encoder.encode(
                displays,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        else:
            self._candidate_emb = None

    @staticmethod
    def _load_encoder(name: str) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name)

    def predict(
        self,
        *,
        input_fhir: dict[str, Any] | None = None,
        input_text: str | None = None,
        top_k: int = 5,
    ) -> list[Prediction]:
        if self._candidate_emb is None:
            return []
        query = _query_text(input_fhir, input_text)
        if not query:
            return []
        q_emb = self._encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        # Normalized vectors → dot product == cosine similarity.
        sims = self._candidate_emb @ q_emb
        order = np.argsort(-sims)[:top_k]
        out: list[Prediction] = []
        for i in order:
            cand = self._candidates[int(i)]
            out.append(
                Prediction(
                    system=ICD10_SYSTEM,
                    code=cand.code,
                    display=cand.display,
                    score=float(sims[int(i)]),
                )
            )
        return out
