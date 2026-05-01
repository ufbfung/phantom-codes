"""Constrained-only baselines: exact, fuzzy, TF-IDF.

All baselines work over a fixed candidate list — they have no notion of "open prediction"
because they have no prior knowledge about ICD-10 outside what's in the candidate set.
This is the natural floor against which LLMs and trained models are compared.

For a degraded FHIR Condition input, the input "text" is whatever survived the degradation:
- D1/D2: derived from the FHIR resource (display + text fields)
- D3:    the `code.text` field
- D4:    the natural-language summary

We extract a single text snippet from the input (FHIR or text), then match it against the
candidate displays.
"""

from __future__ import annotations

import json
from typing import Any

from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from phantom_codes.data.degrade import ICD10_SYSTEM
from phantom_codes.data.disease_groups import CandidateCode
from phantom_codes.models.base import ConceptNormalizer, Prediction


def _query_text(
    input_fhir: dict[str, Any] | None,
    input_text: str | None,
) -> str:
    """Reduce a degraded input to a single string for matching."""
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
    # Fall back to the whole serialized resource — TF-IDF can still find signal.
    return json.dumps(input_fhir)


class ExactMatchBaseline(ConceptNormalizer):
    """Exact (case-insensitive) string match on the candidate display."""

    name = "exact"

    def __init__(self, candidates: list[CandidateCode]) -> None:
        self._by_display = {c.display.lower(): c for c in candidates if c.display}

    def predict(
        self,
        *,
        input_fhir: dict[str, Any] | None = None,
        input_text: str | None = None,
        top_k: int = 5,
    ) -> list[Prediction]:
        q = _query_text(input_fhir, input_text).strip().lower()
        if not q:
            return []
        match = self._by_display.get(q)
        if match is None:
            return []
        return [
            Prediction(
                system=ICD10_SYSTEM,
                code=match.code,
                display=match.display,
                score=1.0,
            )
        ][:top_k]


class FuzzyMatchBaseline(ConceptNormalizer):
    """rapidfuzz token-set ratio against candidate displays. Returns top-k by similarity."""

    name = "fuzzy"

    def __init__(self, candidates: list[CandidateCode]) -> None:
        self._candidates = candidates
        self._displays = [c.display for c in candidates]

    def predict(
        self,
        *,
        input_fhir: dict[str, Any] | None = None,
        input_text: str | None = None,
        top_k: int = 5,
    ) -> list[Prediction]:
        q = _query_text(input_fhir, input_text)
        if not q or not self._displays:
            return []
        # `process.extract` returns (choice, score, index).
        results = process.extract(
            q,
            self._displays,
            scorer=fuzz.token_set_ratio,
            limit=top_k,
        )
        out: list[Prediction] = []
        for _, score, idx in results:
            cand = self._candidates[idx]
            out.append(
                Prediction(
                    system=ICD10_SYSTEM,
                    code=cand.code,
                    display=cand.display,
                    score=score / 100.0,
                )
            )
        return out


class TfidfBaseline(ConceptNormalizer):
    """TF-IDF over candidate displays + cosine similarity ranking."""

    name = "tfidf"

    def __init__(self, candidates: list[CandidateCode]) -> None:
        self._candidates = candidates
        displays = [c.display or "" for c in candidates]
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
        )
        if displays and any(displays):
            self._candidate_matrix = self._vectorizer.fit_transform(displays)
        else:
            self._candidate_matrix = None

    def predict(
        self,
        *,
        input_fhir: dict[str, Any] | None = None,
        input_text: str | None = None,
        top_k: int = 5,
    ) -> list[Prediction]:
        if self._candidate_matrix is None:
            return []
        q = _query_text(input_fhir, input_text)
        if not q:
            return []
        q_vec = self._vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self._candidate_matrix).ravel()
        # argsort descending; take top_k.
        order = sims.argsort()[::-1][:top_k]
        out: list[Prediction] = []
        for i in order:
            score = float(sims[i])
            if score <= 0.0:
                continue
            cand = self._candidates[i]
            out.append(
                Prediction(
                    system=ICD10_SYSTEM,
                    code=cand.code,
                    display=cand.display,
                    score=score,
                )
            )
        return out
