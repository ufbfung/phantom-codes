"""Abstract base class for phantom-codes models.

Every model — LLM, retrieval, classifier, baseline — implements `predict()` and returns a
list of `Prediction` objects ranked by score (descending). Uniformity here is what makes the
evaluation matrix fair across model families.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Prediction:
    """A single (system, code) prediction with an optional human-readable display + score.

    `score` is whatever the model uses internally for ranking (logit, similarity, log-prob).
    Comparable only within a single model's outputs.
    """

    system: str
    code: str
    display: str | None
    score: float


class ConceptNormalizer(ABC):
    """Common interface for all models in the benchmark.

    Subclasses must accept either `input_fhir` (a FHIR Condition dict) or `input_text`
    (a string), depending on the degradation mode. They must return predictions sorted
    by score, descending. `top_k` controls how many to return.
    """

    name: str

    @abstractmethod
    def predict(
        self,
        *,
        input_fhir: dict[str, Any] | None = None,
        input_text: str | None = None,
        top_k: int = 5,
    ) -> list[Prediction]:
        """Predict ICD codes for a single degraded input. Return at most `top_k`, ranked."""

    def predict_batch(
        self,
        inputs: list[tuple[dict[str, Any] | None, str | None]],
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        """Default batch implementation — subclasses can override for efficiency."""
        return [
            self.predict(input_fhir=fhir, input_text=text, top_k=top_k)
            for fhir, text in inputs
        ]
