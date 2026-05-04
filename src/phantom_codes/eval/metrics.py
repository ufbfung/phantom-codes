"""Per-prediction outcome classification + aggregate metrics.

Six-way taxonomy aligned with the literature:

- EXACT_MATCH       — predicted code equals truth (exact match / top-1 accuracy)
- CATEGORY_MATCH    — same 3-char ICD-10 category (e.g., E11.0 vs E11.9 both Type 2 DM)
                       (block-level / category-level match in ICD coding papers, e.g., CAML)
- CHAPTER_MATCH     — same ICD-10 chapter (first letter), different category
                       (chapter-level match)
- OUT_OF_DOMAIN     — real ICD-10-CM code, no hierarchical relation to truth
                       (OOD prediction, Hendrycks & Gimpel 2017)
- NO_PREDICTION     — model returned no usable prediction (empty `predictions` array,
                       transient API failure, or refusal). Distinct from HALLUCINATION:
                       the model didn't fabricate anything, it abstained. From a
                       deployment-safety perspective, abstention is strictly preferable
                       to fabrication (no spurious code propagates downstream); from a
                       model-quality perspective, persistent abstention indicates the
                       model isn't useful for the task.
- HALLUCINATION     — predicted code does NOT exist in ICD-10-CM
                       (fabrication / hallucination, narrow definition; Ji et al. 2023)

Outcomes are mutually exclusive, exhaustive, and ordered by quality (best→worst).
Every prediction lands in exactly one bucket. Note the rank ordering treats abstention
(NO_PREDICTION) as preferable to fabrication (HALLUCINATION) — see the deployment-
safety rationale above; persistent abstention is still a quality concern but doesn't
emit harmful artifacts the way fabrication does.

Top-k variants apply the same classification to ranked predictions and take the *best*
outcome among the top-k.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from phantom_codes.data.degrade import ICD10_SYSTEM
from phantom_codes.models.base import Prediction


class Outcome(StrEnum):
    EXACT_MATCH = "exact_match"
    CATEGORY_MATCH = "category_match"
    CHAPTER_MATCH = "chapter_match"
    OUT_OF_DOMAIN = "out_of_domain"
    NO_PREDICTION = "no_prediction"
    HALLUCINATION = "hallucination"


# Ordered best→worst. Used for top-k aggregation: take the best outcome in the top-k slice.
# NO_PREDICTION is ranked BETTER than HALLUCINATION because abstention does not emit a
# spurious code that propagates downstream — see the module docstring for the safety
# rationale.
OUTCOME_RANK: dict[Outcome, int] = {
    Outcome.EXACT_MATCH: 0,
    Outcome.CATEGORY_MATCH: 1,
    Outcome.CHAPTER_MATCH: 2,
    Outcome.OUT_OF_DOMAIN: 3,
    Outcome.NO_PREDICTION: 4,
    Outcome.HALLUCINATION: 5,
}


class IcdValidator(Protocol):
    """Tells us whether a given (system, code) is a real ICD-10-CM code.

    Used solely for hallucination detection. Implementations may load from a bundled
    CMS table, query a terminology server, etc. Tests can use an in-memory stub.
    """

    def exists(self, system: str, code: str) -> bool: ...


@dataclass(frozen=True)
class Truth:
    """Ground-truth (system, code) pair."""

    system: str
    code: str


def category(code: str) -> str:
    """ICD-10 3-char category prefix (e.g., 'E11.9' → 'E11', 'I10' → 'I10')."""
    head = code.split(".", 1)[0]
    return head[:3] if len(head) >= 3 else head


def chapter(code: str) -> str:
    """ICD-10 chapter — first character (e.g., 'E11.9' → 'E')."""
    return code[:1] if code else ""


def classify(prediction: Prediction, truth: Truth, validator: IcdValidator) -> Outcome:
    """Classify a single prediction against ground truth.

    Decision order (mutually exclusive):
    1. EXACT_MATCH if same system + code
    2. HALLUCINATION if predicted code doesn't exist in the ICD-10-CM tabular list
       (or system isn't ICD-10-CM at all — counts as fabrication of the wrong domain)
    3. CATEGORY_MATCH if same 3-char ICD-10 category prefix
    4. CHAPTER_MATCH if same chapter (first character)
    5. OUT_OF_DOMAIN otherwise (real ICD-10 code, no hierarchical relation)
    """
    if prediction.system == truth.system and prediction.code == truth.code:
        return Outcome.EXACT_MATCH

    # Hallucination check: predicted code must exist in the tabular ICD-10-CM list.
    # If predicted system is not ICD-10-CM, treat as hallucination (we asked for
    # ICD-10-CM and the model returned something else — non-existent in our domain).
    if prediction.system != ICD10_SYSTEM:
        return Outcome.HALLUCINATION
    if not validator.exists(prediction.system, prediction.code):
        return Outcome.HALLUCINATION

    if category(prediction.code) == category(truth.code):
        return Outcome.CATEGORY_MATCH
    if chapter(prediction.code) == chapter(truth.code):
        return Outcome.CHAPTER_MATCH
    return Outcome.OUT_OF_DOMAIN


def best_outcome_in_topk(
    predictions: Sequence[Prediction],
    truth: Truth,
    validator: IcdValidator,
    k: int,
) -> Outcome:
    """Return the best (lowest-rank) outcome among the top-k predictions.

    If there are fewer than k predictions, classify what we have. If there are zero
    predictions, return NO_PREDICTION (model abstained / failed to return anything).
    Distinct from HALLUCINATION because nothing was fabricated.
    """
    if not predictions:
        return Outcome.NO_PREDICTION
    sliced = list(predictions[:k])
    outcomes = [classify(p, truth, validator) for p in sliced]
    return min(outcomes, key=lambda o: OUTCOME_RANK[o])


@dataclass
class MetricsSummary:
    """Aggregated outcome counts and per-outcome rates over a set of predictions."""

    n: int
    counts: dict[Outcome, int]

    @property
    def exact_match_rate(self) -> float:
        return self.rate(Outcome.EXACT_MATCH)

    @property
    def hallucination_rate(self) -> float:
        return self.rate(Outcome.HALLUCINATION)

    def rate(self, outcome: Outcome) -> float:
        if self.n == 0:
            return 0.0
        return self.counts.get(outcome, 0) / self.n

    def to_dict(self) -> dict[str, float]:
        return {
            "n": float(self.n),
            **{outcome.value: self.rate(outcome) for outcome in Outcome},
        }


def summarize(outcomes: Sequence[Outcome]) -> MetricsSummary:
    """Bucket outcomes into a summary (counts + rates)."""
    counts = Counter(outcomes)
    return MetricsSummary(
        n=len(outcomes),
        counts={o: counts.get(o, 0) for o in Outcome},
    )
