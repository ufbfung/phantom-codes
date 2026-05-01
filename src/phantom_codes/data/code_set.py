"""Build the ICD-10 code vocabulary used by the classifier head.

We pick the top-N most frequent ICD-10-CM codes in the cohort. The retrieval model and LLMs
use the full value-set candidate space and don't need this vocabulary.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from phantom_codes.data.degrade import ICD10_SYSTEM, extract_ground_truth


@dataclass(frozen=True)
class CodeVocab:
    """Ordered list of (system, code, display) tuples — index in the list = class id."""

    entries: list[tuple[str, str, str | None]]

    @property
    def size(self) -> int:
        return len(self.entries)

    def index(self, system: str, code: str) -> int | None:
        for i, (s, c, _) in enumerate(self.entries):
            if s == system and c == code:
                return i
        return None

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {"id": i, "system": s, "code": c, "display": d}
            for i, (s, c, d) in enumerate(self.entries)
        ]


def build_vocab(
    conditions: Iterable[dict[str, Any]],
    top_n: int,
    system: str = ICD10_SYSTEM,
) -> CodeVocab:
    """Pick the top-N most frequent codes (in `system`) across the input conditions."""
    counter: Counter[tuple[str, str]] = Counter()
    displays: dict[tuple[str, str], str] = {}

    for cond in conditions:
        try:
            gt = extract_ground_truth(cond)
        except ValueError:
            continue
        if gt.system != system:
            continue
        key = (gt.system, gt.code)
        counter[key] += 1
        if gt.display and key not in displays:
            displays[key] = gt.display

    top = counter.most_common(top_n)
    entries = [(s, c, displays.get((s, c))) for (s, c), _ in top]
    return CodeVocab(entries=entries)
