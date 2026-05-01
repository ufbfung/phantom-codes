"""ICD-10-CM existence check for hallucination detection.

Loads a bundled snapshot of CMS-published ICD-10-CM codes (FY2026, ~74k codes) and
provides O(1) `exists(code)` checks. Used by `eval/metrics.py` to distinguish
hallucinated predictions from real-but-wrong ones.

Source: CDC NCHS, https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/
Public domain.
"""

from __future__ import annotations

import gzip
from functools import cache
from importlib import resources

from phantom_codes.data.degrade import ICD10_SYSTEM

_BUNDLED_FILE = "icd10cm-codes-2026.txt.gz"


class Icd10CmValidator:
    """Concrete `IcdValidator` backed by an in-memory set of all real ICD-10-CM codes.

    Initialization loads ~75k codes (~5MB raw / ~170KB compressed). For the project's
    scale (one validator instance per process) this is the simplest correct implementation.
    """

    def __init__(self, codes: frozenset[str]) -> None:
        self._codes = codes

    def exists(self, system: str, code: str) -> bool:
        return system == ICD10_SYSTEM and code in self._codes

    @property
    def n_codes(self) -> int:
        return len(self._codes)


@cache
def load() -> Icd10CmValidator:
    """Load the bundled ICD-10-CM snapshot. Cached for the process."""
    pkg = resources.files("phantom_codes.data.icd10cm")
    with (pkg / _BUNDLED_FILE).open("rb") as raw:
        with gzip.open(raw, "rt", encoding="utf-8") as f:
            codes = frozenset(line.strip() for line in f if line.strip())
    return Icd10CmValidator(codes=codes)
