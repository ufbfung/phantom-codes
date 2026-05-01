"""ACCESS Model scope: which ICD-10-CM codes are in our research scope.

Loads two ValueSets bundled with the package, both pulled from
https://dsacms.github.io/cmmi-access-model/ (CMS ACCESS Model FHIR IG, v0.9.6):

- ACCESSCKMDiagnosisVS — diabetes, ASCVD, CKD stage 3
- ACCESSeCKMDiagnosisVS — hypertension, dyslipidemia, prediabetes, obesity

We do NOT include MSK or BH ValueSets in v1.

ValueSet semantics implemented here:
- `compose.include[].filter` with `op: is-a` and an ICD-10-CM concept → all codes equal to or
  hierarchically under that concept (children are dot-separated, e.g., E11.9 is-a E11).
- `compose.include[].concept` → explicit codes.
- `compose.exclude[].concept` → explicit codes removed from the include set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Literal

from phantom_codes.data.degrade import ICD10_SYSTEM

GroupName = Literal["ckm", "eckm"]


@dataclass(frozen=True)
class CandidateCode:
    """A specific ICD-10-CM code listed in an ACCESS ValueSet."""

    code: str
    display: str
    group: GroupName


@dataclass
class _ValueSetRules:
    """Parsed include/exclude rules for one ValueSet."""

    is_a_prefixes: list[str]
    explicit: dict[str, str]  # code -> display
    excludes: set[str]


@dataclass
class AccessScope:
    """Combined CKM + eCKM scope. Use the module-level `load()` to construct."""

    ckm: _ValueSetRules
    eckm: _ValueSetRules

    def is_in_scope(self, code: str) -> bool:
        """True if `code` is in either CKM or eCKM scope after applying excludes."""
        return self.group_for(code) is not None

    def group_for(self, code: str) -> GroupName | None:
        """Return 'ckm', 'eckm', or None. CKM takes precedence on overlap."""
        if self._matches(code, self.ckm):
            return "ckm"
        if self._matches(code, self.eckm):
            return "eckm"
        return None

    def candidate_codes(self) -> list[CandidateCode]:
        """All explicit codes from both ValueSets (i.e., the `concept` arrays).

        Note: this does NOT enumerate `is-a` expansions (e.g., it does not list every
        E11.x child of `is-a E11`). Sufficient as a starting candidate vocabulary; for
        the full classifier vocab we'll filter MIMIC's observed codes via `is_in_scope`.
        """
        out: list[CandidateCode] = []
        for code, display in self.ckm.explicit.items():
            if code not in self.ckm.excludes:
                out.append(CandidateCode(code=code, display=display, group="ckm"))
        for code, display in self.eckm.explicit.items():
            if code not in self.eckm.excludes:
                out.append(CandidateCode(code=code, display=display, group="eckm"))
        return out

    def candidates_for_codes(
        self, observed: list[tuple[str, str]]
    ) -> list[CandidateCode]:
        """Build a candidate list from observed (code, display) pairs in the cohort.

        Combines explicit ValueSet candidates with codes actually seen in the data
        (e.g., E11.9 from MIMIC, which is in scope via `is-a E11` but isn't enumerated).
        Used for the constrained-LLM and baseline candidate lists in real eval runs.
        """
        out = list(self.candidate_codes())
        seen = {c.code for c in out}
        for code, display in observed:
            if code in seen:
                continue
            group = self.group_for(code)
            if group is None:
                continue
            out.append(CandidateCode(code=code, display=display or code, group=group))
            seen.add(code)
        return out

    @staticmethod
    def _matches(code: str, rules: _ValueSetRules) -> bool:
        if code in rules.excludes:
            return False
        if code in rules.explicit:
            return True
        for prefix in rules.is_a_prefixes:
            if _is_a(code, prefix):
                return True
        return False


def _is_a(code: str, prefix: str) -> bool:
    """ICD-10-CM `is-a`: `code` is `prefix` itself or a dotted descendant."""
    return code == prefix or code.startswith(prefix + ".")


def _parse_valueset(data: dict) -> _ValueSetRules:
    is_a_prefixes: list[str] = []
    explicit: dict[str, str] = {}
    for inc in data.get("compose", {}).get("include", []):
        if inc.get("system") and inc["system"] != ICD10_SYSTEM:
            raise ValueError(f"Unexpected system {inc['system']!r} — expected ICD-10-CM")
        for f in inc.get("filter", []):
            if f.get("op") == "is-a":
                is_a_prefixes.append(f["value"])
        for concept in inc.get("concept", []):
            explicit[concept["code"]] = concept.get("display", "")

    excludes: set[str] = set()
    for exc in data.get("compose", {}).get("exclude", []):
        for concept in exc.get("concept", []):
            excludes.add(concept["code"])

    return _ValueSetRules(is_a_prefixes=is_a_prefixes, explicit=explicit, excludes=excludes)


@cache
def load() -> AccessScope:
    """Load both ACCESS ValueSets bundled with the package. Cached for the process."""
    pkg = resources.files("phantom_codes.data.access_valuesets")
    with (pkg / "ACCESSCKMDiagnosisVS.json").open("r") as f:
        ckm = _parse_valueset(json.load(f))
    with (pkg / "ACCESSeCKMDiagnosisVS.json").open("r") as f:
        eckm = _parse_valueset(json.load(f))
    return AccessScope(ckm=ckm, eckm=eckm)
