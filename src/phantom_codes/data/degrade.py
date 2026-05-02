"""Degradation pipeline: produces (input, ground_truth) records from a coded FHIR Condition.

Four modes isolate the contribution of structured FHIR context vs. plain text:

- D1_full         : nothing removed (control / upper bound)
- D2_no_code      : strip coding.code + coding.system; keep display, text, status, category
- D3_text_only    : strip all coding; keep only `text`
- D4_abbreviated  : convert to a one-sentence summary using clinical abbreviations
                    (T2DM, HTN, CKD, etc.) — strips the canonical display tokens that
                    baselines lean on, while preserving the underlying diagnosis

Determinism: every mode is deterministic given the input. D4 uses a fixed
abbreviation table loaded from `abbreviations.yaml`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from phantom_codes.data.abbreviate import abbreviate

# Canonical (HL7-standard) FHIR system URIs for ICD coding. These are
# what the rest of the codebase — scope filter, validator, eval matrix —
# uses internally. We normalize source-specific URIs to these at the
# extraction boundary (`extract_ground_truth`).
ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10-cm"
ICD9_SYSTEM = "http://hl7.org/fhir/sid/icd-9-cm"
ICD_SYSTEMS = frozenset({ICD10_SYSTEM, ICD9_SYSTEM})

# MIMIC-FHIR uses its own namespaced CodeSystems instead of the canonical
# HL7 URIs. We treat them as equivalent and normalize on extraction so
# downstream code only ever sees the canonical URI. Source:
# https://kind-lab.github.io/mimic-fhir/CodeSystem-mimic-diagnosis-icd10.html
_MIMIC_ICD10_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-diagnosis-icd10"
_MIMIC_ICD9_SYSTEM = "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-diagnosis-icd9"

# Lookup: any URI in the keys is recognized; the value is the canonical URI it normalizes to.
_SYSTEM_ALIASES: dict[str, str] = {
    ICD10_SYSTEM: ICD10_SYSTEM,
    ICD9_SYSTEM: ICD9_SYSTEM,
    _MIMIC_ICD10_SYSTEM: ICD10_SYSTEM,
    _MIMIC_ICD9_SYSTEM: ICD9_SYSTEM,
}


def _normalize_icd_code(code: str) -> str:
    """Canonicalize an ICD-9/ICD-10 code to the dotted representation.

    MIMIC stores codes without the conventional decimal point (e.g.,
    ``E1110`` instead of ``E11.10``). The convention for both ICD-9 and
    ICD-10 is a dot after the first three characters when the code is
    longer than three characters; 3-character chapter-level codes
    (``E11``, ``I10``) are unchanged. Already-dotted codes pass through
    unchanged so the function is idempotent and safe for canonical-form
    inputs (Synthea, hand-built fixtures).
    """
    if not code or "." in code or len(code) <= 3:
        return code
    return f"{code[:3]}.{code[3:]}"


class DegradationMode(StrEnum):
    D1_FULL = "D1_full"
    D2_NO_CODE = "D2_no_code"
    D3_TEXT_ONLY = "D3_text_only"
    D4_ABBREVIATED = "D4_abbreviated"


@dataclass(frozen=True)
class GroundTruth:
    """The (system, code) pair we expect the model to recover."""

    system: str
    code: str
    display: str | None


@dataclass(frozen=True)
class DegradedRecord:
    """A single (input, ground_truth) record produced by the degradation pipeline."""

    resource_id: str
    mode: DegradationMode
    input_fhir: dict[str, Any] | None
    input_text: str | None
    ground_truth: GroundTruth


def extract_ground_truth(condition: dict[str, Any]) -> GroundTruth:
    """Pull the ICD coding out of a Condition resource as ground truth.

    Picks the first coding[] entry whose system is recognized as ICD-9 or
    ICD-10 (canonical HL7 URIs *or* MIMIC-namespaced URIs — both
    normalize to the canonical URI on output). Codes are canonicalized
    to the conventional dotted form (``E1110`` → ``E11.10``).

    Falls back to the first coding[] entry if no ICD coding is found
    (shouldn't happen for MimicCondition, but keeps the function total).
    """
    coding_list = _coding_list(condition)
    if not coding_list:
        raise ValueError(f"Condition {condition.get('id')!r} has no code.coding[]")

    for coding in coding_list:
        canonical = _SYSTEM_ALIASES.get(coding.get("system", ""))
        if canonical is not None:
            return GroundTruth(
                system=canonical,
                code=_normalize_icd_code(coding.get("code", "")),
                display=coding.get("display"),
            )

    first = coding_list[0]
    return GroundTruth(
        system=first.get("system", ""),
        code=first.get("code", ""),
        display=first.get("display"),
    )


def degrade(
    condition: dict[str, Any],
    mode: DegradationMode,
) -> DegradedRecord:
    """Apply a single degradation mode to a Condition; return the (input, ground_truth) record."""
    ground_truth = extract_ground_truth(condition)
    resource_id = condition.get("id", "")

    if mode == DegradationMode.D1_FULL:
        return DegradedRecord(
            resource_id=resource_id,
            mode=mode,
            input_fhir=copy.deepcopy(condition),
            input_text=None,
            ground_truth=ground_truth,
        )

    if mode == DegradationMode.D2_NO_CODE:
        return DegradedRecord(
            resource_id=resource_id,
            mode=mode,
            input_fhir=_strip_code_and_system(condition),
            input_text=None,
            ground_truth=ground_truth,
        )

    if mode == DegradationMode.D3_TEXT_ONLY:
        return DegradedRecord(
            resource_id=resource_id,
            mode=mode,
            input_fhir=None,
            input_text=_extract_text(condition),
            ground_truth=ground_truth,
        )

    if mode == DegradationMode.D4_ABBREVIATED:
        return DegradedRecord(
            resource_id=resource_id,
            mode=mode,
            input_fhir=None,
            input_text=_to_abbreviated_sentence(condition),
            ground_truth=ground_truth,
        )

    raise ValueError(f"Unknown degradation mode: {mode!r}")


def degrade_all_modes(condition: dict[str, Any]) -> list[DegradedRecord]:
    """Apply every degradation mode to a single Condition."""
    return [degrade(condition, mode) for mode in DegradationMode]


def _coding_list(condition: dict[str, Any]) -> list[dict[str, Any]]:
    code = condition.get("code") or {}
    return list(code.get("coding") or [])


def _strip_code_and_system(condition: dict[str, Any]) -> dict[str, Any]:
    """D2: remove `code` and `system` from each coding[] entry; keep display + text + structure."""
    out = copy.deepcopy(condition)
    code = out.get("code") or {}
    for coding in code.get("coding") or []:
        coding.pop("code", None)
        coding.pop("system", None)
    return out


def _extract_text(condition: dict[str, Any]) -> str:
    """D3: just the free-text label, falling back to the first display if text is absent."""
    code = condition.get("code") or {}
    text = code.get("text")
    if text:
        return text
    for coding in code.get("coding") or []:
        if coding.get("display"):
            return coding["display"]
    return ""


def _to_abbreviated_sentence(condition: dict[str, Any]) -> str:
    """D4: deterministic one-sentence summary with clinical abbreviations applied.

    The canonical display tokens are replaced with abbreviations (T2DM, HTN, CKD,
    etc.) per `abbreviations.yaml`. This strips the lexical overlap that string-
    matching baselines depend on while preserving the underlying diagnosis.
    """
    label = _extract_text(condition)
    if not label:
        return "Pt with unspecified condition."

    abbr = abbreviate(label)
    clinical_status = _first_coding_code(condition.get("clinicalStatus"))

    if clinical_status == "resolved":
        return f"Pt with hx of {abbr}, resolved."
    if clinical_status in {"recurrence", "relapse"}:
        return f"Pt with {clinical_status} of {abbr}."
    if clinical_status == "remission":
        return f"Pt's {abbr} in remission."
    if clinical_status == "inactive":
        return f"Pt with inactive {abbr}."
    return f"Pt with {abbr}."


def _first_coding_code(codeable_concept: dict[str, Any] | None) -> str | None:
    if not codeable_concept:
        return None
    for coding in codeable_concept.get("coding") or []:
        code = coding.get("code")
        if code:
            return code
    return None
