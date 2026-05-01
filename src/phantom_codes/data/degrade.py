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

# FHIR system URIs for ICD coding
ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10-cm"
ICD9_SYSTEM = "http://hl7.org/fhir/sid/icd-9-cm"
ICD_SYSTEMS = frozenset({ICD10_SYSTEM, ICD9_SYSTEM})


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

    Picks the first coding[] entry whose system is ICD-10 or ICD-9. Falls back to
    the first coding[] entry if no ICD coding is found (shouldn't happen for
    MimicCondition, but keeps the function total).
    """
    coding_list = _coding_list(condition)
    if not coding_list:
        raise ValueError(f"Condition {condition.get('id')!r} has no code.coding[]")

    for coding in coding_list:
        if coding.get("system") in ICD_SYSTEMS:
            return GroundTruth(
                system=coding["system"],
                code=coding["code"],
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
