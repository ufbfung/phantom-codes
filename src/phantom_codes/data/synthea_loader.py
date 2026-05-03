"""Synthea-specific FHIR Bundle parsing for the inference benchmark.

Synthea generates one FHIR Bundle JSON file per synthetic patient, each
containing a Patient resource and the patient's full lifetime history
(Conditions, Observations, Procedures, MedicationRequests, etc.). For
the Phantom Codes benchmark we only care about the Conditions, and we
need them as individual resources matching the schema our existing
`prepare` pipeline expects.

This module is the Synthea-side equivalent of
`phantom_codes.data.fhir_loader` (which targets MIMIC's per-resource
ndjson.gz files).

Three responsibilities:
    1. Walk a directory of Synthea-generated Bundle files.
    2. Yield individual Condition resources from each Bundle.
    3. De-duplicate: Synthea generates multiple Condition encounters
       for the same diagnosis over a patient's lifetime
       (e.g., diabetes diagnosed in 2018 might be referenced in many
       subsequent encounter Bundles). For our benchmark we want
       unique (patient, diagnosis) pairs, not every encounter.

Compliance note: Synthea data is open (Apache 2.0, synthetic patients
only — no real PHI). Safe to handle, share, and inspect freely. This
module never touches MIMIC content.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

# FHIR coding-system URIs used by Synthea-generated Conditions.
SNOMED_SYSTEM = "http://snomed.info/sct"
ICD10CM_SYSTEM = "http://hl7.org/fhir/sid/icd-10-cm"


def iter_synthea_bundles(directory: str | Path) -> Iterator[dict[str, Any]]:
    """Yield FHIR Bundle JSON objects from a directory of Synthea outputs.

    Synthea's FHIR exporter writes one .json file per patient (the
    Patient's full lifetime history wrapped in a Bundle resource).
    Recursively walks the directory; tolerates non-Bundle files by
    skipping them with a warning.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Synthea bundle directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {directory}")

    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if payload.get("resourceType") == "Bundle":
            yield payload


def extract_conditions(bundle: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield Condition resources from a Synthea Bundle.

    Synthea Bundles use `entry[].resource` to hold each resource. We
    yield only those whose `resourceType == "Condition"`.

    We *preserve* the original coding[] arrays as Synthea emitted them
    — both SNOMED (always present) and ICD-10-CM (present if Synthea was
    configured with `--exporter.code_map.icd10-cm`). Downstream
    `extract_ground_truth()` in degrade.py picks the ICD-10-CM coding
    via the existing `_SYSTEM_ALIASES` lookup.
    """
    for entry in bundle.get("entry") or []:
        resource = entry.get("resource") or {}
        if resource.get("resourceType") == "Condition":
            yield resource


def _condition_dedup_key(condition: dict[str, Any]) -> tuple[str, str] | None:
    """Build a (patient_ref, icd10cm_code) key for de-duplication.

    Returns None if either piece is missing — caller treats those as
    non-deduplicable (kept distinct) rather than collapsed.
    """
    subject_ref = (condition.get("subject") or {}).get("reference") or ""
    code_obj = condition.get("code") or {}
    icd_code = ""
    for coding in code_obj.get("coding") or []:
        if coding.get("system") == ICD10CM_SYSTEM:
            icd_code = str(coding.get("code") or "")
            break
    if not subject_ref or not icd_code:
        return None
    return (subject_ref, icd_code)


def dedupe_per_resource(
    conditions: Iterable[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Collapse multiple Condition encounters for the same (patient, diagnosis).

    Synthea generates one Condition resource per encounter at which a
    diagnosis was active; for chronic conditions this means many
    duplicate Conditions per patient. For our benchmark, the
    *diagnosis* is the unit of measurement, not the encounter.

    Keeps the FIRST occurrence (which is typically the onset Condition,
    closest to the original diagnosis event in the patient's history).
    Conditions whose dedup key is None (missing patient or ICD code)
    are kept untouched — they'll either get filtered downstream by
    `filter_in_scope()` or surface as edge cases.
    """
    seen: set[tuple[str, str]] = set()
    for condition in conditions:
        key = _condition_dedup_key(condition)
        if key is None:
            yield condition
            continue
        if key in seen:
            continue
        seen.add(key)
        yield condition


def iter_conditions_from_directory(
    directory: str | Path,
) -> Iterator[dict[str, Any]]:
    """End-to-end iterator: directory of Bundles → deduped Conditions.

    Convenience wrapper composing `iter_synthea_bundles`,
    `extract_conditions`, and `dedupe_per_resource`. Drop-in compatible
    with the `iter_conditions` signature from `fhir_loader.py`, so the
    same downstream `prepare_from_iter` logic in prepare.py consumes
    it without per-source branching.
    """
    bundles = iter_synthea_bundles(directory)

    def all_conditions() -> Iterator[dict[str, Any]]:
        for bundle in bundles:
            yield from extract_conditions(bundle)

    yield from dedupe_per_resource(all_conditions())
