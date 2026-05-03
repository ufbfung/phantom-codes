"""Unit tests for Synthea Bundle parsing.

Uses hand-built tiny Synthea-style fixtures (NOT real Synthea outputs
— just minimal JSON shaped like Synthea Bundles). No MIMIC content;
no real FHIR data; no I/O beyond pytest's tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from phantom_codes.data.synthea_loader import (
    ICD10CM_SYSTEM,
    SNOMED_SYSTEM,
    dedupe_per_resource,
    extract_conditions,
    iter_conditions_from_directory,
    iter_synthea_bundles,
)

# ─────────────────────────────────────────────────────────────────────────
# Hand-built fixtures (Synthea-shaped but not real Synthea outputs)
# ─────────────────────────────────────────────────────────────────────────


def _make_condition(
    *,
    resource_id: str,
    patient_ref: str,
    snomed_code: str,
    snomed_display: str,
    icd_code: str | None = None,
    icd_display: str | None = None,
) -> dict[str, Any]:
    """Build a Condition resource shaped like Synthea's FHIR output."""
    coding = [{"system": SNOMED_SYSTEM, "code": snomed_code, "display": snomed_display}]
    if icd_code:
        coding.append(
            {"system": ICD10CM_SYSTEM, "code": icd_code, "display": icd_display or icd_code}
        )
    return {
        "resourceType": "Condition",
        "id": resource_id,
        "subject": {"reference": patient_ref},
        "code": {
            "coding": coding,
            "text": (icd_display or snomed_display),
        },
    }


def _make_bundle(*resources: dict[str, Any]) -> dict[str, Any]:
    """Wrap resources in a Synthea-style transaction Bundle."""
    return {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [{"resource": r} for r in resources],
    }


def _make_patient(patient_id: str) -> dict[str, Any]:
    return {"resourceType": "Patient", "id": patient_id}


# ─────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────


def test_iter_synthea_bundles_walks_directory(tmp_path: Path) -> None:
    """Loader yields one Bundle per .json file in the directory."""
    b1 = _make_bundle(_make_patient("p1"))
    b2 = _make_bundle(_make_patient("p2"))
    (tmp_path / "patient1.json").write_text(json.dumps(b1))
    (tmp_path / "patient2.json").write_text(json.dumps(b2))
    # Drop a non-Bundle file in the same directory; loader should skip it.
    (tmp_path / "metadata.json").write_text(json.dumps({"resourceType": "Other"}))

    bundles = list(iter_synthea_bundles(tmp_path))
    assert len(bundles) == 2
    assert all(b["resourceType"] == "Bundle" for b in bundles)


def test_iter_synthea_bundles_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_synthea_bundles(tmp_path / "does-not-exist"))


def test_iter_synthea_bundles_skips_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "good.json").write_text(json.dumps(_make_bundle(_make_patient("p1"))))
    (tmp_path / "bad.json").write_text("{this is not valid json")

    bundles = list(iter_synthea_bundles(tmp_path))
    assert len(bundles) == 1


def test_extract_conditions_from_bundle() -> None:
    """Bundle with mix of resources yields only Condition entries."""
    bundle = _make_bundle(
        _make_patient("p1"),
        _make_condition(
            resource_id="cond-1",
            patient_ref="Patient/p1",
            snomed_code="44054006",
            snomed_display="Type 2 DM",
        ),
        {"resourceType": "Observation", "id": "obs-1"},
        _make_condition(
            resource_id="cond-2",
            patient_ref="Patient/p1",
            snomed_code="59621000",
            snomed_display="Essential HTN",
        ),
    )

    conditions = list(extract_conditions(bundle))
    assert len(conditions) == 2
    assert all(c["resourceType"] == "Condition" for c in conditions)
    assert {c["id"] for c in conditions} == {"cond-1", "cond-2"}


def test_extract_conditions_preserves_dual_codings() -> None:
    """When ICD-10-CM is present alongside SNOMED, both codings carry through."""
    cond = _make_condition(
        resource_id="cond-1",
        patient_ref="Patient/p1",
        snomed_code="44054006",
        snomed_display="Type 2 DM",
        icd_code="E11.9",
        icd_display="Type 2 diabetes mellitus without complications",
    )
    bundle = _make_bundle(cond)
    extracted = next(extract_conditions(bundle))

    coding_systems = {c["system"] for c in extracted["code"]["coding"]}
    assert SNOMED_SYSTEM in coding_systems
    assert ICD10CM_SYSTEM in coding_systems


def test_dedupe_collapses_repeat_diagnoses() -> None:
    """Multiple Conditions with same (patient, ICD code) collapse to one."""
    cond_a = _make_condition(
        resource_id="cond-a",
        patient_ref="Patient/p1",
        snomed_code="44054006",
        snomed_display="Type 2 DM",
        icd_code="E11.9",
    )
    cond_b = _make_condition(
        resource_id="cond-b",
        patient_ref="Patient/p1",
        snomed_code="44054006",
        snomed_display="Type 2 DM (later encounter)",
        icd_code="E11.9",
    )
    cond_c = _make_condition(
        resource_id="cond-c",
        patient_ref="Patient/p2",
        snomed_code="44054006",
        snomed_display="Type 2 DM",
        icd_code="E11.9",
    )

    deduped = list(dedupe_per_resource([cond_a, cond_b, cond_c]))
    # cond_a survives, cond_b is dropped (same patient + same ICD), cond_c kept (different patient)
    assert len(deduped) == 2
    assert deduped[0]["id"] == "cond-a"
    assert deduped[1]["id"] == "cond-c"


def test_dedupe_keeps_distinct_diagnoses_for_same_patient() -> None:
    """Same patient with two different diagnoses keeps both."""
    diabetes = _make_condition(
        resource_id="cond-1",
        patient_ref="Patient/p1",
        snomed_code="44054006",
        snomed_display="Type 2 DM",
        icd_code="E11.9",
    )
    htn = _make_condition(
        resource_id="cond-2",
        patient_ref="Patient/p1",
        snomed_code="59621000",
        snomed_display="Essential HTN",
        icd_code="I10",
    )

    deduped = list(dedupe_per_resource([diabetes, htn]))
    assert len(deduped) == 2


def test_dedupe_passes_through_when_key_missing() -> None:
    """Conditions without subject or ICD code are kept (can't dedup)."""
    no_subject = {
        "resourceType": "Condition",
        "id": "cond-1",
        "code": {"coding": [{"system": ICD10CM_SYSTEM, "code": "E11.9"}]},
    }
    no_icd = _make_condition(
        resource_id="cond-2",
        patient_ref="Patient/p1",
        snomed_code="44054006",
        snomed_display="Type 2 DM",
        # no icd_code
    )

    deduped = list(dedupe_per_resource([no_subject, no_icd]))
    assert len(deduped) == 2


def test_iter_conditions_from_directory_e2e(tmp_path: Path) -> None:
    """End-to-end: directory of bundles → deduped conditions ready for prepare."""
    bundle1 = _make_bundle(
        _make_patient("p1"),
        _make_condition(
            resource_id="cond-1",
            patient_ref="Patient/p1",
            snomed_code="44054006",
            snomed_display="Type 2 DM",
            icd_code="E11.9",
        ),
        _make_condition(
            resource_id="cond-1-dup",
            patient_ref="Patient/p1",
            snomed_code="44054006",
            snomed_display="Type 2 DM",
            icd_code="E11.9",
        ),
    )
    bundle2 = _make_bundle(
        _make_patient("p2"),
        _make_condition(
            resource_id="cond-2",
            patient_ref="Patient/p2",
            snomed_code="59621000",
            snomed_display="Essential HTN",
            icd_code="I10",
        ),
    )
    (tmp_path / "p1.json").write_text(json.dumps(bundle1))
    (tmp_path / "p2.json").write_text(json.dumps(bundle2))

    conditions = list(iter_conditions_from_directory(tmp_path))
    # 3 raw → 2 after dedup (cond-1-dup drops)
    assert len(conditions) == 2
    icd_codes = [
        c["code"]["coding"][1]["code"]
        for c in conditions
    ]
    assert sorted(icd_codes) == ["E11.9", "I10"]
