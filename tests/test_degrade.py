"""Unit tests for the degradation pipeline."""

from typing import Any

import pytest

from phantom_codes.data.degrade import (
    ICD9_SYSTEM,
    ICD10_SYSTEM,
    DegradationMode,
    GroundTruth,
    degrade,
    degrade_all_modes,
    extract_ground_truth,
)


def test_extract_ground_truth_icd10(fixture_by_id: dict[str, dict[str, Any]]) -> None:
    gt = extract_ground_truth(fixture_by_id["fixture-001"])
    assert gt == GroundTruth(
        system=ICD10_SYSTEM,
        code="E11.9",
        display="Type 2 diabetes mellitus without complications",
    )


def test_extract_ground_truth_icd9(fixture_by_id: dict[str, dict[str, Any]]) -> None:
    gt = extract_ground_truth(fixture_by_id["fixture-006-icd9-out-of-scope"])
    assert gt.system == ICD9_SYSTEM
    assert gt.code == "250.00"


def test_extract_ground_truth_missing_coding_raises() -> None:
    bad: dict[str, Any] = {"resourceType": "Condition", "id": "x", "code": {"coding": []}}
    with pytest.raises(ValueError, match="no code.coding"):
        extract_ground_truth(bad)


def test_extract_ground_truth_normalizes_mimic_icd10_system_and_dots_code() -> None:
    """MIMIC-FHIR uses its own CodeSystem URI and undotted codes; both normalize."""
    cond = {
        "resourceType": "Condition",
        "id": "mimic-x",
        "code": {
            "coding": [
                {
                    "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-diagnosis-icd10",
                    "code": "E1110",
                    "display": "Type 2 diabetes mellitus with hyperosmolarity",
                }
            ]
        },
    }
    gt = extract_ground_truth(cond)
    assert gt.system == ICD10_SYSTEM  # normalized away from MIMIC URI
    assert gt.code == "E11.10"        # dot inserted after position 3


def test_extract_ground_truth_normalizes_mimic_icd9_system() -> None:
    cond = {
        "resourceType": "Condition",
        "id": "mimic-y",
        "code": {
            "coding": [
                {
                    "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-diagnosis-icd9",
                    "code": "25000",
                    "display": "Diabetes mellitus without complication",
                }
            ]
        },
    }
    gt = extract_ground_truth(cond)
    assert gt.system == ICD9_SYSTEM
    assert gt.code == "250.00"


def test_extract_ground_truth_short_codes_unchanged() -> None:
    """3-character chapter-level codes (E11, I10) stay undotted."""
    cond = {
        "resourceType": "Condition",
        "id": "x",
        "code": {
            "coding": [
                {
                    "system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-diagnosis-icd10",
                    "code": "I10",
                }
            ]
        },
    }
    assert extract_ground_truth(cond).code == "I10"


def test_extract_ground_truth_already_dotted_codes_pass_through() -> None:
    """Idempotent: Synthea-style dotted codes don't get a second dot."""
    cond = {
        "resourceType": "Condition",
        "id": "x",
        "code": {
            "coding": [{"system": ICD10_SYSTEM, "code": "E11.9", "display": "T2DM"}]
        },
    }
    assert extract_ground_truth(cond).code == "E11.9"


def test_d1_full_preserves_resource(fixture_by_id: dict[str, dict[str, Any]]) -> None:
    cond = fixture_by_id["fixture-001"]
    rec = degrade(cond, DegradationMode.D1_FULL)
    assert rec.input_fhir == cond
    assert rec.input_text is None
    assert rec.ground_truth.code == "E11.9"


def test_d1_full_does_not_mutate_input(fixture_by_id: dict[str, dict[str, Any]]) -> None:
    cond = fixture_by_id["fixture-001"]
    snapshot = {**cond, "code": {**cond["code"]}}
    snapshot["code"]["coding"] = [{**c} for c in cond["code"]["coding"]]
    rec = degrade(cond, DegradationMode.D1_FULL)
    assert rec.input_fhir is not cond  # deep-copied
    rec.input_fhir["code"]["coding"][0]["code"] = "MUTATED"
    assert cond["code"]["coding"][0]["code"] == "E11.9"


def test_d2_no_code_strips_code_and_system_keeps_display(
    fixture_by_id: dict[str, dict[str, Any]],
) -> None:
    cond = fixture_by_id["fixture-001"]
    rec = degrade(cond, DegradationMode.D2_NO_CODE)
    coding = rec.input_fhir["code"]["coding"][0]
    assert "code" not in coding
    assert "system" not in coding
    assert coding["display"] == "Type 2 diabetes mellitus without complications"
    assert rec.input_fhir["code"]["text"] == "Type 2 diabetes mellitus without complications"
    assert rec.input_fhir["clinicalStatus"]["coding"][0]["code"] == "active"
    assert rec.ground_truth.code == "E11.9"


def test_d2_does_not_mutate_input(fixture_by_id: dict[str, dict[str, Any]]) -> None:
    cond = fixture_by_id["fixture-001"]
    degrade(cond, DegradationMode.D2_NO_CODE)
    assert cond["code"]["coding"][0]["code"] == "E11.9"
    assert cond["code"]["coding"][0]["system"] == ICD10_SYSTEM


def test_d3_text_only_uses_text_field(fixture_by_id: dict[str, dict[str, Any]]) -> None:
    cond = fixture_by_id["fixture-002"]
    rec = degrade(cond, DegradationMode.D3_TEXT_ONLY)
    assert rec.input_fhir is None
    assert rec.input_text == "Essential (primary) hypertension"
    assert rec.ground_truth.code == "I10"


def test_d3_text_only_falls_back_to_display(
    fixture_by_id: dict[str, dict[str, Any]],
) -> None:
    """fixture-004 has no code.text — should fall back to coding[0].display."""
    cond = fixture_by_id["fixture-004-out-of-scope"]
    assert "text" not in cond["code"]
    rec = degrade(cond, DegradationMode.D3_TEXT_ONLY)
    assert rec.input_text == "Acute kidney failure, unspecified"


def test_d4_abbreviated_strips_canonical_diabetes_tokens(
    fixture_by_id: dict[str, dict[str, Any]],
) -> None:
    """E11.9: 'Type 2 diabetes mellitus without complications' → 'T2DM unc'."""
    cond = fixture_by_id["fixture-001"]
    rec = degrade(cond, DegradationMode.D4_ABBREVIATED)
    assert rec.input_fhir is None
    assert rec.input_text == "Pt with T2DM unc."
    # Verify the canonical tokens that string-matching baselines depend on are gone.
    assert "Type 2 diabetes mellitus" not in rec.input_text
    assert "without complications" not in rec.input_text


def test_d4_abbreviated_resolved_status(fixture_by_id: dict[str, dict[str, Any]]) -> None:
    cond = fixture_by_id["fixture-005-out-of-scope"]
    rec = degrade(cond, DegradationMode.D4_ABBREVIATED)
    assert "hx of" in rec.input_text
    assert "resolved" in rec.input_text


def test_d4_abbreviated_no_clinical_status(
    fixture_by_id: dict[str, dict[str, Any]],
) -> None:
    cond = fixture_by_id["fixture-004-out-of-scope"]
    assert "clinicalStatus" not in cond
    rec = degrade(cond, DegradationMode.D4_ABBREVIATED)
    # 'unspecified' → 'NOS' applies; rest of label has no abbrev.
    assert rec.input_text == "Pt with Acute kidney failure, NOS."


def test_d4_abbreviated_is_deterministic(fixture_by_id: dict[str, dict[str, Any]]) -> None:
    cond = fixture_by_id["fixture-001"]
    runs = {degrade(cond, DegradationMode.D4_ABBREVIATED).input_text for _ in range(5)}
    assert len(runs) == 1


def test_d4_abbreviated_handles_hypertension(
    fixture_by_id: dict[str, dict[str, Any]],
) -> None:
    """I10: 'Essential (primary) hypertension' → 'essential HTN'."""
    cond = fixture_by_id["fixture-002"]
    rec = degrade(cond, DegradationMode.D4_ABBREVIATED)
    assert rec.input_text == "Pt with essential HTN."


def test_d4_abbreviated_handles_ckd(
    fixture_by_id: dict[str, dict[str, Any]],
) -> None:
    """N18.31: 'Chronic kidney disease, stage 3a' → 'CKD IIIa'."""
    cond = fixture_by_id["fixture-008-ckm-ckd3a"]
    rec = degrade(cond, DegradationMode.D4_ABBREVIATED)
    assert rec.input_text == "Pt with CKD IIIa."


def test_degrade_all_modes_returns_one_per_mode(
    fixture_by_id: dict[str, dict[str, Any]],
) -> None:
    cond = fixture_by_id["fixture-001"]
    records = degrade_all_modes(cond)
    assert [r.mode for r in records] == list(DegradationMode)
    for r in records:
        assert r.ground_truth.code == "E11.9"


def test_all_fixtures_round_trip_all_modes(
    fixture_conditions: list[dict[str, Any]],
) -> None:
    """Every fixture should produce 4 records without raising and preserve the ground-truth code."""
    for cond in fixture_conditions:
        records = degrade_all_modes(cond)
        codes = {r.ground_truth.code for r in records}
        assert len(codes) == 1
        assert codes.pop() != ""
