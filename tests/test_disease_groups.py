"""Tests for ACCESS Model scope: ValueSet parsing + scope/group queries."""

from __future__ import annotations

from phantom_codes.data.disease_groups import load


def test_load_is_cached() -> None:
    a = load()
    b = load()
    assert a is b


def test_explicit_ckm_codes_are_in_scope() -> None:
    scope = load()
    # I20.0 (Unstable angina) is an explicit CKM code.
    assert scope.is_in_scope("I20.0")
    assert scope.group_for("I20.0") == "ckm"
    # N18.31 (CKD stage 3a) is explicit.
    assert scope.group_for("N18.31") == "ckm"


def test_is_a_diabetes_expansion_with_excludes() -> None:
    """E11 has `is-a` inclusion in CKM, but the ValueSet excludes the parent + complications
    (E11.0..E11.6) explicitly while keeping uncomplicated forms (E11.9, E11.65) in scope.
    """
    scope = load()
    # Excluded: parent code itself.
    assert not scope.is_in_scope("E11")
    # Excluded: complication subtypes.
    assert not scope.is_in_scope("E11.0")
    assert not scope.is_in_scope("E11.32")
    # In scope: uncomplicated DM2 — the canonical CKM diagnosis.
    assert scope.group_for("E11.9") == "ckm"
    assert scope.group_for("E11.65") == "ckm"  # DM2 with hyperglycemia


def test_explicit_eckm_codes_are_in_scope() -> None:
    scope = load()
    assert scope.group_for("I10") == "eckm"  # essential hypertension
    assert scope.group_for("E66.9") == "eckm"  # obesity, unspecified
    assert scope.group_for("R73.03") == "eckm"  # prediabetes
    assert scope.group_for("E78.5") == "eckm"  # hyperlipidemia, unspecified


def test_out_of_scope_codes_return_none() -> None:
    scope = load()
    # Pneumonia, kidney failure (acute, not CKD3), heart failure — none are in ACCESS scope.
    assert scope.group_for("J18.9") is None
    assert scope.group_for("N17.9") is None
    assert scope.group_for("I50.9") is None
    assert scope.group_for("Z99.9") is None


def test_candidate_codes_are_non_empty_and_well_formed() -> None:
    scope = load()
    candidates = scope.candidate_codes()
    assert len(candidates) > 50  # both ValueSets have explicit codes
    groups = {c.group for c in candidates}
    assert groups == {"ckm", "eckm"}
    # All entries should have non-empty code; display may be empty but should be a string.
    for c in candidates:
        assert c.code
        assert isinstance(c.display, str)


def test_candidates_only_contain_explicitly_listed_codes() -> None:
    """Candidate codes are the explicit `concept` arrays from the ValueSets.

    `is-a` expansions like all E11.x children aren't enumerated here — that requires
    the full ICD-10-CM CodeSystem and we do that against MIMIC's observed codes later.
    """
    scope = load()
    candidate_codes = {c.code for c in scope.candidate_codes()}
    # I20.0 is explicit in the CKM include list.
    assert "I20.0" in candidate_codes
    # I10 is explicit in eCKM.
    assert "I10" in candidate_codes
    # E11.9 is in scope via is-a but is NOT in the candidate list because it isn't
    # one of the explicit concepts. (This is fine — for LLM/classifier vocab we'll
    # combine candidate_codes with MIMIC observed codes.)
    assert "E11.9" not in candidate_codes


def test_known_high_value_codes_resolved_correctly() -> None:
    """Spot-check several codes a clinician would expect to land in specific groups."""
    scope = load()
    expected = {
        "N18.30": "ckm",  # CKD stage 3 unspecified
        "N18.31": "ckm",  # CKD stage 3a
        "N18.32": "ckm",  # CKD stage 3b
        "I20.0": "ckm",  # unstable angina
        "I63.9": "ckm",  # cerebral infarction unspecified
        "I10": "eckm",  # essential hypertension
        "E78.5": "eckm",  # hyperlipidemia
        "E66.01": "eckm",  # morbid obesity
    }
    for code, group in expected.items():
        assert scope.group_for(code) == group, code


def test_candidates_for_codes_adds_observed_in_scope_codes() -> None:
    """E11.9 is in scope via `is-a E11` but not in the explicit ValueSet — adding via
    observed codes should include it with the right group label."""
    scope = load()
    observed = [
        ("E11.9", "Type 2 diabetes mellitus without complications"),
        ("E11.65", "Type 2 diabetes mellitus with hyperglycemia"),
        ("J18.9", "Pneumonia"),  # out of scope — should be skipped
    ]
    candidates = scope.candidates_for_codes(observed)
    codes = {c.code for c in candidates}
    assert "E11.9" in codes
    assert "E11.65" in codes
    # Out-of-scope code should NOT appear.
    assert "J18.9" not in codes
    # Original explicit codes are still there.
    assert "I10" in codes
    # Group labels correct on added observed codes.
    e119 = next(c for c in candidates if c.code == "E11.9")
    assert e119.group == "ckm"
    assert e119.display == "Type 2 diabetes mellitus without complications"


def test_candidates_for_codes_dedupes_against_explicit_list() -> None:
    """If an observed code is already explicit, don't duplicate it."""
    scope = load()
    observed = [("I10", "Essential (primary) hypertension")]  # already explicit
    candidates = scope.candidates_for_codes(observed)
    i10_count = sum(1 for c in candidates if c.code == "I10")
    assert i10_count == 1
