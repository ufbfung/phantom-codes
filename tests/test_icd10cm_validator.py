"""Tests for the bundled ICD-10-CM validator."""

from __future__ import annotations

from phantom_codes.data.degrade import ICD10_SYSTEM
from phantom_codes.data.icd10cm.validator import load


def test_load_is_cached() -> None:
    a = load()
    b = load()
    assert a is b


def test_size_matches_cms_2026_snapshot() -> None:
    """CMS FY2026 ICD-10-CM has 74,719 codes after dot reinsertion."""
    v = load()
    assert v.n_codes == 74719


def test_known_real_codes_exist() -> None:
    v = load()
    for code in ["E11.9", "I10", "N18.31", "N18.32", "E78.5", "E66.9", "I25.10", "J18.9"]:
        assert v.exists(ICD10_SYSTEM, code), code


def test_made_up_codes_do_not_exist() -> None:
    v = load()
    # Plausible-looking but not real ICD-10-CM codes.
    assert not v.exists(ICD10_SYSTEM, "E11.99")  # E11.9 exists; .99 doesn't
    assert not v.exists(ICD10_SYSTEM, "Z99.99")  # made up
    assert not v.exists(ICD10_SYSTEM, "")


def test_wrong_system_returns_false() -> None:
    v = load()
    # SNOMED CT 44054006 = Type 2 diabetes mellitus — real SNOMED, but not ICD-10.
    assert not v.exists("http://snomed.info/sct", "44054006")
    # ICD-9 — out of scope for the validator.
    assert not v.exists("http://hl7.org/fhir/sid/icd-9-cm", "250.00")
