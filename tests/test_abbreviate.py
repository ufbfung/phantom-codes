"""Tests for the abbreviation substitution used by D4_abbreviated."""

from __future__ import annotations

from phantom_codes.data.abbreviate import abbreviate


def test_abbreviates_canonical_diabetes_phrasing() -> None:
    assert abbreviate("Type 2 diabetes mellitus without complications") == "T2DM unc"


def test_abbreviates_hypertension() -> None:
    assert abbreviate("Essential (primary) hypertension") == "essential HTN"


def test_abbreviates_ckd_with_stage() -> None:
    assert abbreviate("Chronic kidney disease, stage 3a") == "CKD IIIa"
    assert abbreviate("Chronic kidney disease, stage 3b") == "CKD IIIb"


def test_abbreviates_hyperlipidemia() -> None:
    assert abbreviate("Hyperlipidemia, unspecified") == "HLD NOS"


def test_abbreviates_obesity() -> None:
    assert abbreviate("Obesity, unspecified") == "obesity NOS"


def test_longest_pattern_wins() -> None:
    """If 'Type 2 diabetes mellitus' is in the rules AND 'diabetes mellitus' is too,
    the long one must match first or we'd produce 'Type 2 DM' instead of 'T2DM'."""
    out = abbreviate("Type 2 diabetes mellitus without complications")
    assert "Type 2" not in out  # the longer pattern stripped it
    assert out == "T2DM unc"


def test_case_insensitive_matching() -> None:
    assert abbreviate("TYPE 2 DIABETES MELLITUS") == "T2DM"
    assert abbreviate("type 2 diabetes mellitus") == "T2DM"


def test_idempotent_no_op_when_no_patterns_match() -> None:
    # No abbreviation in the table for "rare condition unspec".
    text = "Some rare disease entity"
    assert abbreviate(text) == text


def test_empty_string_passes_through() -> None:
    assert abbreviate("") == ""


def test_collapses_extra_whitespace_after_substitution() -> None:
    # If a substitution leaves double spaces, they should collapse.
    out = abbreviate("Type 2 diabetes mellitus  without complications")
    assert "  " not in out
