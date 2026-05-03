"""Validation tests for the curated SNOMED → ICD-10-CM map.

The map at `data/synthea/snomed_to_icd10cm.json` is in Synthea's
expected `code_map.icd10-cm` format:

    {
      "<source-snomed-code>": [
        {
          "code": "<destination-icd-code>",
          "description": "<destination-display>",
          ...curation metadata fields ignored by Synthea...
        }
      ]
    }

Synthea consumes only the `code` and `description` (and optional
`weight`) keys per array entry. We add extra fields for curation
transparency: `snomed_display`, `access_group`, `confidence`,
`source`, `notes`, `n_observed_pilot`. Synthea silently ignores them.

The optional top-level `_README` key is documentation; tests skip it.

These tests run quickly (no I/O beyond the map file itself) and
protect the curated map from regressions.

If `data/synthea/snomed_to_icd10cm.json` does not yet exist, every
test in this file is *skipped* (rather than failing). The map is
created during WS1 of the Synthea benchmark workstream — see the
project plan's "Curation handoff protocol" section. Once the map is
committed, these tests become required.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

MAP_PATH = Path(__file__).resolve().parents[1] / "data" / "synthea" / "snomed_to_icd10cm.json"

ICD10CM_PATTERN = re.compile(r"^[A-Z]\d{2}(\.\d{1,4})?$")
SNOMED_PATTERN = re.compile(r"^\d+$")
VALID_CONFIDENCE = {"high", "needs_review", "verified"}
VALID_ACCESS_GROUPS = {"ckm", "eckm", "OUT_OF_SCOPE"}
REQUIRED_FIELDS = {"code", "description", "snomed_display", "access_group",
                   "confidence", "source"}


@pytest.fixture(scope="module")
def map_payload() -> dict:
    if not MAP_PATH.exists():
        pytest.skip(
            f"{MAP_PATH} not yet curated. Run the Synthea WS1 inventory + "
            "curation steps to produce it; tests in this file then become "
            "required."
        )
    return json.loads(MAP_PATH.read_text())


@pytest.fixture(scope="module")
def entries(map_payload: dict) -> dict:
    """Filter top-level dict down to actual SNOMED-keyed entries
    (skip `_README` / `_metadata` / any other underscore-prefixed
    documentation keys)."""
    return {k: v for k, v in map_payload.items() if not k.startswith("_")}


def test_map_has_at_least_one_entry(entries: dict) -> None:
    assert len(entries) > 0, "map must have at least one SNOMED→ICD entry"


def test_keys_are_snomed_concept_ids(entries: dict) -> None:
    """Top-level keys must be all-numeric SNOMED concept IDs."""
    for key in entries:
        assert SNOMED_PATTERN.match(key), (
            f"key {key!r} doesn't look like a SNOMED concept ID "
            "(should be all digits)"
        )


def test_each_entry_is_array_of_dicts(entries: dict) -> None:
    """Synthea's format requires array of {code, description} per source."""
    for snomed_id, value in entries.items():
        assert isinstance(value, list), (
            f"entry {snomed_id} value must be a list (got {type(value).__name__})"
        )
        assert len(value) >= 1, f"entry {snomed_id} has empty array"
        for item in value:
            assert isinstance(item, dict), (
                f"entry {snomed_id} array contains non-dict item"
            )


def test_every_entry_has_required_fields(entries: dict) -> None:
    for snomed_id, value in entries.items():
        for item in value:
            missing = REQUIRED_FIELDS - set(item.keys())
            assert not missing, f"entry {snomed_id} missing fields: {missing}"


def test_no_remaining_todos(entries: dict) -> None:
    """Curation step must replace all TODO placeholders with real codes."""
    for snomed_id, value in entries.items():
        for item in value:
            assert item["code"] != "TODO", (
                f"entry {snomed_id} still has TODO code; "
                "complete the curation step before running cohort generation"
            )
            assert item["description"] != "TODO", (
                f"entry {snomed_id} still has TODO description"
            )


def test_icd10cm_codes_are_well_formed(entries: dict) -> None:
    """Codes match the canonical ICD-10-CM regex."""
    for snomed_id, value in entries.items():
        for item in value:
            code = item["code"]
            assert ICD10CM_PATTERN.match(code), (
                f"entry {snomed_id} has malformed ICD-10-CM code {code!r} "
                f"(expected pattern like E11.9 or I10)"
            )


def test_confidence_values_are_recognized(entries: dict) -> None:
    for snomed_id, value in entries.items():
        for item in value:
            assert item["confidence"] in VALID_CONFIDENCE, (
                f"entry {snomed_id} has unknown confidence "
                f"{item['confidence']!r}; valid: {sorted(VALID_CONFIDENCE)}"
            )


def test_access_groups_are_recognized(entries: dict) -> None:
    for snomed_id, value in entries.items():
        for item in value:
            assert item["access_group"] in VALID_ACCESS_GROUPS, (
                f"entry {snomed_id} has unknown access_group "
                f"{item['access_group']!r}; valid: {sorted(VALID_ACCESS_GROUPS)}"
            )


def test_in_scope_entries_are_actually_in_access_scope(entries: dict) -> None:
    """Every entry tagged ckm/eckm must have an ICD code that's actually in scope.

    OUT_OF_SCOPE entries are exempt — they're documented as such and
    will be filtered out downstream anyway.
    """
    from phantom_codes.data.disease_groups import load as load_scope

    scope = load_scope()
    misclassified = []
    for snomed_id, value in entries.items():
        for item in value:
            if item["access_group"] == "OUT_OF_SCOPE":
                continue
            if not scope.is_in_scope(item["code"]):
                misclassified.append(
                    (snomed_id, item["code"], item["access_group"])
                )

    assert not misclassified, (
        "These entries are tagged in-scope but their ICD code fails is_in_scope():\n"
        + "\n".join(f"  - {sid}: {icd} (group: {grp})" for sid, icd, grp in misclassified)
    )


def test_access_group_matches_canonical_group(entries: dict) -> None:
    """For in-scope entries, the declared `access_group` must match the
    canonical group_for() result. OUT_OF_SCOPE entries skipped."""
    from phantom_codes.data.disease_groups import load as load_scope

    scope = load_scope()
    mismatches = []
    for snomed_id, value in entries.items():
        for item in value:
            if item["access_group"] == "OUT_OF_SCOPE":
                continue
            canonical = scope.group_for(item["code"])
            if canonical != item["access_group"]:
                mismatches.append(
                    (snomed_id, item["code"], item["access_group"], canonical)
                )

    assert not mismatches, (
        "These entries have access_group that doesn't match the canonical group:\n"
        + "\n".join(
            f"  - {sid}: {icd} declared as {dec!r}, canonical is {can!r}"
            for sid, icd, dec, can in mismatches
        )
    )


def test_no_remaining_needs_review_after_signoff(entries: dict) -> None:
    """Once the curation review is complete, no entries should remain
    flagged needs_review. Entries with confidence='needs_review' are
    the ones the curator was asked to inspect / confirm / delete.

    XFAIL during curation handoff (some entries pending review); becomes
    expected pass once Brian signals 'mapping review complete' and any
    remaining needs_review entries are either resolved to high/verified
    or removed from the map.
    """
    pending = [
        snomed_id
        for snomed_id, value in entries.items()
        for item in value
        if item["confidence"] == "needs_review"
    ]
    if pending:
        pytest.xfail(
            f"{len(pending)} entries still flagged needs_review: "
            f"{pending[:5]}{'...' if len(pending) > 5 else ''}"
        )
