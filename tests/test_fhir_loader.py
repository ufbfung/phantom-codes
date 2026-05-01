"""Loader tests — local files only; GCS path verified via mocking, no live calls."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from phantom_codes.data.fhir_loader import (
    count_resources,
    iter_conditions,
    iter_fhir_resources,
)

FIXTURE = Path(__file__).parent / "fixtures" / "conditions.ndjson"


FIXTURE_COUNT = 10  # see tests/fixtures/conditions.ndjson


def test_iter_fhir_resources_local_ndjson() -> None:
    resources = list(iter_fhir_resources(FIXTURE))
    assert len(resources) == FIXTURE_COUNT
    assert all(r["resourceType"] == "Condition" for r in resources)


def test_iter_fhir_resources_handles_gz(tmp_path: Path) -> None:
    """Gzip path must be exercised — most MIMIC files are .ndjson.gz."""
    gz_path = tmp_path / "conditions.ndjson.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        for line in FIXTURE.read_text().splitlines():
            f.write(line + "\n")

    resources = list(iter_fhir_resources(gz_path))
    assert len(resources) == FIXTURE_COUNT


def test_iter_fhir_resources_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "with_blanks.ndjson"
    raw = FIXTURE.read_text()
    p.write_text(raw + "\n\n\n")
    assert len(list(iter_fhir_resources(p))) == FIXTURE_COUNT


def test_iter_conditions_filters_resource_type(tmp_path: Path) -> None:
    p = tmp_path / "mixed.ndjson"
    with p.open("w") as f:
        for line in FIXTURE.read_text().splitlines():
            f.write(line + "\n")
        f.write(json.dumps({"resourceType": "Patient", "id": "p-1"}) + "\n")
        f.write(json.dumps({"resourceType": "Encounter", "id": "e-1"}) + "\n")

    conditions = list(iter_conditions(p))
    assert len(conditions) == FIXTURE_COUNT
    assert all(c["resourceType"] == "Condition" for c in conditions)


def test_count_resources() -> None:
    assert count_resources(FIXTURE) == FIXTURE_COUNT


def test_iter_fhir_resources_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_fhir_resources(tmp_path / "does-not-exist.ndjson"))
