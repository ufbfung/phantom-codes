"""Shared test fixtures."""

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_ndjson(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="session")
def fixture_conditions() -> list[dict[str, Any]]:
    """All hand-curated synthetic Condition resources."""
    return _load_ndjson(FIXTURE_DIR / "conditions.ndjson")


@pytest.fixture(scope="session")
def fixture_by_id(fixture_conditions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c["id"]: c for c in fixture_conditions}
