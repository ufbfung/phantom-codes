"""Apply manually-curated abbreviation substitutions to clinical text.

Used by the D4_abbreviated degradation mode. The goal is to STRIP the canonical
ICD-10-CM display tokens that string-matching baselines depend on, while PRESERVING
the underlying clinical meaning. So "Type 2 diabetes mellitus without complications"
becomes "T2DM unc" — same diagnosis, different surface form.

Mappings live in `abbreviations.yaml` next to this file. Patterns are case-insensitive
and applied longest-first so e.g. "Type 2 diabetes mellitus" matches before "diabetes
mellitus" tries to substitute on the leftover.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from importlib import resources

import yaml


@dataclass(frozen=True)
class _Rule:
    pattern: re.Pattern[str]
    abbr: str


@cache
def _load_rules() -> list[_Rule]:
    pkg = resources.files("phantom_codes.data")
    with (pkg / "abbreviations.yaml").open("r") as f:
        raw = yaml.safe_load(f)
    mappings = raw.get("mappings", [])
    # Sort longest-pattern-first so substring patterns don't preempt their parents.
    sorted_mappings = sorted(mappings, key=lambda m: -len(m["pattern"]))
    return [
        _Rule(
            pattern=re.compile(re.escape(m["pattern"]), re.IGNORECASE),
            abbr=m["abbr"],
        )
        for m in sorted_mappings
    ]


def abbreviate(text: str) -> str:
    """Apply abbreviation rules to `text`. Deterministic; cached rules."""
    if not text:
        return text
    out = text
    for rule in _load_rules():
        out = rule.pattern.sub(rule.abbr, out)
    # Collapse double-spaces and trim.
    out = re.sub(r"\s+", " ", out).strip()
    return out
