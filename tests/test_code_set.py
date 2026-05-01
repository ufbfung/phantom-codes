"""Tests for the ICD vocabulary builder."""

from __future__ import annotations

import copy
from typing import Any

from phantom_codes.data.code_set import build_vocab
from phantom_codes.data.degrade import ICD9_SYSTEM, ICD10_SYSTEM


def _condition(code: str, display: str, system: str = ICD10_SYSTEM) -> dict[str, Any]:
    return {
        "resourceType": "Condition",
        "id": f"cond-{code}",
        "code": {
            "coding": [{"system": system, "code": code, "display": display}],
            "text": display,
        },
    }


def test_top_n_picks_most_frequent() -> None:
    conditions = (
        [_condition("E11.9", "Diabetes")] * 5
        + [_condition("I10", "Hypertension")] * 3
        + [_condition("J18.9", "Pneumonia")] * 1
    )
    vocab = build_vocab(conditions, top_n=2)
    codes = [c for _, c, _ in vocab.entries]
    assert codes == ["E11.9", "I10"]
    assert vocab.size == 2


def test_index_lookup() -> None:
    conditions = [_condition("E11.9", "Diabetes")] * 2 + [_condition("I10", "Hypertension")]
    vocab = build_vocab(conditions, top_n=10)
    assert vocab.index(ICD10_SYSTEM, "E11.9") == 0
    assert vocab.index(ICD10_SYSTEM, "I10") == 1
    assert vocab.index(ICD10_SYSTEM, "Z99.9") is None


def test_only_includes_target_system() -> None:
    conditions = [
        _condition("E11.9", "Diabetes"),
        _condition("250.00", "DM (ICD-9)", system=ICD9_SYSTEM),
        _condition("250.00", "DM (ICD-9)", system=ICD9_SYSTEM),
    ]
    vocab = build_vocab(conditions, top_n=10, system=ICD10_SYSTEM)
    assert [c for _, c, _ in vocab.entries] == ["E11.9"]


def test_to_dict_shape() -> None:
    conditions = [_condition("E11.9", "Diabetes")] * 2
    vocab = build_vocab(conditions, top_n=1)
    out = vocab.to_dict()
    assert out == [{"id": 0, "system": ICD10_SYSTEM, "code": "E11.9", "display": "Diabetes"}]


def test_skips_conditions_without_coding() -> None:
    bad: dict[str, Any] = {"resourceType": "Condition", "id": "x", "code": {"coding": []}}
    good = _condition("E11.9", "Diabetes")
    vocab = build_vocab([bad, good, copy.deepcopy(good)], top_n=10)
    assert vocab.size == 1
