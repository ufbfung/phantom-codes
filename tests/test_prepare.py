"""End-to-end test: read fixtures → degrade → split → write parquet."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from phantom_codes.config import DataConfig
from phantom_codes.data.prepare import (
    build_records,
    filter_in_scope,
    prepare,
    stratified_split,
)

FIXTURE = Path(__file__).parent / "fixtures" / "conditions.ndjson"


def _data_cfg(local_bucket: str) -> DataConfig:
    return DataConfig.model_validate(
        {
            "derived_bucket": local_bucket,
            "resources": ["MimicCondition"],
            "top_n_codes": 50,
            "seed": 42,
            "splits": {"train": 0.7, "val": 0.1, "test": 0.2},
        }
    )


IN_SCOPE_FIXTURE_IDS = {
    "fixture-001",  # E11.9 — DM2 (CKM)
    "fixture-002",  # I10 — hypertension (eCKM)
    "fixture-007-ckm-cad",  # I25.10 — CAD (CKM)
    "fixture-008-ckm-ckd3a",  # N18.31 — CKD 3a (CKM)
    "fixture-009-eckm-dyslipidemia",  # E78.5 (eCKM)
    "fixture-010-eckm-obesity",  # E66.9 (eCKM)
}


def test_filter_in_scope_drops_out_of_scope() -> None:
    conditions = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    in_scope = list(filter_in_scope(conditions))
    in_scope_ids = {c["id"] for c in in_scope}
    assert in_scope_ids == IN_SCOPE_FIXTURE_IDS


def test_build_records_emits_one_per_mode_per_in_scope_condition() -> None:
    """build_records drops out-of-scope conditions and emits 4 modes for each in-scope one."""
    conditions = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    rows = build_records(conditions)
    assert len(rows) == len(IN_SCOPE_FIXTURE_IDS) * 4
    modes_per_id: dict[str, list[str]] = {}
    for r in rows:
        modes_per_id.setdefault(r["resource_id"], []).append(r["mode"])
    assert set(modes_per_id) == IN_SCOPE_FIXTURE_IDS
    for rid, modes in modes_per_id.items():
        assert sorted(modes) == ["D1_full", "D2_no_code", "D3_text_only", "D4_abbreviated"], rid


def test_build_records_attaches_group_label() -> None:
    conditions = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    rows = build_records(conditions)
    by_id = {r["resource_id"]: r["gt_group"] for r in rows}
    assert by_id["fixture-001"] == "ckm"
    assert by_id["fixture-002"] == "eckm"
    assert by_id["fixture-007-ckm-cad"] == "ckm"
    assert by_id["fixture-009-eckm-dyslipidemia"] == "eckm"


def test_stratified_split_keeps_modes_together() -> None:
    """All four modes for a given resource_id must land in the same split."""
    conditions = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    rows = build_records(conditions)
    splits = stratified_split(rows, fractions=(0.6, 0.2, 0.2), seed=0)

    id_to_split: dict[str, set[str]] = {}
    for split_name, split_rows in splits.items():
        for r in split_rows:
            id_to_split.setdefault(r["resource_id"], set()).add(split_name)
    for rid, splits_seen in id_to_split.items():
        assert len(splits_seen) == 1, f"{rid} crossed splits: {splits_seen}"


def test_prepare_writes_parquet_locally(tmp_path: Path) -> None:
    cfg = _data_cfg("gs://throwaway/phantom-codes")
    out = tmp_path / "out"
    written = prepare(cfg, FIXTURE, local_out=out)
    assert set(written.keys()) == {"train", "val", "test"}
    for path in written.values():
        df = pd.read_parquet(path)
        assert {"resource_id", "mode", "input_fhir", "input_text", "gt_code"}.issubset(df.columns)
        assert (df["mode"].isin(["D1_full", "D2_no_code", "D3_text_only", "D4_abbreviated"])).all()


def test_prepare_is_deterministic_under_same_seed(tmp_path: Path) -> None:
    cfg = _data_cfg("gs://throwaway/phantom-codes")
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    prepare(cfg, FIXTURE, local_out=out_a)
    prepare(cfg, FIXTURE, local_out=out_b)

    for split in ["train", "val", "test"]:
        a = pd.read_parquet(out_a / f"{split}.parquet").sort_values("resource_id").reset_index(drop=True)
        b = pd.read_parquet(out_b / f"{split}.parquet").sort_values("resource_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
