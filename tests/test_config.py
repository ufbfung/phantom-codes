"""Config validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from phantom_codes.config import DataConfig, load_data_config


def _valid_config_dict() -> dict:
    return {
        "derived_bucket": "gs://my-bucket/phantom-codes",
        "resources": ["MimicCondition"],
        "top_n_codes": 50,
        "seed": 42,
        "splits": {"train": 0.7, "val": 0.1, "test": 0.2},
    }


def test_valid_config_round_trips() -> None:
    cfg = DataConfig.model_validate(_valid_config_dict())
    assert cfg.derived_bucket == "gs://my-bucket/phantom-codes"
    assert cfg.resources == ["MimicCondition"]


def test_split_must_sum_to_one() -> None:
    bad = _valid_config_dict()
    bad["splits"] = {"train": 0.5, "val": 0.1, "test": 0.2}
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        DataConfig.model_validate(bad)


def test_bucket_must_be_gs_uri() -> None:
    bad = _valid_config_dict()
    bad["derived_bucket"] = "s3://wrong-cloud"
    with pytest.raises(ValidationError, match="gs://"):
        DataConfig.model_validate(bad)


def test_uri_helpers_with_gcs_bucket() -> None:
    cfg = DataConfig.model_validate(_valid_config_dict())
    assert cfg.raw_uri("MimicCondition") == (
        "gs://my-bucket/phantom-codes/mimic/raw/MimicCondition.ndjson.gz"
    )
    assert cfg.derived_split_uri("train") == (
        "gs://my-bucket/phantom-codes/derived/conditions/train.parquet"
    )


def test_uri_helpers_default_to_local_when_bucket_unset() -> None:
    """When derived_bucket is None, raw_uri / derived_split_uri return local paths."""
    cfg = DataConfig.model_validate({
        "resources": ["MimicCondition"],
        "top_n_codes": 50,
        "seed": 42,
        "splits": {"train": 0.7, "val": 0.1, "test": 0.2},
    })
    assert cfg.derived_bucket is None
    assert cfg.raw_uri("MimicCondition") == "data/mimic/raw/MimicCondition.ndjson.gz"
    assert cfg.derived_split_uri("train") == "data/derived/conditions/train.parquet"


def test_derived_bucket_is_optional() -> None:
    """A YAML config without derived_bucket should validate cleanly."""
    cfg = DataConfig.model_validate({
        "resources": ["MimicCondition"],
        "splits": {"train": 0.7, "val": 0.1, "test": 0.2},
    })
    assert cfg.derived_bucket is None


def test_load_from_yaml(tmp_path: Path) -> None:
    p = tmp_path / "data.yaml"
    p.write_text(yaml.safe_dump(_valid_config_dict()))
    cfg = load_data_config(p)
    assert cfg.top_n_codes == 50
