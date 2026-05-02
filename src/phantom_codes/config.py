"""Pydantic config models. Loaded from YAML at CLI entry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class SplitFractions(BaseModel):
    train: float = 0.7
    val: float = 0.1
    test: float = 0.2

    @field_validator("test")
    @classmethod
    def _sum_to_one(cls, v: float, info: Any) -> float:
        train = info.data.get("train", 0.0)
        val = info.data.get("val", 0.0)
        if abs(train + val + v - 1.0) > 1e-6:
            raise ValueError(f"train+val+test must sum to 1.0, got {train + val + v}")
        return v


class DataConfig(BaseModel):
    """Configuration for the data ingestion + preparation pipeline.

    Default operation is fully local: raw FHIR files in `data/mimic/raw/`,
    derived parquet splits in `data/derived/`. To run against a GCS bucket
    instead (README Data setup Option 2), set `derived_bucket` to a
    `gs://...` prefix.
    """

    derived_bucket: str | None = Field(
        default=None,
        description=(
            "Optional GCS bucket prefix (gs://...) for cloud-resident raw + "
            "derived data. When unset (the default), the pipeline reads from "
            "and writes to local paths under `data/`. See README's 'Data "
            "setup' section for when each path applies."
        ),
    )
    resources: list[str] = Field(
        default_factory=lambda: ["MimicCondition"],
        description="FHIR resource file basenames available locally (without .ndjson.gz).",
    )
    top_n_codes: int = Field(default=50, ge=1, description="Vocabulary size for classifier.")
    seed: int = Field(default=42, description="RNG seed for splits and any sampling.")
    splits: SplitFractions = Field(default_factory=SplitFractions)

    @field_validator("derived_bucket")
    @classmethod
    def _is_gs_uri_if_set(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.startswith("gs://"):
            raise ValueError(f"derived_bucket must start with gs:// when set, got {v!r}")
        return v.rstrip("/")

    def raw_uri(self, resource: str) -> str:
        """Where the raw ndjson lives — GCS path if derived_bucket is set, local otherwise."""
        if self.derived_bucket is None:
            return f"data/mimic/raw/{resource}.ndjson.gz"
        return f"{self.derived_bucket}/mimic/raw/{resource}.ndjson.gz"

    def derived_split_uri(self, split: str) -> str:
        """Where derived (degraded) split parquet lives — GCS or local."""
        if self.derived_bucket is None:
            return f"data/derived/conditions/{split}.parquet"
        return f"{self.derived_bucket}/derived/conditions/{split}.parquet"


def load_data_config(path: str | Path) -> DataConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return DataConfig.model_validate(raw)
