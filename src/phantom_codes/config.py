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
    """Configuration for the data ingestion + preparation pipeline."""

    derived_bucket: str = Field(
        ...,
        description=(
            "GCS bucket prefix (gs://...) where the user-uploaded raw FHIR "
            "ndjson files live and where derived parquet splits will be "
            "written. PhysioNet does not host MIMIC-IV-FHIR on GCS, so users "
            "manually wget the files and `gcloud storage cp` them to their "
            "own bucket — see README's 'Data setup' section."
        ),
    )
    resources: list[str] = Field(
        default_factory=lambda: ["MimicCondition"],
        description="FHIR resource file basenames the user has uploaded (without .ndjson.gz).",
    )
    top_n_codes: int = Field(default=50, ge=1, description="Vocabulary size for classifier.")
    seed: int = Field(default=42, description="RNG seed for splits and any sampling.")
    splits: SplitFractions = Field(default_factory=SplitFractions)

    @field_validator("derived_bucket")
    @classmethod
    def _is_gs_uri(cls, v: str) -> str:
        if not v.startswith("gs://"):
            raise ValueError(f"bucket must start with gs://, got {v!r}")
        return v.rstrip("/")

    def raw_uri(self, resource: str) -> str:
        """Where a manually-uploaded raw ndjson lives in the user's derived bucket."""
        return f"{self.derived_bucket}/mimic/raw/{resource}.ndjson.gz"

    def derived_split_uri(self, split: str) -> str:
        """Where a derived (degraded) split parquet lives."""
        return f"{self.derived_bucket}/derived/conditions/{split}.parquet"


def load_data_config(path: str | Path) -> DataConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return DataConfig.model_validate(raw)
