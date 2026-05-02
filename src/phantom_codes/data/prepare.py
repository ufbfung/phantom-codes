"""Build train/val/test splits of (degraded input, ground truth) records as parquet.

Reads FHIR Conditions (from local fixture or GCS), applies all four degradation modes,
splits stratified by ICD code, and writes parquet to the configured derived bucket.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from phantom_codes.config import DataConfig
from phantom_codes.data.degrade import (
    ICD10_SYSTEM,
    degrade_all_modes,
    extract_ground_truth,
)
from phantom_codes.data.disease_groups import AccessScope
from phantom_codes.data.disease_groups import load as load_scope
from phantom_codes.data.fhir_loader import iter_conditions


def filter_in_scope(
    conditions: Iterable[dict[str, Any]],
    scope: AccessScope | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield only conditions whose ground-truth code is in ACCESS scope (CKM or eCKM)
    and uses ICD-10-CM. ICD-9, malformed codings, and out-of-scope codes are dropped.
    """
    s = scope or load_scope()
    for cond in conditions:
        try:
            gt = extract_ground_truth(cond)
        except ValueError:
            continue
        if gt.system != ICD10_SYSTEM:
            continue
        if not s.is_in_scope(gt.code):
            continue
        yield cond


def build_records(
    conditions: Iterable[dict[str, Any]],
    scope: AccessScope | None = None,
) -> list[dict[str, Any]]:
    """For each in-scope condition, emit one row per degradation mode.

    Out-of-scope (non-ACCESS, non-ICD-10) conditions are dropped here.
    """
    s = scope or load_scope()
    rows: list[dict[str, Any]] = []
    for cond in filter_in_scope(conditions, scope=s):
        for rec in degrade_all_modes(cond):
            group = s.group_for(rec.ground_truth.code)
            rows.append(
                {
                    "resource_id": rec.resource_id,
                    "mode": str(rec.mode),
                    "input_fhir": (
                        json.dumps(rec.input_fhir) if rec.input_fhir is not None else None
                    ),
                    "input_text": rec.input_text,
                    "gt_system": rec.ground_truth.system,
                    "gt_code": rec.ground_truth.code,
                    "gt_display": rec.ground_truth.display,
                    "gt_group": group,
                }
            )
    return rows


def stratified_split(
    rows: list[dict[str, Any]],
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Split by `resource_id` (so all 4 modes of the same condition land in the same split),
    stratified by ground-truth code so rare codes appear in every split where possible.
    """
    train_frac, val_frac, _ = fractions

    by_code: dict[str, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()
    for r in rows:
        rid = r["resource_id"]
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        by_code[r["gt_code"]].append(rid)

    rng = random.Random(seed)
    split_for_id: dict[str, str] = {}
    for ids in by_code.values():
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        # Edge case: if n is small, still try to put ≥1 in each split when possible.
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n - n_train - n_val)
            n_train = n - n_val - n_test
        else:
            n_test = n - n_train - n_val
        for i, rid in enumerate(ids):
            if i < n_train:
                split_for_id[rid] = "train"
            elif i < n_train + n_val:
                split_for_id[rid] = "val"
            else:
                split_for_id[rid] = "test"

    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for r in rows:
        out[split_for_id[r["resource_id"]]].append(r)
    return out


RECORD_COLUMNS = [
    "resource_id",
    "mode",
    "input_fhir",
    "input_text",
    "gt_system",
    "gt_code",
    "gt_display",
    "gt_group",
]


def write_splits(
    splits: dict[str, list[dict[str, Any]]],
    config: DataConfig,
    local_dir: Path | None = None,
) -> dict[str, str]:
    """Write each split as parquet. If `local_dir` is set, write there instead of GCS.

    Empty splits still get a parquet file with the canonical schema so downstream
    readers don't need to special-case empty inputs.
    """
    written: dict[str, str] = {}
    for split, rows in splits.items():
        df = pd.DataFrame(rows, columns=RECORD_COLUMNS)
        if local_dir is not None:
            local_dir.mkdir(parents=True, exist_ok=True)
            path = local_dir / f"{split}.parquet"
            df.to_parquet(path, index=False)
            written[split] = str(path)
        else:
            uri = config.derived_split_uri(split)
            # Local paths need their parent dir to exist; gs:// URIs handle
            # this server-side. pd.to_parquet does not auto-create dirs.
            if not uri.startswith("gs://"):
                Path(uri).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(uri, index=False)
            written[split] = uri
    return written


def prepare(
    config: DataConfig,
    source_uri: str | Path,
    local_out: Path | None = None,
) -> dict[str, str]:
    """End-to-end: read conditions from `source_uri`, degrade, split, write parquet."""
    conditions = iter_conditions(source_uri)
    rows = build_records(conditions)
    splits = stratified_split(
        rows,
        fractions=(config.splits.train, config.splits.val, config.splits.test),
        seed=config.seed,
    )
    return write_splits(splits, config, local_dir=local_out)
