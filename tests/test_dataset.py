"""Tests for PhantomCodesDataset.

We avoid downloading PubMedBERT in CI by using a tiny BERT-style
tokenizer instead (`bert-base-uncased` ≈ 28MB, cached). The dataset's
behavior — parquet loading, label vector construction, schema — doesn't
depend on which tokenizer we use.

These tests run on synthetic fixtures only. No MIMIC content involved.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

# Import lazily inside tests so the (heavy) transformers import doesn't
# slow down test collection for unrelated test files.


def _write_synthetic_parquet(tmp_path: Path) -> Path:
    """Build a tiny parquet matching the schema produced by `prepare`."""
    df = pd.DataFrame(
        [
            {
                "resource_id": "r1",
                "mode": "D3_text_only",
                "input_fhir": None,
                "input_text": "Type 2 diabetes mellitus",
                "gt_system": "http://hl7.org/fhir/sid/icd-10-cm",
                "gt_code": "E11.9",
                "gt_display": "Type 2 diabetes mellitus",
                "gt_group": "ckm",
            },
            {
                "resource_id": "r2",
                "mode": "D3_text_only",
                "input_fhir": None,
                "input_text": "Essential hypertension",
                "gt_system": "http://hl7.org/fhir/sid/icd-10-cm",
                "gt_code": "I10",
                "gt_display": "Essential hypertension",
                "gt_group": "eckm",
            },
            {
                "resource_id": "r3",
                "mode": "D4_abbreviated",
                "input_fhir": None,
                "input_text": "Pt with T2DM unc.",
                "gt_system": "http://hl7.org/fhir/sid/icd-10-cm",
                "gt_code": "E11.9",
                "gt_display": "Type 2 diabetes mellitus",
                "gt_group": "ckm",
            },
        ]
    )
    path = tmp_path / "train.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture(scope="module")
def small_tokenizer():
    """Use bert-base-uncased — small, no medical specialization needed for tests."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bert-base-uncased")


def test_build_code_index_picks_top_n_by_frequency(tmp_path: Path) -> None:
    from phantom_codes.training.dataset import build_code_index

    path = _write_synthetic_parquet(tmp_path)
    # E11.9 appears twice, I10 once. top_n=1 → only E11.9.
    idx = build_code_index(path, top_n=1)
    assert idx == {"E11.9": 0}

    idx2 = build_code_index(path, top_n=10)
    assert set(idx2) == {"E11.9", "I10"}
    assert all(0 <= v < 2 for v in idx2.values())


def test_dataset_len_matches_parquet_rows(tmp_path: Path, small_tokenizer) -> None:
    from phantom_codes.training.dataset import PhantomCodesDataset

    path = _write_synthetic_parquet(tmp_path)
    ds = PhantomCodesDataset(path, small_tokenizer, {"E11.9": 0, "I10": 1}, max_seq_length=32)
    assert len(ds) == 3


def test_dataset_returns_correct_tensor_shapes(tmp_path: Path, small_tokenizer) -> None:
    from phantom_codes.training.dataset import PhantomCodesDataset

    path = _write_synthetic_parquet(tmp_path)
    code_to_index = {"E11.9": 0, "I10": 1}
    ds = PhantomCodesDataset(path, small_tokenizer, code_to_index, max_seq_length=32)
    sample = ds[0]
    # input_ids and attention_mask: [seq_length]
    assert sample["input_ids"].shape == (32,)
    assert sample["attention_mask"].shape == (32,)
    # labels: [n_codes]
    assert sample["labels"].shape == (2,)
    assert sample["labels"].dtype == torch.float32


def test_dataset_labels_one_hot_for_known_code(tmp_path: Path, small_tokenizer) -> None:
    from phantom_codes.training.dataset import PhantomCodesDataset

    path = _write_synthetic_parquet(tmp_path)
    code_to_index = {"E11.9": 0, "I10": 1}
    ds = PhantomCodesDataset(path, small_tokenizer, code_to_index, max_seq_length=32)
    # Row 0 has gt_code=E11.9 → label vector [1, 0]
    labels0 = ds[0]["labels"]
    assert labels0[0].item() == 1.0
    assert labels0[1].item() == 0.0
    # Row 1 has gt_code=I10 → label vector [0, 1]
    labels1 = ds[1]["labels"]
    assert labels1[0].item() == 0.0
    assert labels1[1].item() == 1.0


def test_dataset_unknown_code_yields_zero_label_vector(
    tmp_path: Path, small_tokenizer
) -> None:
    """If the gt_code isn't in our top-N vocab, no label position is set."""
    from phantom_codes.training.dataset import PhantomCodesDataset

    path = _write_synthetic_parquet(tmp_path)
    # Vocab only includes I10 — E11.9 rows should produce all-zero labels.
    code_to_index = {"I10": 0}
    ds = PhantomCodesDataset(path, small_tokenizer, code_to_index, max_seq_length=32)
    e11_sample = ds[0]  # gt_code=E11.9, not in vocab
    assert e11_sample["labels"].sum().item() == 0.0
    i10_sample = ds[1]  # gt_code=I10, in vocab
    assert i10_sample["labels"].sum().item() == 1.0
