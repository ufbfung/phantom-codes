"""Tests for the PubMedBERT trainer module.

We use `bert-base-uncased` instead of PubMedBERT so CI doesn't have to
download a 400MB+ medical model. The training-loop logic — forward,
backward, step, checkpointing, early stopping — doesn't care which
encoder we plug in.

These tests run on synthetic fixtures only. No MIMIC content involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch


def _write_synthetic_parquet(path: Path, rows: int = 8) -> Path:
    """Build a tiny parquet matching the schema produced by `prepare`."""
    base = [
        {
            "resource_id": f"r{i}",
            "mode": "D3_text_only",
            "input_fhir": None,
            "input_text": f"Type 2 diabetes mellitus example {i}",
            "gt_system": "http://hl7.org/fhir/sid/icd-10-cm",
            "gt_code": "E11.9" if i % 2 == 0 else "I10",
            "gt_display": "Type 2 diabetes mellitus" if i % 2 == 0 else "Hypertension",
            "gt_group": "ckm" if i % 2 == 0 else "eckm",
        }
        for i in range(rows)
    ]
    pd.DataFrame(base).to_parquet(path, index=False)
    return path


@pytest.fixture(scope="module")
def small_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bert-base-uncased")


def test_classifier_forward_shape(small_tokenizer) -> None:
    """PubMedBERTClassifier returns logits of shape [batch, n_codes]."""
    from phantom_codes.training.trainer import PubMedBERTClassifier

    model = PubMedBERTClassifier(base_model_id="bert-base-uncased", n_codes=4)
    encoded = small_tokenizer(
        ["hello world", "another short text"],
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    logits = model(encoded["input_ids"], encoded["attention_mask"])
    assert logits.shape == (2, 4)


def test_one_training_step_runs_without_crashing(small_tokenizer) -> None:
    """Forward + backward + optimizer.step() on a fake batch updates weights."""
    from phantom_codes.training.trainer import PubMedBERTClassifier

    torch.manual_seed(0)
    model = PubMedBERTClassifier(base_model_id="bert-base-uncased", n_codes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    encoded = small_tokenizer(
        ["diabetes"], padding="max_length", max_length=8, return_tensors="pt"
    )
    labels = torch.tensor([[1.0, 0.0, 0.0]])

    # Snapshot a parameter to confirm the step actually changed it.
    head_weight_before = model.classifier.weight.detach().clone()

    optimizer.zero_grad()
    logits = model(encoded["input_ids"], encoded["attention_mask"])
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).item()
    assert not torch.equal(head_weight_before, model.classifier.weight)


def test_train_end_to_end_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    """Full `train()` runs against synthetic parquets, persists artifacts."""
    from phantom_codes.training.trainer import TrainingConfig, train

    train_path = _write_synthetic_parquet(tmp_path / "train.parquet", rows=8)
    val_path = _write_synthetic_parquet(tmp_path / "val.parquet", rows=4)

    cfg = TrainingConfig(
        base_model="bert-base-uncased",
        top_n_codes=2,
        train_path=str(train_path),
        val_path=str(val_path),
        max_seq_length=16,
        batch_size=2,
        num_epochs=1,
        learning_rate=2e-5,
        warmup_steps=0,
        early_stopping_patience=None,
        seed=42,
        checkpoint_dir=str(tmp_path / "ckpt"),
        metrics_dir=str(tmp_path / "metrics"),
    )
    result = train(cfg)

    assert len(result.epochs) == 1
    assert result.checkpoint_path is not None
    assert Path(result.checkpoint_path).exists()
    assert result.best_epoch == 0

    # Metrics JSON exists and contains aggregate-only fields.
    metrics_files = list((tmp_path / "metrics").glob("train_*.json"))
    assert len(metrics_files) == 1
    payload = json.loads(metrics_files[0].read_text())
    assert "epochs" in payload
    assert payload["epochs"][0].keys() == {
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy_top1",
        "elapsed_seconds",
    }


def test_telemetry_env_vars_set_on_import() -> None:
    """Importing the trainer module sets defensive telemetry guards."""
    import os

    import phantom_codes.training.trainer  # noqa: F401

    assert os.environ.get("WANDB_DISABLED") == "true"
    assert os.environ.get("WANDB_MODE") == "offline"
    assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"


def test_checkpoint_round_trip_preserves_state(tmp_path: Path, small_tokenizer) -> None:
    """Saving a model state and loading it back yields identical outputs."""
    from phantom_codes.training.trainer import PubMedBERTClassifier

    torch.manual_seed(0)
    model = PubMedBERTClassifier(base_model_id="bert-base-uncased", n_codes=3)
    code_to_index = {"E11.9": 0, "I10": 1, "J45.909": 2}

    ckpt_path = tmp_path / "tiny.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "code_to_index": code_to_index,
            "config": {"base_model": "bert-base-uncased", "max_seq_length": 16},
        },
        ckpt_path,
    )

    loaded = torch.load(ckpt_path, weights_only=False)
    rebuilt = PubMedBERTClassifier(base_model_id="bert-base-uncased", n_codes=3)
    rebuilt.load_state_dict(loaded["model_state_dict"])

    encoded = small_tokenizer(
        ["test text"], padding="max_length", max_length=8, return_tensors="pt"
    )
    model.eval()
    rebuilt.eval()
    with torch.no_grad():
        out_a = model(encoded["input_ids"], encoded["attention_mask"])
        out_b = rebuilt(encoded["input_ids"], encoded["attention_mask"])
    assert torch.allclose(out_a, out_b)
    assert loaded["code_to_index"] == code_to_index
