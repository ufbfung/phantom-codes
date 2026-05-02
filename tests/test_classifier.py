"""Tests for ClassifierModel (inference wrapper).

Builds a tiny fake checkpoint with `bert-base-uncased` weights — no
PubMedBERT download required, no MIMIC content involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from phantom_codes.data.degrade import ICD10_SYSTEM


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build and save a tiny untrained checkpoint shaped like the real thing."""
    from phantom_codes.training.trainer import PubMedBERTClassifier

    code_to_index = {"E11.9": 0, "I10": 1, "J45.909": 2}
    torch.manual_seed(0)
    model = PubMedBERTClassifier(base_model_id="bert-base-uncased", n_codes=3)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "code_to_index": code_to_index,
            "config": {
                "base_model": "bert-base-uncased",
                "max_seq_length": 32,
            },
            "epoch": 0,
            "val_loss": 1.0,
        },
        path,
    )
    return path


def test_predict_returns_top_k_predictions(tiny_checkpoint: Path) -> None:
    from phantom_codes.models.classifier import ClassifierModel

    clf = ClassifierModel(
        checkpoint_path=tiny_checkpoint,
        name="test:classifier",
        device=torch.device("cpu"),
    )
    preds = clf.predict(input_text="diabetes mellitus", top_k=2)

    assert len(preds) == 2
    assert all(p.system == ICD10_SYSTEM for p in preds)
    assert all(p.code in {"E11.9", "I10", "J45.909"} for p in preds)
    # Ranked by score descending.
    assert preds[0].score >= preds[1].score
    # Sigmoid outputs lie in [0, 1].
    assert all(0.0 <= p.score <= 1.0 for p in preds)


def test_predict_caps_top_k_at_vocab_size(tiny_checkpoint: Path) -> None:
    """If top_k exceeds n_codes, return only n_codes predictions."""
    from phantom_codes.models.classifier import ClassifierModel

    clf = ClassifierModel(checkpoint_path=tiny_checkpoint, device=torch.device("cpu"))
    preds = clf.predict(input_text="anything", top_k=100)
    assert len(preds) == 3  # vocab size in the fixture


def test_predict_empty_inputs_returns_empty(tiny_checkpoint: Path) -> None:
    from phantom_codes.models.classifier import ClassifierModel

    clf = ClassifierModel(checkpoint_path=tiny_checkpoint, device=torch.device("cpu"))
    assert clf.predict(input_text=None, input_fhir=None) == []
    assert clf.predict(input_text="") == []


def test_predict_extracts_text_from_fhir(tiny_checkpoint: Path) -> None:
    from phantom_codes.models.classifier import ClassifierModel

    clf = ClassifierModel(checkpoint_path=tiny_checkpoint, device=torch.device("cpu"))
    fhir = {
        "resourceType": "Condition",
        "code": {
            "text": "essential hypertension",
            "coding": [{"display": "Essential hypertension"}],
        },
    }
    preds = clf.predict(input_fhir=fhir, top_k=1)
    assert len(preds) == 1
    assert preds[0].system == ICD10_SYSTEM


def test_predict_falls_back_to_coding_display(tiny_checkpoint: Path) -> None:
    from phantom_codes.models.classifier import ClassifierModel

    clf = ClassifierModel(checkpoint_path=tiny_checkpoint, device=torch.device("cpu"))
    fhir = {
        "resourceType": "Condition",
        "code": {"coding": [{"display": "Asthma, unspecified"}]},
    }
    preds = clf.predict(input_fhir=fhir, top_k=1)
    assert len(preds) == 1


def test_classifier_name_propagates(tiny_checkpoint: Path) -> None:
    from phantom_codes.models.classifier import ClassifierModel

    clf = ClassifierModel(
        checkpoint_path=tiny_checkpoint,
        name="pubmedbert:test",
        device=torch.device("cpu"),
    )
    assert clf.name == "pubmedbert:test"
