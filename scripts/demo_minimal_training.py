"""End-to-end PubMedBERT training demo on synthetic fixtures.

What this script does:
    1. Builds tiny train + val parquets from synthetic data (no MIMIC
       content involved).
    2. Runs the full training loop for a couple of epochs using
       `bert-base-uncased` (small, downloaded once, cached) instead of
       PubMedBERT (would require the larger medical model download).
    3. Loads the resulting checkpoint via `ClassifierModel` and prints a
       sample top-K prediction so you can see the inference path work.

What this script is for:
    - A safe place to debug the training/inference pipeline without
      touching real MIMIC data.
    - A hands-on companion to `docs/learning/00..04`. Modify
      hyperparameters, swap the encoder, change the loss — observe what
      happens.
    - A smoke test of the entire training pipeline. If this fails, the
      real-MIMIC training would also fail.

Compliance:
    Synthetic fixtures only. No MIMIC content is read or produced.
    Outputs go to a temp directory — never `models/checkpoints/`.

Usage:
    uv run python scripts/demo_minimal_training.py

Expected runtime:
    Under 60s on M1 MPS. First run is slower (downloads bert-base-uncased
    on the order of ~440MB; cached locally afterward).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import torch

from phantom_codes.models.classifier import ClassifierModel
from phantom_codes.training.trainer import TrainingConfig, train

# ─────────────────────────────────────────────────────────────────────────
# Hand-built synthetic data — no MIMIC content. Repeats the same handful
# of conditions enough times to give the model something learnable.
# ─────────────────────────────────────────────────────────────────────────
SYNTHETIC_ROWS = [
    ("Type 2 diabetes mellitus without complications", "E11.9"),
    ("T2DM, uncomplicated", "E11.9"),
    ("Diabetes mellitus type 2 unspecified", "E11.9"),
    ("Patient with type 2 diabetes", "E11.9"),
    ("Essential (primary) hypertension", "I10"),
    ("Essential hypertension, primary", "I10"),
    ("HTN, essential", "I10"),
    ("Patient with primary hypertension", "I10"),
    ("Asthma, unspecified", "J45.909"),
    ("Unspecified asthma without complications", "J45.909"),
    ("Asthma NOS", "J45.909"),
    ("Chronic asthma unspecified", "J45.909"),
]


def _build_parquet(path: Path, repeats: int) -> Path:
    rows = []
    for repeat in range(repeats):
        for i, (text, code) in enumerate(SYNTHETIC_ROWS):
            rows.append(
                {
                    "resource_id": f"r-{repeat}-{i}",
                    "mode": "D3_text_only",
                    "input_fhir": None,
                    "input_text": text,
                    "gt_system": "http://hl7.org/fhir/sid/icd-10-cm",
                    "gt_code": code,
                    "gt_display": code,
                    "gt_group": "ckm",
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="phantom_demo_") as tmp:
        tmp_path = Path(tmp)
        print(f"[demo] working dir: {tmp_path}")

        train_path = _build_parquet(tmp_path / "train.parquet", repeats=8)
        val_path = _build_parquet(tmp_path / "val.parquet", repeats=2)
        print(
            f"[demo] built train ({len(SYNTHETIC_ROWS) * 8} rows) "
            f"and val ({len(SYNTHETIC_ROWS) * 2} rows)"
        )

        cfg = TrainingConfig(
            base_model="bert-base-uncased",
            top_n_codes=3,
            train_path=str(train_path),
            val_path=str(val_path),
            max_seq_length=32,
            batch_size=8,
            num_epochs=3,
            learning_rate=5e-5,
            warmup_steps=5,
            early_stopping_patience=None,
            seed=42,
            checkpoint_dir=str(tmp_path / "ckpt"),
            metrics_dir=str(tmp_path / "metrics"),
        )

        print("[demo] starting training…")
        result = train(cfg)
        print(
            f"[demo] training done. best_epoch={result.best_epoch} "
            f"best_val_loss={result.best_val_loss:.4f}"
        )

        if result.checkpoint_path is None:
            print("[demo] no checkpoint produced — bailing")
            return

        print("[demo] reloading checkpoint via ClassifierModel…")
        clf = ClassifierModel(
            checkpoint_path=result.checkpoint_path,
            name="demo:classifier",
            device=torch.device("cpu"),
        )

        # Print top-K predictions for a few sanity-check inputs. Safe to
        # show because these are synthetic strings we wrote ourselves.
        for probe in (
            "Type 2 diabetes mellitus",
            "Essential hypertension",
            "Asthma without complications",
        ):
            preds = clf.predict(input_text=probe, top_k=3)
            ranked = ", ".join(f"{p.code}={p.score:.3f}" for p in preds)
            print(f"[demo]   {probe!r:50} → {ranked}")


if __name__ == "__main__":
    main()
