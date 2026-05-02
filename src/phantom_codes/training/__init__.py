"""PyTorch training utilities for the PubMedBERT classifier.

This package contains everything needed to fine-tune PubMedBERT on local
MIMIC train/val/test splits and persist the resulting checkpoint for
later use by `phantom_codes.models.classifier.ClassifierModel` in the
eval matrix.

Submodules:
    devices  — auto-detect MPS / CUDA / CPU
    seeding  — reproducible RNG initialization
    dataset  — PyTorch Dataset over our parquet splits
    trainer  — training loop with validation, early stopping, checkpointing

The trainer module sets defensive environment variables at import time
to disable cloud telemetry libraries (wandb, mlflow, etc.) by default.
This is belt-and-suspenders compliance with PhysioNet's responsible-LLM-
use policy: even if a future contributor accidentally imports wandb,
no MIMIC-derived training metrics leak to a cloud service.

For a learning-oriented introduction to what this code does, start with
docs/learning/00-pytorch-foundations.md.
"""

from phantom_codes.training.devices import get_device
from phantom_codes.training.seeding import seed_everything

__all__ = ["get_device", "seed_everything"]
