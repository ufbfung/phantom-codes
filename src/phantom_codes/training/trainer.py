"""PyTorch fine-tuning loop for the PubMedBERT classifier.

This module is the heart of training. It implements the canonical
fine-tuning pattern: load pre-trained PubMedBERT, attach a randomly-
initialized linear classification head, and train both together on
the train split with periodic validation against the val split. Best
checkpoint (lowest val loss) gets persisted; aggregate metrics are
written to JSON.

------------------------------------------------------------------------
Telemetry / compliance posture
------------------------------------------------------------------------
PhysioNet's responsible-LLM-use policy and the project's CLAUDE.md hard
rules require that no MIMIC-derivative content leaks to third-party
services. Several common ML libraries (wandb, mlflow, comet, etc.)
default to *cloud-syncing* training metrics — which is fine for
non-credentialed data but violates the policy here.

We set defensive environment variables at module import so even if a
future contributor accidentally `import wandb`s, no telemetry leaks:
"""

from __future__ import annotations

import json
import os

# ─────────────────────────────────────────────────────────────────────────
# Defensive telemetry guards (set at module-import time, before any
# downstream library can read them and configure itself).
#
# `setdefault` won't override an explicit user choice — if you actually
# want online wandb on a non-MIMIC run, set `WANDB_DISABLED=false`
# explicitly in your environment first.
# ─────────────────────────────────────────────────────────────────────────
os.environ.setdefault("WANDB_DISABLED", "true")     # don't init wandb at all
os.environ.setdefault("WANDB_MODE", "offline")      # if it inits anyway, stay offline
os.environ.setdefault("MLFLOW_TRACKING_URI", "")    # disable mlflow remote tracking
os.environ.setdefault("COMET_DISABLE_AUTO_LOGGING", "1")
# HuggingFace transformers itself does not sync to anywhere by default,
# but the trainer sometimes asks. Force its tracking off too.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")  # 0 allows HF Hub downloads (PubMedBERT base) but does not push
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # no usage stats sent

# ruff: noqa: E402, I001 — imports below intentionally come after the env-var
# setup above so downstream libraries see our defensive defaults at import.
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from phantom_codes.training.dataset import PhantomCodesDataset, build_code_index
from phantom_codes.training.devices import get_device
from phantom_codes.training.seeding import seed_everything


# ─────────────────────────────────────────────────────────────────────────
# Config dataclass — mirrors configs/training.yaml fields exactly so
# we have one canonical source of truth for hyperparameter names.
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class TrainingConfig:
    """Training hyperparameters. Loadable from configs/training.yaml."""

    base_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    top_n_codes: int = 50

    train_path: str = "data/derived/conditions/train.parquet"
    val_path: str = "data/derived/conditions/val.parquet"
    test_path: str = "data/derived/conditions/test.parquet"

    max_seq_length: int = 256

    batch_size: int = 8
    num_epochs: int = 5
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    early_stopping_patience: int | None = 2

    seed: int = 42

    checkpoint_dir: str = "models/checkpoints/pubmedbert"
    metrics_dir: str = "models/metrics"


# ─────────────────────────────────────────────────────────────────────────
# Model definition — a thin wrapper around the pre-trained encoder
# plus a fresh linear classification head.
#
# Educational note: in PyTorch, "model" = an `nn.Module` subclass. The
# subclass must implement `__init__` (define layers) and `forward`
# (define computation). PyTorch then handles backprop automatically
# via autograd — you never write a backward pass by hand.
# ─────────────────────────────────────────────────────────────────────────


class PubMedBERTClassifier(torch.nn.Module):
    """PubMedBERT encoder + linear head for multi-label ICD classification.

    Architecture:
        input_ids → PubMedBERT encoder → [CLS] embedding (768-dim)
                                           ↓
                                    Linear(768, n_codes)
                                           ↓
                                    raw logits (one per code)

    We use the [CLS] token's final-layer embedding as a fixed-size
    summary of the input. This is the standard BERT-style pattern —
    the encoder is trained (during pre-training) so [CLS] aggregates
    sentence-level information.

    For inference, downstream code can apply `torch.sigmoid` to convert
    logits to per-code probabilities. We don't do that here because
    BCEWithLogitsLoss numerically expects raw logits (it fuses the
    sigmoid into the loss for stability).
    """

    def __init__(self, base_model_id: str, n_codes: int) -> None:
        super().__init__()
        # `AutoModel` loads the pre-trained encoder WITHOUT any task-
        # specific head. PubMedBERT-base has 12 transformer layers,
        # 768 hidden dim, 12 attention heads — ~110M params total.
        self.encoder = AutoModel.from_pretrained(base_model_id)
        hidden_size = self.encoder.config.hidden_size

        # Our classification head: a single linear layer mapping
        # 768-dim → n_codes-dim. Starts random (Kaiming-uniform-ish
        # by default) and gets trained to project the [CLS] embedding
        # into per-code logit space.
        self.classifier = torch.nn.Linear(hidden_size, n_codes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Encoder forward pass. `last_hidden_state` has shape
        # [batch, seq_len, hidden_size]. We grab the [CLS] position
        # (index 0) which BERT-style models use as a sentence summary.
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [batch, hidden_size]

        # Linear head: project to per-code logits.
        logits = self.classifier(cls_embedding)  # [batch, n_codes]
        return logits


# ─────────────────────────────────────────────────────────────────────────
# Training run state — what we want to track epoch-by-epoch.
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class EpochMetrics:
    """Per-epoch training and validation metrics. Aggregate only — no
    per-record content. Safe to persist and share."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy_top1: float  # exact-match top-1 prediction rate
    elapsed_seconds: float


@dataclass
class TrainingResult:
    """Full training run output. Used by both the CLI and tests."""

    epochs: list[EpochMetrics] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    checkpoint_path: str | None = None


# ─────────────────────────────────────────────────────────────────────────
# The training loop itself.
#
# The canonical structure:
#
#     for epoch in range(num_epochs):
#         for batch in train_loader:
#             optimizer.zero_grad()
#             logits = model(**batch)
#             loss = loss_fn(logits, batch["labels"])
#             loss.backward()       ← computes gradients via autograd
#             optimizer.step()      ← updates weights using gradients
#         val_loss = evaluate(model, val_loader)
#         maybe_save_checkpoint(model)
#
# Every additional concern (warmup scheduler, mixed precision, early
# stopping, metrics logging) wraps this kernel.
# ─────────────────────────────────────────────────────────────────────────


def train(config: TrainingConfig) -> TrainingResult:
    """Fine-tune PubMedBERT on the train split, validate on val.

    Returns a TrainingResult with per-epoch metrics. The best (lowest
    val-loss) checkpoint is saved to `config.checkpoint_dir`.
    """
    # Reproducibility: seed before doing ANYTHING else that touches
    # random state (model init, data shuffling, dropout).
    seed_everything(config.seed)
    device = get_device()
    print(f"[train] device={device.type}, seed={config.seed}")

    # ─── Tokenizer & code vocabulary ───
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    code_to_index = build_code_index(config.train_path, top_n=config.top_n_codes)
    n_codes = len(code_to_index)
    print(f"[train] base_model={config.base_model}, n_codes={n_codes}")

    # ─── Datasets & DataLoaders ───
    train_ds = PhantomCodesDataset(
        config.train_path, tokenizer, code_to_index, config.max_seq_length
    )
    val_ds = PhantomCodesDataset(
        config.val_path, tokenizer, code_to_index, config.max_seq_length
    )
    # `shuffle=True` on train so batches are random each epoch (helps
    # generalization). `shuffle=False` on val so validation is
    # deterministic and comparable across runs.
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    print(f"[train] train_size={len(train_ds)}, val_size={len(val_ds)}")

    # ─── Model, loss, optimizer, scheduler ───
    model = PubMedBERTClassifier(config.base_model, n_codes).to(device)

    # BCEWithLogitsLoss = sigmoid + binary cross-entropy fused together
    # for numerical stability. Multi-label means each output dimension
    # is independently predicted (yes/no for each ICD code).
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # AdamW decouples weight decay from the gradient update — vanilla
    # Adam's L2 regularization interacts badly with adaptive learning
    # rates and silently underperforms.
    # See: https://arxiv.org/abs/1711.05101
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Linear warmup + linear decay schedule. Without warmup, BERT
    # fine-tuning often diverges in the first few hundred steps because
    # the random classification head produces wild gradients that
    # destabilize the pre-trained encoder.
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    # ─── Output dirs ───
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.metrics_dir).mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_path = Path(config.checkpoint_dir) / f"best_{run_id}.pt"
    metrics_path = Path(config.metrics_dir) / f"train_{run_id}.json"

    # ─── Training loop ───
    result = TrainingResult()
    epochs_without_improvement = 0
    for epoch in range(config.num_epochs):
        epoch_start = time.perf_counter()

        # ── Train phase ──
        # `model.train()` enables dropout, batch-norm in training mode,
        # etc. (Some layers behave differently between training and
        # eval; this flag is how PyTorch tells them which.)
        model.train()
        running_train_loss = 0.0
        for batch in train_loader:
            # Move every tensor in the batch to the model's device.
            batch = {k: v.to(device) for k, v in batch.items()}

            # The four-line training step:
            optimizer.zero_grad()                            # clear stale grads
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn(logits, batch["labels"])
            loss.backward()                                  # compute grads (autograd)
            optimizer.step()                                 # apply grads to weights
            scheduler.step()                                 # advance LR schedule

            # `.item()` converts a 0-dim tensor to a Python float and
            # detaches it from the autograd graph — critical for
            # accumulating metrics without holding references to the
            # entire computation history (memory leak otherwise).
            running_train_loss += loss.item()
        avg_train_loss = running_train_loss / max(len(train_loader), 1)

        # ── Validation phase ──
        # `model.eval()` disables dropout, freezes batch-norm running
        # stats, etc. `torch.no_grad()` tells PyTorch not to track
        # gradients during this block — saves memory and time.
        model.eval()
        running_val_loss = 0.0
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = loss_fn(logits, batch["labels"])
                running_val_loss += loss.item()

                # Top-1 accuracy: did the highest-logit code match the
                # ground-truth code (the one with label = 1.0)?
                pred_idx = logits.argmax(dim=-1)
                target_idx = batch["labels"].argmax(dim=-1)
                n_correct += int((pred_idx == target_idx).sum().item())
                n_total += int(batch["labels"].size(0))
        avg_val_loss = running_val_loss / max(len(val_loader), 1)
        val_accuracy = n_correct / n_total if n_total else 0.0

        # ── Record + report aggregate metrics ──
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_accuracy_top1=val_accuracy,
            elapsed_seconds=time.perf_counter() - epoch_start,
        )
        result.epochs.append(epoch_metrics)
        # Print aggregate-only — never sample predictions, never per-row data.
        # This output is safe to share with Claude / review.
        print(
            f"[train] epoch={epoch} "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} "
            f"val_top1={val_accuracy:.3f} "
            f"elapsed={epoch_metrics.elapsed_seconds:.0f}s"
        )

        # ── Checkpoint best-by-val-loss ──
        if avg_val_loss < result.best_val_loss:
            result.best_val_loss = avg_val_loss
            result.best_epoch = epoch
            result.checkpoint_path = str(checkpoint_path)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "code_to_index": code_to_index,
                    "config": _config_to_dict(config),
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                },
                checkpoint_path,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # ── Early stopping ──
        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(
                f"[train] early stop after {config.early_stopping_patience} "
                f"epochs without improvement"
            )
            break

    # ─── Persist aggregate metrics JSON ───
    with open(metrics_path, "w") as f:
        json.dump(_result_to_dict(result), f, indent=2)
    print(f"[train] metrics → {metrics_path}")
    if result.checkpoint_path:
        print(f"[train] best checkpoint → {result.checkpoint_path}")

    return result


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    """Serialize TrainingConfig into a JSON-safe dict for checkpoint storage."""
    return {
        "base_model": config.base_model,
        "top_n_codes": config.top_n_codes,
        "max_seq_length": config.max_seq_length,
        "batch_size": config.batch_size,
        "num_epochs": config.num_epochs,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "seed": config.seed,
    }


def _result_to_dict(result: TrainingResult) -> dict[str, Any]:
    """Serialize TrainingResult into a JSON-safe dict for metrics persistence."""
    return {
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "checkpoint_path": result.checkpoint_path,
        "epochs": [
            {
                "epoch": e.epoch,
                "train_loss": e.train_loss,
                "val_loss": e.val_loss,
                "val_accuracy_top1": e.val_accuracy_top1,
                "elapsed_seconds": e.elapsed_seconds,
            }
            for e in result.epochs
        ],
    }
