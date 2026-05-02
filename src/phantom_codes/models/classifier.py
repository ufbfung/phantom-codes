"""Inference wrapper for the trained PubMedBERT classifier.

This is the "deployment side" of the trained model — it loads a
checkpoint produced by `phantom_codes.training.trainer.train()` and
implements the `ConceptNormalizer.predict()` interface so the
classifier can drop into the eval matrix alongside LLMs and baselines.

Conceptual primer (training vs inference):
    A trained model has two lives. *Training* fits the model's weights
    to data; *inference* uses the fixed weights to make predictions on
    new inputs. Different code paths, different concerns:

    Training: forward + backward + optimizer step; needs gradients,
              dropout active, batch norm in training mode, etc.
    Inference: forward only; no gradients (`torch.no_grad`), dropout
              disabled (`model.eval()`), no optimizer.

    PyTorch's `model.eval()` and `torch.no_grad()` are how we tell
    the framework which mode we're in. Forgetting either one during
    inference is a common bug — gives slightly different results AND
    wastes memory by tracking gradient graphs we'll never use.

This class is inference-only. For training, use the `train()` function
in `phantom_codes.training.trainer`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from phantom_codes.data.degrade import ICD10_SYSTEM
from phantom_codes.models.base import ConceptNormalizer, Prediction
from phantom_codes.training.devices import get_device
from phantom_codes.training.trainer import PubMedBERTClassifier


class ClassifierModel(ConceptNormalizer):
    """Loads a trained PubMedBERT checkpoint and implements `predict()`.

    Args:
        checkpoint_path: Path to a `.pt` file produced by
            `training.trainer.train()`. Contains the model state dict,
            the code-to-index mapping, and the training config.
        name: Model name in the eval matrix (e.g.,
            "pubmedbert:classifier"). Used as the `model_name`
            column in the per-prediction CSV.
        device: Optional `torch.device` override. Default: auto-detect
            (MPS → CUDA → CPU).

    Example:
        >>> clf = ClassifierModel("models/checkpoints/pubmedbert/best_*.pt")
        >>> preds = clf.predict(input_text="Type 2 diabetes mellitus", top_k=5)
        >>> [p.code for p in preds]
        ['E11.9', 'E11.65', 'E11.0', ...]
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        name: str = "pubmedbert:classifier",
        device: torch.device | None = None,
    ) -> None:
        self.name = name
        self._device = device if device is not None else get_device()

        # Load the checkpoint dict. `weights_only=True` is the
        # safety-default in modern PyTorch — restricts loading to
        # tensors only (no arbitrary code execution from pickle).
        # We need code_to_index too, so we set it to False but only
        # because we trust the file (we wrote it ourselves).
        ckpt: dict[str, Any] = torch.load(
            checkpoint_path, map_location=self._device, weights_only=False
        )

        # Reconstruct the model architecture from the saved config,
        # then load the trained weights into it.
        config = ckpt["config"]
        self._code_to_index: dict[str, int] = ckpt["code_to_index"]
        self._index_to_code: list[str] = sorted(
            self._code_to_index, key=lambda c: self._code_to_index[c]
        )
        self._model = PubMedBERTClassifier(
            base_model_id=config["base_model"],
            n_codes=len(self._code_to_index),
        ).to(self._device)
        self._model.load_state_dict(ckpt["model_state_dict"])

        # Inference mode — disables dropout, freezes batch-norm stats.
        self._model.eval()

        # Tokenizer matches training settings.
        self._tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        self._max_seq_length = config["max_seq_length"]

    def predict(
        self,
        *,
        input_fhir: dict[str, Any] | None = None,
        input_text: str | None = None,
        top_k: int = 5,
    ) -> list[Prediction]:
        # Resolve the text to feed the model — same logic as the
        # training-time dataset.
        text = self._resolve_text(input_fhir, input_text)
        if not text:
            return []

        # Tokenize. We do this per-call rather than batching across
        # records because the eval runner calls us one record at a
        # time — keeps the inference path simple. If we ever needed
        # throughput we'd add a batched-predict method.
        encoded = self._tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self._max_seq_length,
            return_tensors="pt",
        ).to(self._device)

        # Forward pass under `no_grad` — we don't need gradients for
        # inference, so disable autograd to save memory and time.
        with torch.no_grad():
            logits = self._model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
        # `logits` shape: [1, n_codes]. Squeeze the batch dim.
        logits = logits.squeeze(0)

        # Convert to per-code probabilities via sigmoid (multi-label,
        # not softmax). Then pick top-k by probability.
        probs = torch.sigmoid(logits)
        topk = torch.topk(probs, k=min(top_k, len(self._index_to_code)))

        out: list[Prediction] = []
        for prob, idx in zip(topk.values.tolist(), topk.indices.tolist(), strict=True):
            code = self._index_to_code[idx]
            out.append(
                Prediction(
                    system=ICD10_SYSTEM,
                    code=code,
                    display=None,  # we don't store displays in the index
                    score=float(prob),
                )
            )
        return out

    @staticmethod
    def _resolve_text(
        input_fhir: dict[str, Any] | None, input_text: str | None
    ) -> str:
        """Pick the text to feed the model.

        Mirrors the training-time logic in
        `phantom_codes.training.dataset.PhantomCodesDataset._extract_text`.
        Inference inputs come from the eval runner, which constructs
        them from degraded FHIR Conditions (D1–D4 modes).
        """
        if input_text:
            return input_text
        if input_fhir is None:
            return ""
        code = input_fhir.get("code") or {}
        text = code.get("text")
        if text:
            return str(text)
        for coding in code.get("coding") or []:
            if isinstance(coding.get("display"), str):
                return str(coding["display"])
        # Last resort — JSON-dump the FHIR resource.
        import json

        return json.dumps(input_fhir)
