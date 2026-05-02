"""PyTorch Dataset over our prepared parquet splits.

Conceptual primer (PyTorch's data abstraction):
    PyTorch's training loop expects a `Dataset` and a `DataLoader`.
    They have a clean separation of concerns:

    - `Dataset`  — knows how to produce ONE example by integer index.
                   Implements `__len__` (how many examples?) and
                   `__getitem__(i)` (give me the i-th example).
    - `DataLoader` — knows how to BATCH many examples for the GPU.
                   Handles shuffling, parallel loading, pinning memory,
                   collation. You don't write this; you instantiate it.

    The DataLoader calls our Dataset's `__getitem__(0)`, `__getitem__(1)`,
    etc. in some order, collects `batch_size` examples, stacks them
    into tensors, and yields the batch to the training loop. We never
    write the batching logic ourselves — DataLoader does it.

What this Dataset does:
    For each row in our prepared parquet (one per (resource_id, mode)
    combination), produces:
      - input_text: the degraded clinical text the model sees
                    (D1_full = full FHIR JSON; D2/D3/D4 = progressively
                    less context; see data/degrade.py for details)
      - label_vec: a multi-hot vector of length `n_codes` with a 1 in
                   the position of the ground-truth ICD code (0
                   elsewhere). Multi-hot because the *eventual* loss is
                   binary cross-entropy with logits, which expects
                   floating-point targets.

Why text → tokens happens here (not separately):
    We tokenize inside `__getitem__` so the DataLoader's worker
    processes can parallelize tokenization across CPUs. If we
    pre-tokenized everything upfront, we'd need ~400× more disk space
    AND we'd lose the chance to use different tokenizers without
    re-preparing the data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class PhantomCodesDataset(Dataset):
    """Multi-label ICD-code classification dataset over our parquet splits.

    Args:
        parquet_path: Path to a parquet file produced by
            `phantom-codes prepare` (one of train/val/test). Must have
            columns: `input_fhir`, `input_text`, `gt_code`, `mode`,
            `resource_id`. Schema documented in `data/prepare.py`.
        tokenizer: Pre-loaded HuggingFace tokenizer (e.g.
            `AutoTokenizer.from_pretrained("microsoft/...")`).
        code_to_index: Mapping from ICD code string → integer label
            position (0 to n_codes-1). Codes not in the map are skipped
            silently — they don't contribute a label. Build this once
            from the train split's code distribution.
        max_seq_length: Token cap. Longer inputs get truncated.
            Padding to this length happens inside `__getitem__` (so
            every sample has identical shape and the DataLoader can
            stack them without dynamic padding logic).

    Example:
        >>> from transformers import AutoTokenizer
        >>> tok = AutoTokenizer.from_pretrained(model_id)
        >>> ds = PhantomCodesDataset(
        ...     "data/derived/conditions/train.parquet", tok,
        ...     code_to_index={"E11.9": 0, "I10": 1, ...},
        ...     max_seq_length=256,
        ... )
        >>> len(ds)
        24
        >>> sample = ds[0]
        >>> sample["input_ids"].shape, sample["labels"].shape
        (torch.Size([256]), torch.Size([50]))
    """

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        code_to_index: dict[str, int],
        max_seq_length: int = 256,
    ) -> None:
        # Load the entire parquet into memory. Our scale (low tens of
        # thousands of records) fits comfortably; if we ever scale up
        # we'd switch to lazy loading via PyArrow's iterators.
        self._df = pd.read_parquet(parquet_path)
        self._tokenizer = tokenizer
        self._code_to_index = code_to_index
        self._n_codes = len(code_to_index)
        self._max_seq_length = max_seq_length

    def __len__(self) -> int:
        # PyTorch needs to know how many examples exist so the
        # DataLoader can compute batch boundaries and per-epoch step
        # counts. This is one of two methods Dataset MUST implement.
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # The other required method. Returns a single example as a
        # dict of tensors. The DataLoader will call this many times,
        # then stack the per-key tensors along a new batch dimension.
        # Whatever keys we return here become keys in the batch dict.
        row = self._df.iloc[idx]

        # Build the text the model will see. Most degradation modes
        # already produce a plain `input_text`; only D1_full and
        # D2_no_code carry structured FHIR JSON we need to flatten.
        text = self._extract_text(row)

        # Tokenize. `padding="max_length"` pads every sample to
        # `max_seq_length` tokens — wastes a bit of compute on padding
        # but keeps batch shapes uniform. `truncation=True` cuts
        # samples longer than the cap. `return_tensors="pt"` produces
        # PyTorch tensors instead of plain Python lists.
        encoded = self._tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self._max_seq_length,
            return_tensors="pt",
        )

        # `encoded` is a BatchEncoding with leading batch dim 1.
        # We squeeze it out so each tensor here has shape
        # [seq_length] — DataLoader will add the batch dim back when
        # it stacks N samples together.
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Build the multi-hot label vector. For multi-label classification,
        # each output position is INDEPENDENT — we use sigmoid + BCE,
        # not softmax + cross-entropy. So a sample with label `E11.9`
        # has 1.0 in position `code_to_index["E11.9"]` and 0.0
        # everywhere else. Float dtype because BCEWithLogitsLoss
        # expects floats.
        labels = torch.zeros(self._n_codes, dtype=torch.float32)
        gt_code = str(row["gt_code"])
        if gt_code in self._code_to_index:
            labels[self._code_to_index[gt_code]] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @staticmethod
    def _extract_text(row: pd.Series) -> str:
        """Get the model-input text from a parquet row.

        D3_text_only and D4_abbreviated already store text in `input_text`.
        D1_full and D2_no_code store FHIR JSON in `input_fhir`; we
        flatten that to a string the tokenizer can chew on.
        """
        text = row.get("input_text")
        if isinstance(text, str) and text:
            return text

        fhir_json = row.get("input_fhir")
        if isinstance(fhir_json, str) and fhir_json:
            # Already JSON-string in the parquet (per `data/prepare.py`).
            # We could try to extract `code.text` and `coding[].display`
            # cleanly, but for v1 we just feed the whole JSON to the
            # tokenizer — PubMedBERT handles structured-ish text fine.
            try:
                obj = json.loads(fhir_json)
                # Prefer the canonical text fields if present;
                # otherwise dump the whole condition.
                code_section: dict[str, Any] = obj.get("code", {}) or {}
                if isinstance(code_section.get("text"), str):
                    return str(code_section["text"])
                for coding in code_section.get("coding") or []:
                    if isinstance(coding.get("display"), str):
                        return str(coding["display"])
                return fhir_json
            except (json.JSONDecodeError, TypeError):
                return fhir_json

        return ""


def build_code_index(parquet_path: str | Path, top_n: int) -> dict[str, int]:
    """Build a `code → integer index` map from the most-frequent codes.

    Build this ONCE from the training split, then pass the same map
    into the validation and test datasets so the label space is
    identical across splits. (If a code only appears in val/test, it
    just won't have a label — and the model can't predict it.)
    """
    df = pd.read_parquet(parquet_path)
    counts = df["gt_code"].value_counts().head(top_n)
    return {code: idx for idx, code in enumerate(counts.index.tolist())}
