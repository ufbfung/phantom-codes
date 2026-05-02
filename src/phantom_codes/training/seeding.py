"""Reproducible random number generators for training.

Why this matters: ML training is full of randomness — weight
initialization, dropout masks, data shuffling, sometimes even GPU
matrix-multiplication order. Two training runs with the same code,
data, and hyperparameters will produce different model weights unless
every random seed is locked.

For a research paper, this is a methodology requirement. A reviewer
asking "can you re-run the headline experiment and get the same
numbers?" needs the answer to be yes — bit-for-bit if possible, within
floating-point noise at minimum.

PyTorch alone doesn't fix this. There are *four* independent random
number generators that can affect a training run:

  1. Python's `random` module — used by some HuggingFace utilities,
     custom samplers, etc.
  2. NumPy's RNG — used by data-augmentation libraries, scikit-learn,
     pandas sampling
  3. PyTorch CPU RNG — used by `torch.randn`, dropout layers running
     on CPU, etc.
  4. PyTorch GPU/MPS RNG — used by dropout layers running on GPU/MPS

`seed_everything` initializes all four to the same value. Call it once
at the start of training, before any model construction or data
shuffling.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set Python, NumPy, and PyTorch (CPU + CUDA + MPS) seeds.

    Also sets `PYTHONHASHSEED` so dict iteration order is deterministic
    across Python invocations — important if any code path uses dict
    ordering (it shouldn't, but defense in depth).

    Note: full bitwise determinism on GPU/MPS additionally requires
    setting `torch.backends.cudnn.deterministic = True` (CUDA only) and
    accepting a meaningful speed penalty. We don't enable that here —
    floating-point determinism within the same hardware is enough for
    research reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # `torch.cuda.manual_seed_all` is safe to call even when CUDA isn't
    # available — it's a no-op. Same for the MPS equivalent below.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        # MPS doesn't expose a separate seed function, but `torch.manual_seed`
        # above propagates to the MPS backend. This branch is for clarity.
        pass
