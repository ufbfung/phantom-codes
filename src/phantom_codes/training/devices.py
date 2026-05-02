"""Pick the best available PyTorch device.

PyTorch supports multiple "device" backends — different chips that can
hold tensors and run computation:

  - **CPU**: every machine has this; slowest for matrix-heavy work but
    always available
  - **CUDA**: NVIDIA GPUs (data-center machines, RTX/GTX desktops)
  - **MPS**: Apple Silicon GPU (M1/M2/M3/M4 Macs) via Apple's Metal
    Performance Shaders backend

Why this matters: a tensor on the wrong device can't interoperate with
a tensor on another device. You move tensors with `.to(device)` — and
the model and data must end up on the same device before forward pass.
We centralize the device pick here so every script makes the same
choice.

Auto-detection priority: MPS → CUDA → CPU. MPS comes first because we
explicitly target Apple Silicon for v1; if you're running on an NVIDIA
workstation instead, CUDA wins.
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the best available `torch.device`.

    Priority: MPS (Apple Silicon) → CUDA (NVIDIA) → CPU (fallback).
    The choice is silent — the caller can `.type` the result if it
    wants to log which backend was picked.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
