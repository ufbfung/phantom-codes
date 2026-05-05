# Hardware and Optimization

## Hardware

All training and validation runs execute on a 2020-vintage Apple
MacBook Pro with the M1 system-on-chip and 16 GB of unified memory,
running macOS 25.3.0. PyTorch 2.x's MPS (Metal Performance Shaders)
backend exposes the M1 GPU to CUDA-style tensor operations; we use
MPS for all forward and backward passes, with the CPU handling
DataLoader tokenization and disk I/O.

| Component | Specification |
|---|---|
| CPU | Apple M1 (8 cores: 4 performance + 4 efficiency) |
| GPU | Apple M1 8-core integrated GPU, accessed via PyTorch MPS |
| Memory | 16 GB unified (shared between CPU and GPU) |
| Storage | Internal NVMe SSD |
| Compute backend | PyTorch `torch.device("mps")` |
| Precision | float32 throughout (no mixed precision) |

This is intentionally modest hardware. The choice is a methodological
commitment, not a resource constraint: training a research-grade
clinical classifier on personal hardware demonstrates that the
contribution is reproducible by any researcher with PhysioNet
credentialing, without requiring cloud GPU access, institutional
compute, or budget approval. The same code path supports CUDA
(NVIDIA) and CPU backends and could be retargeted to a Vertex AI
Workbench instance or AWS EC2 GPU for faster wall-clock if a future
contributor preferred — at the cost of the data-residency story.

## Hardware-driven hyperparameter choices

Batch size 16 at maximum sequence length 128 uses ~5–6 GB of
unified memory and leaves headroom for normal laptop operation
during a ~15-hour training run. Sequence-length 128 truncates the
longest input modes (D1\_full / D2\_no\_code FHIR JSON payloads at
~200–400 tokens) but leaves the headline experiment's
D3\_text\_only / D4\_abbreviated inputs (10–50 tokens) untouched.
Steady-state throughput is ~1.85 iterations per second on M1 MPS,
yielding ~7 hours per epoch (train + validation) and ~15 hours
total wall-clock for an early-stopping run at epoch 2. An A100
cloud GPU would run ~3–5× faster at ~\$4–7 per training run if
the data-residency story permits.

## Optimization

We use **AdamW** [@Loshchilov2019] with decoupled weight decay
(0.01) and a peak learning rate of 2.0×10⁻⁵, the canonical
fine-tuning rate for BERT-family models [@Devlin2019]. The
learning rate follows a **linear-warmup, linear-decay** schedule:
ramped from zero to peak over 500 steps (~0.4% of total training),
then linearly decayed to zero over the remaining ~128,400 steps.
Warmup is essential — without it, the gradient signal from the
randomly-initialized classification head can destabilize the
pre-trained encoder in the first few hundred steps and cause
divergence [@Liu2020].

Training runs for up to **3 epochs** with **early stopping** on
validation loss (patience 2 epochs). The best checkpoint by
validation loss is persisted; we do not use the final-epoch
checkpoint, which may have begun to overfit. Aggregate per-epoch
metrics (training loss, validation loss, validation top-1
accuracy) are written to a JSON metrics file; we do not persist
per-record predictions during training, both to limit checkpoint
size and to preserve the compliance posture (the only per-record
predictions that ever get persisted are from the Synthea
evaluation matrix, which is freely shareable).
