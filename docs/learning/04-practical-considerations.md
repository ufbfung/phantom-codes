# Practical Considerations

> Final primer in the series. The previous four were about what
> training *is*; this one is about what goes wrong, what to look at,
> and how to read the numbers.

## Hyperparameter intuition

You will spend more time setting hyperparameters than writing model code. A working starting set for BERT fine-tuning:

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| `learning_rate` | `2e-5` | Standard for BERT-family fine-tuning. Higher destabilizes; lower wastes time. Try `5e-5` if loss isn't moving. |
| `batch_size` | `8`–`32` | Bigger = smoother gradients but more memory. M1 MPS comfortably handles 8 at seq_length=256; bigger may swap. |
| `max_seq_length` | `128`–`512` | Memory and compute both scale O(seq²) thanks to attention. Pick the smallest that covers your typical input. |
| `num_epochs` | `3`–`10` | More epochs ≠ better; watch val loss. Most runs converge in 3–5. |
| `warmup_steps` | `100`–`1000` | About 10% of total training steps. Less if your dataset is tiny. |
| `weight_decay` | `0.01` | AdamW default; rarely needs tuning. Bigger penalizes bigger weights more. |
| `early_stopping_patience` | `2` | Two epochs of no val-loss improvement = quit. Set to `None` to disable. |

These are starting points, not magic. The classic advice: keep everything fixed and change one thing at a time. If you change three, you won't know which one moved the metric.

### When to tune what

- **Loss not decreasing at all** → LR too low, or warmup too long, or data pipeline broken (check shapes!).
- **Loss decreasing then exploding to NaN** → LR too high, no warmup, or numerical instability somewhere.
- **Train loss low, val loss high** → overfitting; reduce epochs, or add dropout, or get more data.
- **Train loss flat after a few epochs** → underfitting; bigger model, or more epochs, or higher LR.
- **Val accuracy high but val loss high** → confident on the easy ones, very wrong on the hard ones. This is a sign your loss landscape isn't smooth — try a smaller LR.

## Watching the numbers during a run

What our trainer prints each epoch:

```
[train] epoch=0 train_loss=0.6234 val_loss=0.5891 val_top1=0.412 elapsed=87s
[train] epoch=1 train_loss=0.4012 val_loss=0.4334 val_top1=0.587 elapsed=84s
[train] epoch=2 train_loss=0.2891 val_loss=0.4102 val_top1=0.621 elapsed=85s
[train] epoch=3 train_loss=0.1934 val_loss=0.4421 val_top1=0.618 elapsed=86s
                                              ^
                                              val started rising — overfit
```

What "good" looks like:

- Train loss decreases monotonically.
- Val loss decreases for a few epochs, then either plateaus or rises.
- Val accuracy increases roughly inversely to val loss.
- Per-epoch time is roughly constant (if it's growing, you have a memory leak).

What's a red flag:

- Train and val loss both stuck at the initial value → forward pass not connected to backward, or LR is 0.
- Train loss plummets but val loss is flat → severe overfitting. Probably your dataset is tiny.
- Loss = NaN after step 1 → you have a numerical bug. Check for `inf` in inputs, `log(0)` somewhere, or a layer that's producing huge outputs.

## MPS-specific quirks

Apple Silicon's GPU is ~3× slower than a high-end NVIDIA GPU but free, local, and PHI-safe. Things to know:

### Performance

- **First step is slow.** PyTorch compiles MPS kernels lazily on first use. Don't benchmark the first step — measure the steady-state from step 5+.
- **Mixed precision is bf16, not fp16.** The `accelerate` library will pick this if you ask for mixed precision; it's a small speed and memory win.
- **Large batch sizes hit shared-memory limits before VRAM limits.** On a 16GB M1, batch=32 at seq_length=512 will start paging to disk. Watch Activity Monitor's memory pressure.

### Reliability

- **Some ops aren't implemented.** When you hit `NotImplementedError: ... aten::...`, set `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU for those ops. It'll be slower but it'll run.
- **Nondeterminism is worse than CUDA.** Seeding helps but doesn't fully fix it. Don't expect bit-exact runs.
- **Save checkpoints to disk, not GPU memory.** `torch.save(model.state_dict(), path)` automatically copies tensors to CPU first. Don't try to `pickle.dump(model, ...)` — that captures the device, and reloading on a different machine fails.

### Diagnostic commands

```bash
# Is MPS available?
python -c "import torch; print(torch.backends.mps.is_available())"

# How much GPU memory is in use? (rough — Activity Monitor is more accurate)
python -c "import torch; print(torch.mps.current_allocated_memory() / 1e9, 'GB')"

# Force everything to CPU for debugging:
CUDA_VISIBLE_DEVICES="" PYTORCH_MPS_HIGH_WATERMARK_RATIO=0 python ...
```

## Debugging when training silently underperforms

The hardest bugs in ML aren't crashes — they're "the loss goes down and the metrics look fine but the model is secretly bad." A checklist for when you suspect something is off:

### 1. Check that your model is actually learning

Run with batch_size=1 on a single sample for many steps. The loss should drive to ~0. If it doesn't, the model literally cannot fit one example — your forward pass, loss, or gradient flow is broken.

```python
batch = next(iter(train_loader))
for step in range(200):
    optimizer.zero_grad()
    loss = loss_fn(model(**batch), batch["labels"])
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"step={step} loss={loss.item():.4f}")
```

If loss decreases → forward+backward are wired correctly; problem is elsewhere.
If loss is flat → somewhere in the chain, gradients aren't flowing. Common causes: `with torch.no_grad():` accidentally wrapping training, parameters with `requires_grad=False`, or a layer like `argmax` that has no gradient.

### 2. Check shapes at every boundary

Add prints (or breakpoints) at the entry and exit of each method, printing `tensor.shape`. ML bugs are very often "this should be `[batch, n_codes]` but it's `[n_codes, batch]`" or similar. Always check.

### 3. Sanity-check the loss value against random

For multi-label BCE with `n_codes=50` and `~1` true label per sample, a *random* model should produce loss around `~0.69` (ln 2, the BCE of a 50/50 guess on each binary output). If your initial loss is wildly different from this, something is wrong with your label encoding or model output.

### 4. Look at predictions, not just metrics

Even with our compliance constraints (no per-record output from MIMIC runs), you can sanity-check on synthetic fixtures and inspect:

- What are the top-K predicted codes for a known sample?
- Are predictions confidently correct, weakly correct, or randomly all over the place?
- Is the model always predicting the same most-frequent code? (That's a sign your loss is dominated by class imbalance.)

For real-MIMIC runs, this kind of inspection has to happen *locally* without surfacing the predictions. The eval matrix on Synthea data is where deep prediction analysis lives.

### 5. Compare against a baseline

If your fine-tuned model isn't beating TF-IDF + logistic regression, something is very wrong. Modern transformers should crush classical baselines on text classification. Use the [baselines](../../src/phantom_codes/models/baselines.py) as a sanity check — if the trained model isn't 5–10 points better on top-K accuracy, debug it before trusting it.

## Resource expectations on M1

From a quick benchmark on the M1 MacBook Pro 16GB:

- **Throughput**: ~1 step/second at batch=8, seq=256, for PubMedBERT-base.
- **Per-epoch time** (~10K train samples): ~20 minutes.
- **Total training** (5 epochs + early stopping): 1–2 hours.
- **Disk per checkpoint**: ~440MB (PubMedBERT-base + classifier head + config).

These are workable. For a real run on the full MIMIC train split (a few hundred thousand condition records), expect 4–10 hours total — overnight, not multi-day.

## Compliance reminders for local training

The [trainer.py](../../src/phantom_codes/training/trainer.py) module sets defensive env vars at import time so accidentally importing wandb/mlflow/comet won't sync metrics to the cloud. But that's a defense, not a license — the rules in [CLAUDE.md](../../CLAUDE.md) still apply:

- **Don't add cloud telemetry hooks.** No `wandb.init(mode="online")`, no `mlflow.set_tracking_uri("https://...")`, no Comet, no LangSmith.
- **Don't push checkpoints to HuggingFace Hub.** Trained weights are MIMIC-derivative.
- **Don't write per-record predictions to disk during training.** Aggregate metrics only. Per-record prediction storage happens in the eval matrix on Synthea, where the data is freely shareable.
- **Don't print sample predictions to stdout during a real-MIMIC run.** That output ends up in shell scrollback, terminal logs, IDE history — all places MIMIC content is not allowed.

The trainer module is structured so the default behavior is compliant; the failure modes listed above all require you to *add* something, not remove a guard. If something feels like it requires going around a guardrail, ask first.

## What to read next

You've now got the conceptual scaffolding to work productively in [training/](../../src/phantom_codes/training/) and [models/classifier.py](../../src/phantom_codes/models/classifier.py). For hands-on practice, the demo script at [scripts/demo_minimal_training.py](../../scripts/demo_minimal_training.py) runs the full pipeline on synthetic fixtures in under 60 seconds. Modify it, break it, fix it. That's how the concepts above turn into intuition.
