# The Training Loop

> Fourth in the PyTorch primer series. Builds on the previous three.

## The four-line kernel

Every PyTorch training loop, no matter how complex, has the same kernel:

```python
optimizer.zero_grad()                    # 1. clear stale gradients
logits = model(**batch)                  # 2. forward pass
loss = loss_fn(logits, batch["labels"])  # 3. compute loss
loss.backward()                          # 4. populate .grad on every parameter
optimizer.step()                         # 5. update weights using .grad
```

(Five lines if you count `zero_grad`, but conceptually steps 1+5 belong to the optimizer and 2+3+4 are the math.)

Look at [trainer.py:286-292](../../src/phantom_codes/training/trainer.py#L286-L292) — that's literally these five lines, plus a `scheduler.step()` for the learning rate. Every other line in the file is *bookkeeping*: moving data to the GPU, accumulating loss for reporting, validation, checkpointing.

## Why `zero_grad`?

Recall from the foundations primer: `loss.backward()` *accumulates* into `.grad`. If you don't clear before the next batch, you'd be summing gradients from multiple batches, which gives the wrong update direction.

This accumulation behavior is intentional — it lets you simulate larger batch sizes by skipping `zero_grad` for a few steps (gradient accumulation). But for plain training, every step starts with `zero_grad`.

## The optimizer

An optimizer holds a reference to the model's parameters and knows how to update them given their `.grad`:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01,
)
```

`model.parameters()` is a generator over every `nn.Parameter` registered in the model — including the encoder's weights (110M of them) and the classifier head's weights (768 × 50 = 38K). The optimizer treats them all uniformly: "here's a big bag of tensors with gradients, please update them."

### Why AdamW for transformer fine-tuning?

A few competitors and why we don't use them:

- **SGD** — simple, predictable, but slow to converge for transformers. Used to be standard for vision; transformers prefer adaptive methods.
- **Adam** — adaptive learning rates per parameter; good for transformers, but its L2 regularization interacts badly with the adaptive scaling. Specifically: L2 ends up applying *more* decay to rarely-updated parameters, which is the opposite of what you want.
- **AdamW** ([paper](https://arxiv.org/abs/1711.05101)) — Adam with the weight decay decoupled from the gradient update. Decay is applied directly to the weights, independent of the adaptive scaling. This single fix made transformer fine-tuning much more reliable.

Default learning rate for fine-tuning BERT-family models: **2e-5 to 5e-5**. Going higher destabilizes the pre-trained encoder; going lower wastes time.

## The learning-rate scheduler

The scheduler dynamically changes the learning rate during training. The most common pattern for transformer fine-tuning is **linear warmup + linear decay**:

```python
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps,
)
```

What this looks like:

```
LR
 ^
 |    /\___
 |   /     \___
 |  /          \___
 | /               \___
 |/_____________________\___> step
 0   100              total
     ^^^               ^^^
     warmup            decay
```

- **Warmup** (steps 0–100): LR ramps linearly from 0 to the target (`2e-5`). This protects the pre-trained encoder from the wild gradients of the freshly-initialized classification head.
- **Decay** (steps 100–total): LR ramps linearly back down to 0. This stabilizes late training when the model is close to convergence — large updates would just oscillate around the optimum.

You call `scheduler.step()` *after* `optimizer.step()` every batch, and the scheduler bumps the optimizer's LR for the next iteration. Without warmup, BERT fine-tuning often diverges in the first few hundred steps. It's not optional.

## Validation: `model.eval()` and `torch.no_grad()`

After each epoch, we run the model on the validation split to track generalization. Two flags matter:

```python
model.eval()              # disables dropout, freezes batch-norm stats
with torch.no_grad():     # disables autograd graph-building
    for batch in val_loader:
        logits = model(...)
        ...
```

Why both:

- **`model.eval()`** changes layer behavior. Dropout, batch norm, and layer norm have different forward-pass logic in train vs. eval mode. Forgetting this is a common bug — your val loss will be subtly wrong (often too low, because dropout in train mode adds noise that reduces measured train loss).
- **`torch.no_grad()`** tells autograd "don't bother building a computation graph, we won't backprop." Saves significant memory (no need to keep activations around) and a bit of time (no graph bookkeeping).

After validation, switch back: `model.train()` re-enables dropout for the next training epoch. (`with torch.no_grad():` exits its scope automatically.)

## Computing accuracy alongside loss

Loss is what the optimizer cares about. Humans care about accuracy. They diverge — a model can have lower loss but worse accuracy if it's confidently wrong on a few examples. Always log both.

For our multi-label setup, "top-1 accuracy" means: did the highest-logit code match the ground-truth code?

```python
pred_idx = logits.argmax(dim=-1)        # [batch] -- index of the max logit per sample
target_idx = batch["labels"].argmax(dim=-1)  # [batch] -- index of the 1.0 in the label
n_correct += int((pred_idx == target_idx).sum().item())
n_total   += int(batch["labels"].size(0))
```

This is per-sample, computed on the GPU (`.argmax`, `==`, `.sum()` all run on tensors), then accumulated as Python ints once we've squeezed out the per-batch number with `.item()`.

## Early stopping

If validation loss stops improving, keep going? Probably not — you're starting to memorize the train set. Early stopping watches for this:

```python
if avg_val_loss < result.best_val_loss:
    result.best_val_loss = avg_val_loss
    epochs_without_improvement = 0
    save_checkpoint(...)
else:
    epochs_without_improvement += 1

if epochs_without_improvement >= patience:
    break
```

`patience=2` means "give it two epochs of no improvement before quitting." Higher patience = more chances to escape a temporary plateau, at the cost of more wasted compute.

The convention is to checkpoint **only when val loss improves**, then load the best-checkpoint at the end. That way, even if you train past the optimum, the saved model is the one with the best generalization seen so far.

## Checkpointing

A "checkpoint" is the serialized state needed to reconstruct the trained model. The minimal contents:

```python
torch.save({
    "model_state_dict": model.state_dict(),    # all weights and biases
    "code_to_index": code_to_index,            # the label-space definition
    "config": dict(...),                       # hyperparameters used
    "epoch": epoch,
    "val_loss": avg_val_loss,
}, "checkpoint.pt")
```

What `state_dict()` returns: a Python dict mapping parameter names (`"encoder.layer.0.attention.weight"` etc.) to their current tensor values. Saving the dict is enough — the architecture is implicit in the code, you just need the values.

### Why save more than just `state_dict`

A bare `state_dict` is useless without:

- The **architecture** that produced it (so you can recreate the empty model and `load_state_dict` into it).
- The **code-to-index mapping** (so the inference code knows what each output position means).
- The **tokenizer config** (so you tokenize new inputs the same way).

We pack all of this into one file in [trainer.py:349-358](../../src/phantom_codes/training/trainer.py#L349-L358). The inference wrapper [classifier.py:75](../../src/phantom_codes/models/classifier.py#L75) reads everything it needs from that one file.

## Reproducibility: seeding

Three sources of randomness in training:

- Python's `random` (used by some HuggingFace components).
- NumPy's `np.random` (used by some data utilities).
- PyTorch's RNGs — separate streams for CPU, CUDA, MPS.

To reproduce a run, seed all three. [training/seeding.py](../../src/phantom_codes/training/seeding.py) handles this:

```python
def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

Call this *first* in your training run, before constructing the model (so weight initialization is reproducible) and before constructing the DataLoader (so shuffle order is reproducible).

Caveat: even with seeding, MPS and CUDA ops aren't 100% deterministic by default — some GPU kernels use atomic adds whose order depends on scheduling. For perfectly bit-exact reproducibility you'd also need `torch.use_deterministic_algorithms(True)` and accept the 10-30% performance hit. We don't bother for this project; the seeding gets us "the same numbers within noise" which is enough for research.

## Putting it together

Read [trainer.py](../../src/phantom_codes/training/trainer.py) end-to-end now — every piece should be familiar:

- Lines 211–218: device + tokenizer + vocab setup.
- Lines 221–232: datasets and dataloaders.
- Lines 235–261: model + loss + optimizer + scheduler.
- Lines 264–268: output-directory setup.
- Lines 273–298: train phase (the four-line kernel + LR schedule).
- Lines 305–323: validation phase (no_grad + accuracy).
- Lines 326–342: metrics recording and aggregate-only logging.
- Lines 345–359: best-checkpoint persistence.
- Lines 364–372: early-stopping check.

The next primer ([04-practical-considerations.md](04-practical-considerations.md)) covers the gotchas you'll only learn the hard way: hyperparameter intuition, MPS-specific quirks, and what to look at when training silently underperforms.
