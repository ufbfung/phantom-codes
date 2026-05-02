# PyTorch Foundations

> A pedagogical primer written alongside the Phantom Codes PubMedBERT
> classifier. Read in order with the other four primers; each layers
> on the last.

## Why PyTorch (and what it actually is)

PyTorch is two things bolted together:

1. **A numerical array library** (`torch.Tensor`), structurally similar to NumPy, but with the crucial extra property that operations on tensors can be tracked and *differentiated automatically*.
2. **A neural-network framework** (`torch.nn`, `torch.optim`) built on top of (1).

You can — and we sometimes do — use just (1) without ever touching (2). A tensor is just a multi-dimensional array of numbers with a `dtype` (float32, int64, etc.) and a `device` (CPU, CUDA, MPS).

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2×2 float matrix
x.shape   # torch.Size([2, 2])
x.dtype   # torch.float32
x.device  # device(type='cpu')
```

The library you'll spend most of your time with is `torch.nn` (layers and models) and `torch.optim` (optimizers). The core training loop boils down to *autograd* — the automatic-differentiation engine — which is the next thing to understand.

## Autograd in one sentence

> When you call `loss.backward()`, PyTorch walks the computation graph it built behind your back and fills in `.grad` for every tensor that has `requires_grad=True`.

That's it. The rest is bookkeeping. Here's the smallest possible example:

```python
w = torch.tensor([2.0], requires_grad=True)
x = torch.tensor([3.0])
y = (w * x) ** 2          # y = (wx)^2 = 36
y.backward()              # compute dy/dw
w.grad                    # tensor([12.]) -- because dy/dw = 2*w*x^2 = 12
```

A few non-obvious consequences:

- **Gradients accumulate by default.** If you call `backward()` twice, `.grad` doubles. This is why every training loop starts with `optimizer.zero_grad()` — to clear stale grads from the previous step.
- **Autograd tracks the graph as you compute.** It doesn't pre-compile. This is what makes PyTorch feel "Pythonic" — you can use plain `if`/`for`/`while` inside `forward()` and autograd handles whichever path you took.
- **`.detach()` and `with torch.no_grad():` opt out of graph-building.** Both are critical for inference (where you don't need gradients) and for accumulating metrics (where you don't want a million tensors held alive).

In our trainer, you'll see `.item()` everywhere we record loss values — that's because `.item()` extracts a Python float and detaches from the graph. Skipping it is a classic memory leak.

## `nn.Module` — the unit of model code

A model in PyTorch is a class that subclasses `torch.nn.Module` and implements:

- `__init__` — declares the layers (which are themselves `nn.Module`s).
- `forward` — defines the computation when you call the model on an input.

You never call `forward` directly. You call the model itself; PyTorch routes through `forward` and adds bookkeeping (hooks, autograd registration, etc.):

```python
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)

model = Tiny()
out = model(torch.randn(2, 10))   # NOT model.forward(...)
out.shape                         # torch.Size([2, 3])
```

`Linear(10, 3)` is just a learned `y = xW^T + b` with `W` shape `[3, 10]` and `b` shape `[3]`. Internally it's two `nn.Parameter`s — these are tensors with `requires_grad=True` that PyTorch automatically registers when you assign them as attributes of an `nn.Module`. That registration is how `model.parameters()` knows what to give the optimizer.

Our `PubMedBERTClassifier` ([trainer.py:104](../../src/phantom_codes/training/trainer.py#L104)) is exactly this pattern: an encoder submodule plus a `Linear` head, wired together in `forward`.

## Devices: CPU, CUDA, MPS

Every tensor lives on a device. So does every `nn.Module`. They have to match — if you call `model(x)` with the model on GPU and `x` on CPU, PyTorch will complain.

The three relevant devices for this project:

| Device | Hardware | When to use |
|--------|----------|-------------|
| `cpu`  | Any machine | Tests, small models, debugging |
| `cuda` | NVIDIA GPU | Vertex AI / cloud training (not used in v1) |
| `mps`  | Apple Silicon (M-series) GPU | Local training on the M1 MacBook Pro |

Moving things to a device is `.to(device)`:

```python
device = torch.device("mps")  # or "cuda" or "cpu"
model = model.to(device)
batch = {k: v.to(device) for k, v in batch.items()}
```

We centralize this in [training/devices.py](../../src/phantom_codes/training/devices.py) so the rest of the code never has to ask "what device am I on?". The priority order is MPS → CUDA → CPU, which lines up with where this project actually runs.

### MPS quirks worth knowing

- MPS is younger than CUDA in PyTorch and has occasional gaps. If an op isn't supported, you'll see an error like `NotImplementedError: ... aten::...`. Workaround: `PYTORCH_ENABLE_MPS_FALLBACK=1` makes those ops silently run on CPU.
- MPS doesn't support `float64` for many ops. We use `float32` everywhere (the default).
- Memory is shared with system RAM on Apple Silicon — there's no separate VRAM. So OOM behavior is "your whole machine slows down" rather than "CUDA error: out of memory".

## Where to go from there

You now know enough to read [trainer.py](../../src/phantom_codes/training/trainer.py) and follow what the layers, optimizer, and `loss.backward()` calls are doing. The next primer ([01-data-pipeline.md](01-data-pipeline.md)) covers the *other* half of training: getting batches of data into the model efficiently.
