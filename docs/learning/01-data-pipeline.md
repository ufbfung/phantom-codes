# Data Pipeline: Dataset, DataLoader, Tokenization

> Second in the PyTorch primer series. Builds on
> [00-pytorch-foundations.md](00-pytorch-foundations.md).

## Why data is its own concern

Training a transformer is two loosely-coupled problems:

1. **Compute** — running the forward and backward pass on the GPU as fast as possible.
2. **Data movement** — feeding the GPU a steady stream of correctly-shaped batches so it never idles.

If you do (2) badly, your GPU sits at 30% utilization waiting for the next batch. PyTorch separates the two with `Dataset` and `DataLoader` precisely so each can be optimized independently.

## The `Dataset` protocol

A `torch.utils.data.Dataset` is any object that implements two methods:

```python
class MyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return ...               # how many examples exist?

    def __getitem__(self, idx: int):
        return ...               # give me the idx-th example
```

That's the whole interface. Look at [training/dataset.py](../../src/phantom_codes/training/dataset.py) — that's exactly what `PhantomCodesDataset` does:

- `__len__` returns `len(self._df)` — the parquet row count.
- `__getitem__(idx)` reads one row, tokenizes the text, builds the multi-hot label vector, and returns a dict of tensors.

A few non-obvious design choices:

### Why tokenize inside `__getitem__` instead of upfront?

We could pre-tokenize every row when `__init__` runs and store the token IDs. Two reasons we don't:

- **Disk + memory cost.** Pre-tokenized tensors of shape `[max_seq_length]` per row would be ~400× larger than the raw text strings. For the train split that's the difference between a few MB and a couple of GB.
- **DataLoader workers parallelize tokenization for free.** When you set `num_workers > 0`, the DataLoader spins up that many subprocesses, each calling `__getitem__` independently. Tokenization happens on multiple CPU cores in parallel while the GPU is busy on the previous batch. Pre-tokenizing kills this benefit.

### Why return a dict instead of a tuple?

The DataLoader will call `__getitem__(0)`, `__getitem__(1)`, etc. and **collate** the results — i.e., stack them into a batch. Default collation handles dicts of tensors automatically: each key's tensor gets stacked along a new batch dimension. So `{"input_ids": [seq], "labels": [n_codes]}` becomes `{"input_ids": [batch, seq], "labels": [batch, n_codes]}` with no extra code from us.

If we returned a tuple, we'd have to remember positional order everywhere downstream. Dicts are self-documenting.

## The `DataLoader` does the rest

You almost never write a DataLoader. You instantiate one:

```python
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,        # randomize order each epoch (train only)
    num_workers=4,       # parallel processes calling __getitem__
    pin_memory=True,     # speeds up CPU→GPU transfer on CUDA
)
for batch in loader:
    ...                  # batch is whatever __getitem__ returns, stacked
```

Things the DataLoader handles for free:

- Shuffling the index order each epoch.
- Calling `__getitem__` `batch_size` times.
- Stacking the per-sample tensors into a batch tensor.
- Spawning workers and routing the results back.
- Drop-last logic (so the trailing partial batch doesn't crash).

This is why our trainer's data setup is just three lines:

```python
train_ds = PhantomCodesDataset(...)
train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
```

`shuffle=True` for train means the batches are different every epoch — important because the model would otherwise overfit to the *order*, not just the data. `shuffle=False` for val so validation is deterministic across epochs and runs.

## Tokenization: text → integers

Transformers don't see text. They see integer token IDs. The tokenizer is the bridge:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")

encoded = tok(
    "Type 2 diabetes mellitus",
    padding="max_length",        # pad to max_length so all samples match
    truncation=True,             # cut anything longer than max_length
    max_length=32,
    return_tensors="pt",         # return PyTorch tensors (not lists)
)
encoded["input_ids"].shape       # torch.Size([1, 32])
encoded["attention_mask"].shape  # torch.Size([1, 32])
```

Two outputs to know:

- **`input_ids`**: the integer token IDs. Each token (word piece) maps to an index in the model's vocabulary. `[CLS]` (the special "summary" token) is at position 0; `[SEP]` ends the sequence; `[PAD]` fills the tail.
- **`attention_mask`**: 1s for real tokens, 0s for padding. Tells the model "ignore these positions when computing attention." Without this, padded tokens would influence the [CLS] embedding and ruin classification.

We use `padding="max_length"` (not `padding=True`) so every sample comes out at exactly `max_seq_length` tokens. This wastes a bit of compute on padded positions but means the DataLoader can stack everything naively. The smarter alternative is **dynamic padding** (pad each batch to its own longest sample), which a custom `collate_fn` would handle. We don't bother for v1 — our sequences are short and uniform enough that the savings would be small.

## Multi-hot labels for multi-label classification

Single-label classification (e.g., "is this image a cat or a dog?") uses softmax + cross-entropy and a label vector of shape `[1]` (the integer class index).

Multi-label classification (e.g., "which of these 50 ICD codes apply to this text?") uses sigmoid + binary cross-entropy and a label vector of shape `[n_codes]` with 1s in every applicable position. For our v1 dataset, exactly one code applies per sample, so the label vector is one-hot — but the model's output layer is still multi-label so it can in principle predict multiple codes.

This is what the dataset's `__getitem__` is doing here:

```python
labels = torch.zeros(self._n_codes, dtype=torch.float32)
gt_code = str(row["gt_code"])
if gt_code in self._code_to_index:
    labels[self._code_to_index[gt_code]] = 1.0
```

`float32` (not `int64`) because `BCEWithLogitsLoss` expects floating-point targets. The next primer ([02-models-and-loss.md](02-models-and-loss.md)) explains why.

## A quick sanity-check pattern

Whenever you build a new Dataset, always run this before plugging it into training:

```python
ds = MyDataset(...)
print(f"len: {len(ds)}")
sample = ds[0]
for k, v in sample.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
```

If shapes or dtypes look off, *fix it here* before adding the model on top — debugging shape mismatches at the loss step is much harder.

## What's next

[02-models-and-loss.md](02-models-and-loss.md) covers the actual model: `nn.Module`, `AutoModel` from HuggingFace, the [CLS]-token pattern, and why `BCEWithLogitsLoss` fuses sigmoid into the loss for stability.
