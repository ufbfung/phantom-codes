# Models and Loss

> Third in the PyTorch primer series. Builds on
> [00-pytorch-foundations.md](00-pytorch-foundations.md) and
> [01-data-pipeline.md](01-data-pipeline.md).

## What "fine-tuning" actually is

We don't train PubMedBERT from scratch — we *fine-tune* it. The distinction:

- **Pre-training**: Microsoft trained PubMedBERT on millions of PubMed abstracts with a generic objective (predict masked words). Cost: thousands of GPU-hours. Output: a model that "understands" biomedical English.
- **Fine-tuning**: We start from those pre-trained weights and continue training on our task-specific data (clinical text → ICD codes). Cost: a few hours on a laptop. Output: a model that does *our* task, while keeping the biomedical knowledge.

Mechanically, fine-tuning is just regular training where the initial weights happen to be from a pre-trained checkpoint instead of random. The only special thing we add is a **classification head** — a fresh, randomly-initialized layer on top of the pre-trained encoder that maps the encoder's output to our task's output space.

That's exactly the architecture in [trainer.py:104](../../src/phantom_codes/training/trainer.py#L104):

```python
class PubMedBERTClassifier(torch.nn.Module):
    def __init__(self, base_model_id, n_codes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_id)  # pre-trained
        self.classifier = torch.nn.Linear(
            self.encoder.config.hidden_size, n_codes              # random
        )
```

## The HuggingFace `AutoModel` / `AutoTokenizer` pattern

HuggingFace's `transformers` library standardizes a "load any model by ID" pattern:

```python
from transformers import AutoModel, AutoTokenizer

model_id = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
encoder = AutoModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

Behind `AutoModel` is dispatching logic: it reads the `config.json` of the model, sees it's a BERT, and instantiates `BertModel`. Same model ID for tokenizer guarantees they speak the same vocabulary.

Two `AutoModel` variants worth knowing:

- **`AutoModel`** — returns the encoder *without* any task head. Good for "I'll add my own head."
- **`AutoModelForSequenceClassification`** — returns encoder + a built-in classification head. Convenient, but the head is opinionated (single-label, specific dropout/init). We use plain `AutoModel` + a hand-written `Linear` so we have full control.

## The [CLS] token trick

BERT-style transformers process a sequence of tokens and produce one hidden vector per token. So for input shape `[batch, seq_len]`, the encoder's `last_hidden_state` has shape `[batch, seq_len, hidden_size]`.

For classification, we need a single fixed-size vector per sample, not one per token. The standard trick: take the hidden vector at position 0, which is always the special `[CLS]` token that BERT was trained to use as a "sentence summary":

```python
outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
cls_embedding = outputs.last_hidden_state[:, 0]   # [batch, hidden_size]
logits = self.classifier(cls_embedding)            # [batch, n_codes]
```

Why this works: during pre-training, BERT's "next-sentence prediction" objective forced the [CLS] embedding to summarize the whole sentence. So even though no token "is" the sentence, [CLS] reliably aggregates information about it.

Alternatives: mean-pooling over all token embeddings, max-pooling, or attention-weighted pooling. They sometimes win small accuracy gains but [CLS] is the canonical choice and works well — start there, optimize later if numbers demand it.

## What "logits" means

A `logit` is the raw output of a linear layer, before any activation function. For a classifier:

- A *probability* lives in `[0, 1]`.
- A *logit* lives in `(-inf, +inf)`.

The two are related by the sigmoid function:

```python
prob = 1 / (1 + exp(-logit))
```

Why bother with logits? Two reasons:

1. **Numerical stability.** `sigmoid(very_negative_number)` underflows to 0 in float32; `log(0)` is `-inf`. Doing `BCE(sigmoid(logit), label)` naively can produce NaN gradients. The fused `BCEWithLogitsLoss` rearranges the math to avoid the underflow.
2. **One source of truth for "score."** Logits are unbounded and additive across batch/dimension; you can compare them, rank them, sum them. Probabilities are constrained to a simplex per row and don't compose as cleanly.

Convention in PyTorch: every layer outputs logits; activations are applied either by the loss function (training) or explicitly (inference).

## The loss: `BCEWithLogitsLoss`

For our multi-label problem:

```python
loss_fn = torch.nn.BCEWithLogitsLoss()
loss = loss_fn(logits, labels)   # logits: [batch, n_codes], labels: [batch, n_codes]
```

What this computes, per output position:

```
loss_per_position = label * -log(sigmoid(logit))
                  + (1 - label) * -log(1 - sigmoid(logit))
```

Read it as: "for each (sample, code) cell, treat it as an independent yes/no question, compute the binary cross-entropy of the model's belief vs. the truth, then average over all cells."

The independence is the whole point of *multi-label*. Softmax + cross-entropy would force the predictions to compete (sum to 1 across codes); we don't want that — a sample really can have two diagnoses.

### Why `_with_logits` matters

The non-fused alternative would be:

```python
probs = torch.sigmoid(logits)
loss = torch.nn.BCELoss()(probs, labels)
```

This is mathematically equivalent but numerically worse. `BCEWithLogitsLoss` uses the [log-sum-exp trick](https://en.wikipedia.org/wiki/LogSumExp) internally to stay stable across the full range of logits. Always prefer the `_with_logits` variant if it exists. The HuggingFace tutorials sometimes use plain `BCELoss`; that's a footgun.

## What about softmax + cross-entropy?

For *single-label* classification (only one of N classes is correct), use:

```python
loss_fn = torch.nn.CrossEntropyLoss()       # softmax + NLL fused
loss = loss_fn(logits, target_class_index)
```

Note the target shape: `[batch]` of integer class indices, not `[batch, n_classes]` of one-hot floats. PyTorch handles the one-hot expansion internally. This is why the documentation makes a distinction between `CrossEntropyLoss` (logits + integer indices) and `NLLLoss` (log-probs + integer indices).

For our project, multi-label is the right framing — we want the eval to be able to score top-K predictions and observe when multiple codes share probability mass. Single-label cross-entropy would hide that.

## Initialization matters more than you'd think

When you create `nn.Linear(768, 50)`, PyTorch initializes its weights with Kaiming-uniform by default. That's a sensible choice for ReLU networks but for a BERT classification head, the resulting initial logits can be weirdly large, producing huge gradients on the first batch and destabilizing the pre-trained encoder.

Two mitigations both used in our trainer:

1. **Linear warmup** (next primer) — start with a tiny learning rate and ramp up over the first few hundred steps.
2. **Weight decay** via AdamW — penalizes large weights, which keeps the head from running away.

These compound: warmup keeps the *updates* small, weight decay keeps the *weights* small. Both matter for fine-tuning stability.

## What's next

[03-training-loop.md](03-training-loop.md) puts the model together with the optimizer, the scheduler, and the validation/checkpointing logic — the full canonical training loop, walked through line by line.
