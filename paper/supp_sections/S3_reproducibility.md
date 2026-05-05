# S3 — Reproducibility appendix and trained-model details

This supplement provides everything a reader needs to reproduce the
headline numbers in main §3 from a clean clone of the project
repository, plus the full architectural and training details for the
PubMedBERT classification arm (compressed to a single paragraph in
main §Methods). The reproduction recipe avoids any credentialed
(MIMIC) data — all numbers in main §3 come from the Synthea
evaluation cohort, which is openly redistributable. The trained
PubMedBERT classifier requires PhysioNet credentialing to retrain;
its inference behavior on the Synthea cohort is reproducible from the
published checkpoint hash if a credentialed reader trains their own
(we do not redistribute MIMIC-derivative weights).

## S3.1 — Trained-model architecture

We fine-tune a single transformer encoder with a randomly-initialized
linear classification head over the [CLS] token's final-layer hidden
state. This is the canonical BERT-style classification setup
[@Devlin2019]; we make three deliberate choices on top of it.

**Multi-label rather than single-label.** The classification head
emits one logit per code (50 logits total), trained with binary
cross-entropy on a sigmoid activation (`BCEWithLogitsLoss` for
numerical stability). While our v1 ground truth is one code per
condition, the multi-label formulation allows the model to in
principle predict multiple codes when comorbidities are documented in
the input text — a property we do not exploit in v1 but preserve for
v2's lab/medication extensions where multi-label is the norm.

**Top-N vocabulary cap.** Restricting the head to the 50 most-frequent
codes (rather than all 178 observed codes) trades coverage for
classifier signal density. The long tail of rare codes (≤10 instances
in the train split) cannot be reliably learned regardless of model
choice; including them would dilute the gradient signal on the
high-frequency codes that drive most clinical decisions. This is a
standard trade-off in extreme classification [@Chalkidis2020]; we err
toward the well-supported subset for v1.

**Single linear head, no MLP.** A two-layer MLP head with
non-linearity is sometimes used to give the head capacity to learn
task-specific combinations of encoder features. We use a single
linear layer for v1 because it is the standard baseline and because
the encoder's [CLS] embedding already aggregates sentence-level
information through 12 transformer layers. Adding capacity to the
head trades against training-set memorization risk; the cleaner head
minimizes confound.

## S3.2 — Choice of base encoder: PubMedBERT

We use **PubMedBERT-base-uncased-abstract-fulltext** [@Gu2021] as the
pre-trained encoder for three reasons: (a) domain-adaptive
pre-training (PubMedBERT is trained from scratch on PubMed abstracts
+ full-text articles, producing a tokenizer that treats biomedical
terms as single tokens rather than fragmented subwords [@Devlin2019;
@Gururangan2020]); (b) strong empirical performance on the BLURB
biomedical NLP benchmark [@Gu2021]; and (c) a computational footprint
(~110M parameters) compatible with local fine-tuning on consumer
Apple Silicon, which is the compliance-enabling property — any larger
encoder would have forced us onto cloud GPU infrastructure with
corresponding data-residency review for credentialed MIMIC data.
Alternative encoders (ClinicalBERT [@Alsentzer2019], BioBERT
[@Lee2020], BioLinkBERT [@Yasunaga2022], and larger biomedical models
including BiomedLM, GatorTron, BioMistral, and Med-PaLM via parameter-
efficient fine-tuning [@Hu2022; @Dettmers2023]) were considered and
either ruled out (ClinicalBERT — MIMIC-III pre-training contaminates
MIMIC-IV evaluation) or deferred to a v2 ablation arm.

## S3.3 — Hardware

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
compute, or budget approval.

**Hardware-driven hyperparameter choices.** Batch size 16 at maximum
sequence length 128 uses ~5–6 GB of unified memory and leaves
headroom for normal laptop operation during a ~15-hour training run.
Sequence-length 128 truncates the longest input modes (D1\_full /
D2\_no\_code FHIR JSON payloads at ~200–400 tokens) but leaves the
headline experiment's D3\_text\_only / D4\_abbreviated inputs (10–50
tokens) untouched. Steady-state throughput is ~1.85 iterations per
second on M1 MPS, yielding ~7 hours per epoch (train + validation)
and ~15 hours total wall-clock for an early-stopping run at epoch 2.

## S3.4 — Optimization

We use **AdamW** [@Loshchilov2019] with decoupled weight decay (0.01)
and a peak learning rate of 2.0×10⁻⁵, the canonical fine-tuning rate
for BERT-family models [@Devlin2019]. The learning rate follows a
**linear-warmup, linear-decay** schedule: ramped from zero to peak
over 500 steps (~0.4% of total training), then linearly decayed to
zero over the remaining ~128,400 steps. Warmup is essential — without
it, the gradient signal from the randomly-initialized classification
head can destabilize the pre-trained encoder in the first few hundred
steps and cause divergence [@Liu2020].

Training runs for up to **3 epochs** with **early stopping** on
validation loss (patience 2 epochs). The best checkpoint by
validation loss is persisted; we do not use the final-epoch
checkpoint, which may have begun to overfit. Aggregate per-epoch
metrics (training loss, validation loss, validation top-1 accuracy)
are written to a JSON metrics file; we do not persist per-record
predictions during training, both to limit checkpoint size and to
preserve the compliance posture.

## S3.5 — Training reproducibility

A single integer seed (`seed: 42`) is propagated to Python's
`random`, NumPy, and all PyTorch random number generators (CPU and
MPS) before model construction or data shuffling. Combined with the
deterministic stratified split (also seeded) and full pinning of
dependencies via `uv` lockfile, this is sufficient to reproduce
training outcomes within floating-point noise on the same hardware.
Bit-exact reproducibility on GPU/MPS additionally requires
`torch.use_deterministic_algorithms(True)`, which incurs a 10–30%
performance penalty; we do not enable it for v1.

The full set of training hyperparameters is captured in
`configs/training.yaml` and persisted into each saved checkpoint, so
inference code reads the configuration that produced the weights
rather than relying on a separately-tracked config file.

## S3.6 — Training convergence

Validation top-1 accuracy at convergence reached 0.992–0.993 across
three training epochs on the held-out MIMIC-IV validation split. This
high in-distribution value reflects the long-tail concentration of
the ACCESS-scope cohort (the top-50 codes cover the bulk of training
rows) combined with effective fine-tuning of a pre-trained biomedical
encoder. The same checkpoint evaluated on the Synthea
out-of-distribution cohort yields a substantially lower per-mode
top-1 accuracy (40–50% range, see main §3) — the gap between
in-distribution validation accuracy and out-of-distribution Synthea
accuracy is one of the deployment-relevant measurements this paper
surfaces.

## S3.7 — Reproduction sequence (commands)

The full reproduction recipe is documented in the project's
`BENCHMARK.md` reference file. The headline numbers for v1 were
produced by the following sequence (Synthea generation + preparation
steps deterministic given the pinned-SHA + seed combination).

```bash
# 1. Install dependencies
uv sync --extra dev

# 2. Clone + build Synthea v4.0.0 (pinned SHA in setup script)
./scripts/setup_synthea.sh

# 3. Generate the synthetic cohort (deterministic given seed=42)
./scripts/generate_synthea_cohort.sh

# 4. Build the inference dataset from generated bundles
uv run phantom-codes prepare-synthea

# 5. Run the headline evaluation (~24-36 hr wall clock; $80-300)
uv run phantom-codes evaluate \
    --models-set headline_set \
    --cohort benchmarks/synthetic_v1/conditions.ndjson \
    --max-records 500 --max-cost-usd 500

# 6. Generate the report tables
uv run phantom-codes report \
    --csv results/raw/headline_<utc>.csv \
    --pricing configs/pricing.yaml \
    --out results/summary/n125_run/
```

The `--max-records 500` argument applied to the long-format cohort
yields **125 unique resources × 4 degradation modes = 500 EvalRecord
items** (one cohort row per resource × mode tuple); for full-cohort
reproduction at n=500 unique resources, pass `--max-records 2000`
instead. This semantic is documented inline in BENCHMARK.md.

## S3.8 — Pinned dependencies and software versions

[TODO: capture the output of `uv export --format requirements-txt` at
the headline-run commit SHA for full reproducibility. Pin to specific
package versions so future readers get byte-identical behavior from
the wrapper code.]

## S3.9 — Deterministic configurations

The following configuration files fully specify the headline run:

- `configs/synthea.yaml` — Synthea generator config (seed,
  population\_size, disabled\_modules, FHIR R4 export setup)
- `configs/models.yaml#headline_set` — the 29 model configurations
  evaluated, including LLM model\_id strings (pinned dated snapshots
  where supported by the provider)
- `configs/pricing.yaml` — per-token pricing snapshot with
  `snapshot_date: "2026-05-03"`
- `data/synthea/snomed_to_icd10cm.json` — curated SNOMED CT to
  ICD-10-CM mapping used to inject ICD codings into Synthea
  Conditions at generation time

All four are committed at the headline-run commit SHA recorded in the
run manifest sidecar
(`results/raw/headline_n125_<utc>.manifest.yaml`).

## S3.10 — Synthea cohort manifest

The cohort generated by the recipe above is byte-identical to the one
used for the headline numbers, given the same Synthea pinned SHA
(`0185c09ea9d10a822c6f5f3ef9bdcbcbe960c813`, v4.0.0) and seed (`42`).
A SHA-256 checksum of `benchmarks/synthetic_v1/conditions.ndjson` is
recorded in the project's BENCHMARK.md and can be checked against a
reproducer's output to verify cohort byte-identity before evaluation.

## S3.11 — What's NOT reproducible without credentialed access

The PubMedBERT classifier's inference behavior on the Synthea cohort
requires either (a) a credentialed PhysioNet account to re-train the
classifier locally on MIMIC-IV-FHIR using the published training
script, or (b) acceptance of the trained-classifier numbers
as-published without local replication. We do not redistribute the
trained weights (a fine-tuned PubMedBERT checkpoint is considered
MIMIC-derivative material under PhysioNet's responsible-use policy)
[@PhysioNet2025].

The 24 LLM configurations and 4 non-LLM baselines are fully
reproducible without any credentialed-data access — they evaluate on
Synthea content alone and can be re-run by any reader with the
appropriate API keys.
