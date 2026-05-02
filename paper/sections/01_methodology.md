# Methods: Trained-Model Training and Hardware

> **Status:** Draft v0 (2026-05-02). Numbers from the headline training run
> are filled in inline as `[TBD: from headline run, expected 2026-05-03]`.
> The methodology framing, model-choice rationale, and hardware
> description are settled. This section sits between the introduction
> (frames the research question) and the results (reports the numbers).

---

## Data and policy compliance

Our trained models are fine-tuned on MIMIC-IV-FHIR v2.1
[@Bennett2024; @Bennett2023; @Johnson2023]
under PhysioNet credentialed access. To remain compliant with PhysioNet's
responsible-LLM-use policy [@PhysioNet2025], which prohibits sending
credentialed data through third-party APIs, all training and validation
runs execute entirely on the corresponding author's own hardware (Apple
M1 MacBook Pro, see § Hardware). MIMIC content does not traverse any
network beyond the initial PhysioNet download, is not committed to any
version-control system, and is not used as input to any cloud service —
including telemetry libraries (wandb, mlflow, comet) which the training
module disables defensively at import time via environment variables.

The headline evaluation against frontier LLMs runs against
Synthea-generated FHIR Bundles [@Walonoski2018], a freely-redistributable
synthetic patient dataset with no real patient information. The
trained classifier is evaluated on the same Synthea inputs as the LLMs,
providing both compliance-by-construction and an out-of-distribution
generalization test (Synthea's clinical text differs systematically
from MIMIC's source notes).

## Cohort and label space

We restrict the cohort to ICD-10-CM codes in the CMS ACCESS Model FHIR
Implementation Guide v0.9.6 [@CMS2026], comprising two disease groups:

- **CKM** (cardiometabolic): diabetes (E08/E09/E11/E13), atherosclerotic
  cardiovascular disease, chronic kidney disease stage 3
- **eCKM** (extended cardiometabolic): hypertension, dyslipidemia,
  prediabetes, obesity

The MIMIC-IV-FHIR v2.1 source contains conditions coded under both
ICD-9-CM and ICD-10-CM, using MIMIC-namespaced FHIR `CodeSystem` URIs
(`http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-diagnosis-icd10`)
rather than the canonical HL7 URIs, and using undotted code formats
(`E1110` rather than `E11.10`). We normalize both at the extraction
boundary: codes are converted to the canonical dotted form (decimal point
inserted after position 3 for codes longer than three characters), and
system URIs are mapped to the canonical `http://hl7.org/fhir/sid/icd-10-cm`
so that the rest of the pipeline — scope filter, validator, evaluation
matrix — operates on a single canonical representation regardless of
source.

After ICD-10-CM filtering and ACCESS-scope filtering, the cohort
comprises 245,575 unique conditions distributed 70/10/20 across train,
validation, and test splits by stratified sampling on `resource_id`
(ensuring all four degradation modes of a single condition land in the
same split). 178 unique ICD-10-CM codes appear in the train split. Each
condition is materialized into four rows — one per degradation mode
(D1_full, D2_no_code, D3_text_only, D4_abbreviated) — yielding 687,552
train rows, 98,272 validation rows, and 196,476 test rows. The
classification head is sized for the top-50 most frequent codes, which
together cover the head of the distribution where supervised signal is
densest.

## Trained-model architecture

We fine-tune a single transformer encoder with a randomly-initialized
linear classification head over the [CLS] token's final-layer
hidden state. This is the canonical BERT-style classification setup
[@Devlin2019]; we make three deliberate choices on top of it.

**Multi-label rather than single-label.** The classification head emits
one logit per code (50 logits total), trained with binary cross-entropy
on a sigmoid activation (`BCEWithLogitsLoss` for numerical stability).
While our v1 ground truth is one code per condition, the multi-label
formulation allows the model to in principle predict multiple codes
when comorbidities are documented in the input text — a property we
do not exploit in v1 but preserve for v2's lab/medication extensions
where multi-label is the norm.

**Top-N vocabulary cap.** Restricting the head to the 50 most-frequent
codes (rather than all 178 observed codes) trades coverage for
classifier signal density. The long tail of rare codes (≤10 instances
in the train split) cannot be reliably learned regardless of model
choice; including them would dilute the gradient signal on the
high-frequency codes that drive most clinical decisions. This is a
standard trade-off in extreme classification [@Chalkidis2020]; we
err toward the well-supported subset for v1.

**Single linear head, no MLP.** A two-layer MLP head with non-linearity
is sometimes used to give the head capacity to learn task-specific
combinations of encoder features. We use a single linear layer for
v1 because it is the standard baseline and because the encoder's
[CLS] embedding already aggregates sentence-level information through
12 transformer layers. Adding capacity to the head trades against
training-set memorization risk; the cleaner head minimizes confound.

## Choice of base encoder: PubMedBERT

We use **PubMedBERT-base-uncased-abstract-fulltext** [@Gu2021] as
the pre-trained encoder. The choice is motivated by three properties.

**Domain-adaptive pre-training.** Standard BERT [@Devlin2019] is
pre-trained on Wikipedia and BookCorpus — corpora that contain little
biomedical text. The "Don't Stop Pretraining" line of work [Gururangan
2020] established that continued pre-training on in-domain text
improves downstream task performance, but PubMedBERT goes further by
pre-training *from scratch* on biomedical text (PubMed abstracts +
full-text articles). This produces a tokenizer whose vocabulary
treats biomedical terms (`diabetes`, `myocardial`, `hypertension`,
`hyperlipidemia`) as single tokens rather than fragmented subwords,
and an encoder whose attention patterns reflect biomedical syntax
rather than general English.

**Empirical performance on the BLURB benchmark.** On the BLURB suite
of 13 biomedical NLP tasks, PubMedBERT outperforms BioBERT,
SciBERT, and BlueBERT on the majority of tasks [@Gu2021], including
sentence-level classification tasks structurally similar to ours.

**Computational footprint compatible with local hardware.** At ~110M
parameters, PubMedBERT-base fits comfortably in 16 GB of unified
memory at our chosen batch size and sequence length, allowing the
entire training run to execute on a personal MacBook Pro (see §
Hardware). This is a critical compliance enabler: any larger encoder
would have forced us onto cloud GPU infrastructure with corresponding
data-residency review for credentialed data.

### Alternatives considered and rejected

**ClinicalBERT** [@Alsentzer2019] is an obvious candidate — the model
is pre-trained on MIMIC-III clinical notes and would be expected to
transfer well to MIMIC-IV. We rule it out for **data contamination**:
MIMIC-III's patient cohort partially overlaps with MIMIC-IV's, and
using a model that has seen MIMIC-III text during pre-training to
classify MIMIC-IV conditions would produce optimistic numbers that
do not generalize to the Synthea evaluation. PubMedBERT, pre-trained
exclusively on PubMed, has no such overlap.

**BioBERT** [@Lee2020] is a defensible alternative — also pre-trained
on PubMed (abstracts only, not full text). It is older than PubMedBERT
and underperforms it on BLURB; we list it as a sensitivity-analysis
candidate for v2 but do not include it in v1's headline run.

**BioLinkBERT** [@Yasunaga2022] is pre-trained on linked documents
(citation graphs from PubMed) and shows gains on multi-hop reasoning
tasks. Our task is single-hop (one input → one or more output codes);
the additional pre-training signal is not obviously relevant. Worth
ablating in v2 if results suggest the linear-head bottleneck is not
the limiting factor.

**Larger biomedical models** — BiomedLM (2.7 B parameters), GatorTron
(8.9 B), Med-PaLM (540 B), BioMistral (7 B) — would likely outperform
PubMedBERT-base but cannot be fine-tuned end-to-end on our hardware.
Parameter-efficient fine-tuning (LoRA [@Hu2022], QLoRA [@Dettmers2023]) would close the hardware gap and is on the v2 roadmap, both
as a stronger trained-model baseline and as a "fine-tuned LLM" rung
on the spectrum between zero-shot frontier LLMs and a fully-trained
classifier head.

**Generative LLMs (Claude, GPT, Gemini)** are present in our
evaluation matrix but as the *zero-shot, constrained, and RAG arms*
of the comparison, not as fine-tuned trained models. The trained
classifier head is the discriminative-model anchor of the matrix
against which LLM behavior under degradation is contrasted. Comparing
the same task on the same inputs across both architectures is the
core comparison the paper makes possible.

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

### Hardware-driven hyperparameter choices

The two hyperparameters that interact most strongly with hardware are
**batch size** and **maximum sequence length**, both of which
contribute to per-step memory consumption.

- **Batch size 16** at sequence length 128 uses approximately 5–6 GB of
  unified memory for model weights, activations, and gradients,
  leaving ~10 GB for the operating system, browser, and editor without
  triggering memory swap. Larger batches (32, 64) would likely fit
  but leave little headroom for normal laptop use during the
  ~15-hour training run.
- **Maximum sequence length 128** truncates the longest input modes
  (D1_full and D2_no_code, which carry FHIR JSON payloads of ~200–400
  tokens) but leaves the headline-experiment modes (D3_text_only and
  D4_abbreviated, typically 10–50 tokens) untouched. Transformer
  self-attention is *O(n²)* in sequence length, so dropping from 256
  to 128 tokens approximately quarters the per-step compute. We
  document this in the limitations section: D1/D2 results may
  underestimate the contribution of structured FHIR fields relative
  to a longer-context training run.

### Training throughput

Steady-state throughput on the configured hardware is **1.85 iterations
per second** (one iteration = one forward + backward + optimizer
step on a batch of 16), measured after the first ~100 steps to
exclude PyTorch's MPS-backend kernel-compilation warmup. This gives:

| Phase | Steps | Wall-clock |
|---|---|---|
| One training epoch | 42,972 | ~6.5 hours |
| One validation pass | 6,142 | ~50 minutes |
| Per epoch (train + val) | — | ~7.3 hours |
| Three epochs (full configured run) | — | ~22 hours |
| With early stopping at epoch 2 | — | ~15 hours |

For comparison, the same configuration on a single NVIDIA A100 GPU
would run approximately 3-5× faster (estimated; not measured) at
roughly $4-7 of cloud cost. The local M1 run completes overnight at
zero marginal cost, with the full data-residency guarantee preserved.

## Optimization

We use **AdamW** [@Loshchilov2019] with decoupled weight decay (0.01)
and a peak learning rate of 2.0×10⁻⁵, the canonical fine-tuning rate
for BERT-family models [@Devlin2019]. The learning rate follows a
**linear-warmup, linear-decay** schedule: ramped from zero to peak
over 500 steps (~0.4% of total training), then linearly decayed to
zero over the remaining ~128,400 steps. Warmup is essential — without
it, the gradient signal from the randomly-initialized classification
head can destabilize the pre-trained encoder in the first few
hundred steps and cause divergence [@Liu2020].

Training runs for up to **3 epochs** with **early stopping** on
validation loss (patience 2 epochs). The best checkpoint by
validation loss is persisted; we do not use the final-epoch
checkpoint, which may have begun to overfit. Aggregate per-epoch
metrics (training loss, validation loss, validation top-1 accuracy)
are written to a JSON metrics file; we do not persist per-record
predictions during training, both to limit checkpoint size and to
preserve the compliance posture (the only per-record predictions
that ever get persisted are from the Synthea evaluation matrix,
which is freely shareable).

## Reproducibility

A single integer seed (`seed: 42`) is propagated to Python's
`random`, NumPy, and all PyTorch random number generators (CPU and
MPS) before model construction or data shuffling. Combined with the
deterministic stratified split (also seeded) and the full pinning
of dependencies via `uv` lockfile, this is sufficient to reproduce
training outcomes within floating-point noise on the same hardware.
Bit-exact reproducibility on GPU/MPS additionally requires
`torch.use_deterministic_algorithms(True)`, which incurs a 10–30%
performance penalty; we do not enable it for v1.

The full set of training hyperparameters is captured in
`configs/training.yaml` and persisted into each saved checkpoint, so
inference code reads the configuration that produced the weights
rather than relying on a separately-tracked config file.

## Current results (placeholder)

The first headline training run is in progress at the time of this
draft. Expected outcomes:

- **Validation top-1 accuracy** at convergence: [TBD: from headline
  run]. Range observed in fixture-data smoke tests is misleading
  (perfect convergence in 3 epochs on 12 examples) and is omitted.
  For a 50-code multi-label task with ~80% of examples on
  high-frequency codes, a healthy result would be in the 0.55–0.80
  top-1 range; below 0.40 would indicate a signal problem.
- **Validation loss curve**: monotonic decrease for at least the
  first epoch, plateau or slight increase by epoch 2-3 (expected
  early-stopping trigger).
- **Per-mode accuracy breakdown** on the test split: D1_full ≥
  D3_text_only ≥ D2_no_code ≥ D4_abbreviated is the expected
  ordering. The gap between D3 and D4 is the headline measurement
  we want from the trained model, since it parallels the same
  measurement we will take from frontier LLMs on Synthea.

Final numbers, learning curves, and per-mode breakdowns will replace
this section once the run completes.

## Limitations of the trained-model arm

Three limitations are worth surfacing explicitly:

1. **Top-50 vocabulary cap excludes the long tail by construction.**
   The trained classifier cannot predict any code outside its
   training vocabulary; LLMs in the comparison arm have no such
   restriction (though they have other failure modes, including
   fabricating codes outside the vocabulary). This is a fair
   asymmetry — the LLMs are constrained only when explicitly given a
   menu in `constrained` mode — but it means trained-model recall on
   long-tail codes is mechanically zero, not learned-zero.
2. **Single random seed for v1.** Reported training-run numbers come
   from a single seed. A proper sensitivity analysis (5+ seeds) is
   on the v2 roadmap; for v1 the single-seed result is reported
   alongside an explicit acknowledgment.
3. **D1/D2 sequence-length truncation.** Setting `max_seq_length=128`
   for hardware feasibility truncates D1_full and D2_no_code inputs
   that exceed 128 tokens (a non-trivial fraction of FHIR-JSON
   payloads). The trained model may therefore underperform on D1/D2
   relative to what a longer-context training run would achieve.
   The headline experiment (D4_abbreviated, ~5–30 tokens) is
   unaffected.
