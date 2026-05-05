# Methods

> **Status:** Draft v1 (2026-05-04). Numbers from the headline
> training run filled in. Section reorganized to cover both
> evaluation arms: trained-model training (subsections under
> "Trained-model methodology") and LLM evaluation matrix
> (subsections under "Evaluation methodology"). Both arms terminate
> on the same Synthea cohort (§Results) so they share a common
> evaluation-protocol subsection.

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
the pre-trained encoder for three reasons: (a) domain-adaptive
pre-training (PubMedBERT is trained from scratch on PubMed
abstracts + full-text articles, producing a tokenizer that treats
biomedical terms as single tokens rather than fragmented
subwords [@Devlin2019; @Gururangan2020]); (b) strong empirical
performance on the BLURB biomedical NLP benchmark [@Gu2021]; and
(c) a computational footprint (~110M parameters) compatible with
local fine-tuning on consumer Apple Silicon, which is the
compliance-enabling property — any larger encoder would have
forced us onto cloud GPU infrastructure with corresponding
data-residency review for credentialed MIMIC data. Alternative
encoders (ClinicalBERT [@Alsentzer2019], BioBERT [@Lee2020],
BioLinkBERT [@Yasunaga2022], and larger biomedical models including
BiomedLM, GatorTron, BioMistral, and Med-PaLM via parameter-
efficient fine-tuning [@Hu2022; @Dettmers2023]) were considered
and either ruled out (ClinicalBERT — MIMIC-III pre-training
contaminates MIMIC-IV evaluation) or deferred to a v2 ablation
arm. Generative LLMs (Claude, GPT, Gemini) are present in the
evaluation matrix but as zero-shot / constrained / RAG arms, not
as fine-tuned trained models; comparing the same task across both
architectures is the core comparison the paper enables.

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

Batch size 16 at maximum sequence length 128 uses ~5–6 GB of unified
memory and leaves headroom for normal laptop operation during a
~15-hour training run. Sequence-length 128 truncates the longest
input modes (D1\_full / D2\_no\_code FHIR JSON payloads at
~200–400 tokens) but leaves the headline experiment's
D3\_text\_only / D4\_abbreviated inputs (10–50 tokens) untouched —
documented in §Limitations. Steady-state throughput is ~1.85
iterations per second on M1 MPS, yielding ~7 hours per epoch (train
+ validation) and ~15 hours total wall-clock for an early-stopping
run at epoch 2; an A100 cloud GPU would run ~3–5× faster at ~\$4–7
per training run if the data-residency story permits.

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

## Training convergence

Validation top-1 accuracy at convergence reached 0.992–0.993
across three training epochs on the held-out MIMIC-IV validation
split. This high in-distribution value reflects the long-tail
concentration of the ACCESS-scope cohort (the top-50 codes cover
the bulk of training rows; see §Cohort) combined with effective
fine-tuning of a pre-trained biomedical encoder. The same
checkpoint evaluated on the Synthea out-of-distribution cohort
yields a substantially lower per-mode top-1 accuracy (40–50% range,
see §Results) — the gap between in-distribution validation
accuracy and out-of-distribution Synthea accuracy is one of the
deployment-relevant measurements this paper surfaces.

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

## Evaluation methodology

### Evaluation cohort

The headline matrix is evaluated on a single Synthea-generated
cohort of 125 unique FHIR Conditions, each materialized into the
four degradation modes for a total of 500 EvalRecord items. Cohort
generation is fully deterministic: Synthea v4.0.0
[@Walonoski2018] is run at a pinned commit SHA with `seed=42`,
producing patient bundles whose Conditions are extracted via the
project's Bundle parser, deduplicated per (patient, ICD code), and
filtered to ACCESS-scope codes via the same value-set definitions
used for MIMIC training. The pinned-SHA + seed combination yields
byte-identical cohorts on replication; full reproduction guidance
is in the project's BENCHMARK reference. Cohort composition (12
unique ICD-10-CM codes; distribution skewed toward obesity,
prediabetes, and hypertension by virtue of Synthea's default
module configuration) is reported in §Results.

### Models evaluated

The headline matrix comprises 29 model configurations, drawn from
five families: three string-matching baselines (exact, fuzzy,
TF-IDF), one frozen sentence-transformer retrieval baseline, the
fine-tuned PubMedBERT classifier described above, and 24 frontier
LLM configurations across three providers and three prompting
modes. Each LLM appears in `zeroshot`, `constrained`, and `rag`
configurations:

- **Anthropic** (3 models × 3 modes = 9 configs): Claude Haiku
  4.5 (`claude-haiku-4-5`), Sonnet 4.6 (`claude-sonnet-4-6`),
  Opus 4.7 (`claude-opus-4-7`)
  [@OpenAIModels2026 for the analogous OpenAI guide]
- **OpenAI** (2 models × 3 modes = 6 configs): GPT-5.5 (pinned to
  the dated snapshot `gpt-5.5-2026-04-23`) and GPT-4o-mini
  [@OpenAIModels2026]
- **Google** (3 models × 3 modes = 9 configs): Gemini 2.5 Pro,
  Gemini 2.5 Flash, and Gemini 3 Flash Preview (the latter is
  preview-status at evaluation time and disclosed as such)
  [@GeminiModels2026]

Gemini 3.1 Pro Preview was excluded from the headline matrix due
to insufficient daily quota at our API tier (Tier 1 cap of 250
requests per day, vs. the ~6,000 calls required for full
coverage); a partial-coverage analysis appears in §Supplementary.
Per-token pricing for cost calculations was sourced from each
provider's documentation as of the evaluation date
[@OpenAIPricing2026; @GeminiPricing2026].

### Prompting modes

All LLMs are evaluated under three prompting modes that vary in
how much external structure is provided:

- **`zeroshot`**: a generic system prompt instructs the model to
  return the most likely ICD-10-CM code for the input, with no
  candidate list. The model selects from its full pre-training
  vocabulary, which includes both real codes and the possibility
  of fabricated codes.
- **`constrained`**: the system prompt additionally includes the
  full list of CMS ACCESS-scope candidate codes (with display
  strings) and instructs the model to choose only from that list.
  Mechanical schema enforcement (provider-specific: Anthropic
  forced tool use, OpenAI structured outputs, Google
  `response_schema`) is applied across all modes; in
  `constrained` mode the menu itself is also passed in-prompt.
- **`rag`**: a frozen sentence-transformer (default
  `all-MiniLM-L6-v2`) retrieves the top-k = 10 ACCESS-scope
  candidate codes most similar to the per-record input and
  injects only those into a constrained-style prompt. This
  contrasts with `constrained` (full menu) by varying the
  candidate set per record.

The verbatim system + user prompts and per-provider tool-use /
schema configurations are reproduced in §Supplementary S1.

### Outcome taxonomy

Each per-record top-1 prediction is classified into one of six
mutually exclusive, exhaustive outcome buckets, ordered from best
to worst by deployment relevance:

1. **`exact_match`**: predicted code equals ground-truth code.
2. **`category_match`**: predicted code is in the same ICD-10-CM
   3-character category as ground truth (e.g., E11.0 vs. E11.9 —
   both Type 2 diabetes).
3. **`chapter_match`**: predicted code is in the same ICD-10-CM
   chapter (first character) but a different category.
4. **`out_of_domain`**: predicted code exists in CMS ICD-10-CM but
   has no hierarchical relation to ground truth.
5. **`no_prediction`**: the model returned no usable prediction —
   an empty `predictions` array, an explicit refusal, or a
   transient API failure that exhausted the SDK's internal retry
   budget. The model fabricated nothing; it abstained.
6. **`hallucination`**: the predicted code does not exist in the
   FY2026 CMS ICD-10-CM tabular list (mechanically validated
   against the bundled validator).

The `no_prediction` and `hallucination` buckets capture
qualitatively different failure modes and are therefore reported
separately. From a deployment-safety perspective, abstention is
preferable to fabrication — an abstaining model emits no spurious
code that downstream systems silently mishandle, even though both
fail to produce a usable prediction. From a model-quality
perspective, persistent abstention is still a quality concern but
is not equivalent to fabrication. Earlier work on LLM medical
coding [@Soroush2024] uses a single error bucket that conflates
these failure modes; the six-way split surfaces the distinction
that conflation hides.

### Statistical analysis

Per-(model, mode) outcome rates are reported as point estimates
with 95% Wilson confidence intervals [@Wilson1927-conceptually,
implemented in `eval/report.py` per the formula in any standard
biostatistics text]. Wilson intervals are preferred over
normal-approximation intervals for small N or rates near zero or
one — both common in our matrix (per-cell N = 125; constrained-mode
hallucination rates approach 0% for several models). Within-model
paired comparisons (zero-shot vs. constrained vs. RAG on identical
inputs) are reported as exact McNemar tests on the discordant
prediction pairs. Across-model and across-mode comparisons are
reported descriptively with overlapping/non-overlapping CIs; we
deliberately do not perform null-hypothesis significance tests
across the full 24-LLM × 4-mode × 6-bucket grid because (a) the
intended interpretation is comparative deployment-relevance rather
than detection of any single effect, and (b) Bonferroni or similar
corrections at this matrix size would render most cells
underpowered and obscure the directional patterns the paper aims
to surface.
