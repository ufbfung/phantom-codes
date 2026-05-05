# Materials and Methods

## Data and policy compliance

Trained models are fine-tuned on MIMIC-IV-FHIR v2.1 [@Bennett2024;
@Bennett2023; @Johnson2023] under PhysioNet credentialed access. To
remain compliant with PhysioNet's responsible-LLM-use policy
[@PhysioNet2025], which prohibits sending credentialed data through
third-party APIs, all training and validation runs execute entirely on
the corresponding author's own hardware (Apple M1 MacBook Pro). MIMIC
content does not traverse any network beyond the initial PhysioNet
download, is not committed to version control, and is not used as
input to any cloud service. The headline evaluation matrix runs
against Synthea-generated FHIR Bundles [@Walonoski2018], a
freely-redistributable synthetic patient dataset. The trained
classifier is evaluated on the same Synthea inputs as the LLMs,
providing both compliance-by-construction and an out-of-distribution
generalization test (Synthea's clinical text differs systematically
from MIMIC's source notes).

## Cohort and label space

We restrict the cohort to ICD-10-CM codes in the CMS ACCESS Model FHIR
Implementation Guide v0.9.6 [@CMS2026], comprising two disease groups:
**CKM** (cardiometabolic — diabetes, atherosclerotic cardiovascular
disease, CKD stage 3) and **eCKM** (extended cardiometabolic —
hypertension, dyslipidemia, prediabetes, obesity). MIMIC source codes
are normalized at the extraction boundary to canonical dotted form and
canonical HL7 system URIs. After ICD-10-CM and ACCESS-scope filtering,
the MIMIC cohort comprises 245,575 unique conditions split 70/10/20
across train, validation, and test by stratified sampling on
`resource_id` (so all four degradation modes of one condition land in
the same split); 178 unique ICD-10-CM codes appear in the train split.
Each condition is materialized into four rows — one per degradation
mode (D1_full, D2_no_code, D3_text_only, D4_abbreviated).

## Trained-model arm

A PubMedBERT-base classification head [@Gu2021] sized for the top-50
most frequent ICD-10-CM codes is fine-tuned on the MIMIC training
split using PyTorch's MPS backend on consumer Apple Silicon (≈15 hr
wall-clock for an early-stopping run at epoch 2). The architectural,
encoder-rationale, hyperparameter, and training-convergence details
are reported in **Supplementary §S3** (reproducibility appendix).

## Evaluation cohort

The headline matrix is evaluated on a single Synthea-generated cohort
of **125 unique FHIR Conditions**, each materialized into the four
degradation modes for a total of 500 EvalRecord items. Cohort
generation is fully deterministic: Synthea v4.0.0 [@Walonoski2018] is
run at a pinned commit SHA with `seed=42`, producing patient bundles
whose Conditions are extracted, deduplicated per (patient, ICD code),
and filtered to ACCESS-scope codes via the same value-set definitions
used for MIMIC training. Twelve unique ICD-10-CM codes appear in the
cohort, distributed across the ACCESS-scope groups (obesity ~24%;
prediabetes ~20%; type 2 diabetes variants ~31%; essential
hypertension ~14%; lipid disorders ~10%).

## Models evaluated

The headline matrix comprises 29 model configurations: three
string-matching baselines (exact, fuzzy, TF-IDF), one frozen
sentence-transformer retrieval baseline, the fine-tuned PubMedBERT
classifier, and 24 frontier LLM configurations across three providers
and three prompting modes. Each LLM appears in `zeroshot`,
`constrained`, and `rag` configurations:

- **Anthropic** (3 × 3): Claude Haiku 4.5, Sonnet 4.6, Opus 4.7
- **OpenAI** (2 × 3): GPT-5.5 (`gpt-5.5-2026-04-23`) and GPT-4o-mini
  [@OpenAIModels2026]
- **Google** (3 × 3): Gemini 2.5 Pro, Gemini 2.5 Flash, and Gemini 3
  Flash Preview (preview-status, disclosed as such) [@GeminiModels2026]

Gemini 3.1 Pro Preview was excluded due to insufficient daily quota
at our API tier (a partial-coverage analysis appears in
Supplementary §S2). Per-token pricing for cost calculations was
sourced from each provider's documentation as of the evaluation date
[@OpenAIPricing2026; @GeminiPricing2026].

## Prompting modes

All LLMs are evaluated under three prompting modes that vary in how
much external structure is provided:

- **`zeroshot`**: a generic system prompt instructs the model to
  return the most likely ICD-10-CM code for the input, with no
  candidate list.
- **`constrained`**: the system prompt additionally includes the full
  list of CMS ACCESS-scope candidate codes (with display strings) and
  instructs the model to choose only from that list. Mechanical
  schema enforcement (Anthropic forced tool use, OpenAI structured
  outputs, Google `response_schema`) is applied across all modes; in
  `constrained` mode the menu itself is also passed in-prompt.
- **`rag`**: a frozen sentence-transformer (`all-MiniLM-L6-v2`)
  retrieves the top-k = 10 ACCESS-scope candidate codes most similar
  to the per-record input and injects only those into a
  constrained-style prompt.

The verbatim system + user prompts and per-provider tool-use / schema
configurations are reproduced in Supplementary §S1.

## Outcome taxonomy

Each per-record top-1 prediction is classified into one of six
mutually exclusive, exhaustive outcome buckets, ordered from best to
worst by deployment relevance:

1. **`exact_match`**: predicted code equals ground truth.
2. **`category_match`**: predicted code is in the same ICD-10-CM
   3-character category as ground truth (e.g., E11.0 vs. E11.9 — both
   Type 2 diabetes).
3. **`chapter_match`**: same ICD-10-CM chapter, different category.
4. **`out_of_domain`**: real ICD-10-CM code with no hierarchical
   relation to ground truth.
5. **`no_prediction`**: model returned no usable prediction (empty
   array, refusal, or transient API failure that exhausted the SDK
   retry budget). The model fabricated nothing; it abstained.
6. **`hallucination`**: predicted code does not exist in the FY2026
   CMS ICD-10-CM tabular list (mechanically validated against the
   bundled validator).

The `no_prediction` and `hallucination` buckets capture qualitatively
different failure modes and are reported separately. From a
deployment-safety perspective, abstention is preferable to
fabrication: an abstaining model emits no spurious code that
downstream systems silently mishandle. Earlier work [@Soroush2024]
collapses these into a single error bucket; the six-way split
surfaces the distinction that conflation hides.

## Statistical analysis

Per-(model, mode) outcome rates are reported as point estimates with
95% Wilson confidence intervals [@Wilson1927], preferred over
normal-approximation intervals for small N or rates near zero or one
— both common in our matrix (per-cell N = 125; constrained-mode
hallucination rates approach 0%). Within-model paired comparisons
(zero-shot vs. constrained vs. RAG on identical inputs) are reported
as exact McNemar tests on discordant prediction pairs. Across-model
and across-mode comparisons are reported descriptively with
overlapping/non-overlapping CIs; we deliberately do not perform
null-hypothesis significance tests across the full grid because the
intended interpretation is comparative deployment-relevance rather
than detection of any single effect.
