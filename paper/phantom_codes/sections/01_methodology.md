# MATERIALS AND METHODS

## Data

The headline evaluation matrix runs entirely against
Synthea-generated FHIR Bundles [@Walonoski2018], a freely-
redistributable synthetic patient dataset. Cohort generation is
fully reproducible from a pinned Synthea version and seed (see
*Cohort and label space* below).

## Cohort and label space

We restrict the cohort to ICD-10-CM codes in the CMS ACCESS Model
FHIR Implementation Guide v0.9.6 [@CMS2026]. The Implementation
Guide defines two disease groups: **CKM** (cardiometabolic, covering
diabetes, atherosclerotic cardiovascular disease, and CKD stage 3)
and **eCKM** (extended cardiometabolic, covering hypertension,
dyslipidemia, prediabetes, and obesity). The headline matrix is
evaluated on a single Synthea-generated cohort of **125 unique FHIR
Conditions**, each materialized into four degradation modes for a
total of 500 EvalRecord items. Cohort generation is fully
deterministic: Synthea v4.0.0 [@Walonoski2018] is run at a pinned
commit SHA with `seed=42`, producing patient bundles whose
Conditions are extracted, deduplicated per (patient, ICD code), and
filtered to ACCESS-scope codes via bundled value-set definitions.
Twelve unique ICD-10-CM codes appear in the cohort: obesity (~24%),
prediabetes (~20%), type 2 diabetes variants (~31%), essential
hypertension (~14%), and lipid disorders (~10%).

## Models evaluated

The headline matrix comprises 28 model configurations. These are:
three string-matching baselines (exact, fuzzy, TF-IDF), one frozen
sentence-transformer retrieval baseline, and 24 frontier LLM
configurations across three providers and three prompting modes.
Each LLM appears in `zeroshot`, `constrained`, and `rag`
configurations.

- **Anthropic** (3 × 3): Claude Haiku 4.5, Sonnet 4.6, Opus 4.7
- **OpenAI** (2 × 3): GPT-5.5 (`gpt-5.5-2026-04-23`) and GPT-4o-mini
  [@OpenAIModels2026]
- **Google** (3 × 3): Gemini 2.5 Pro, Gemini 2.5 Flash, and Gemini 3
  Flash Preview (preview-status, disclosed as such) [@GeminiModels2026]

Gemini 3.1 Pro Preview was excluded due to insufficient daily quota
at our API tier; a partial-coverage analysis appears in
Supplementary §S2. Per-token pricing for cost calculations was
sourced from each provider's documentation as of the evaluation date
[@OpenAIPricing2026; @GeminiPricing2026].

## Prompting modes

All LLMs are evaluated under three prompting modes that vary in how
much external structure is provided.

- **`zeroshot`**: a generic system prompt instructs the model to
  return the most likely ICD-10-CM code for the input, with no
  candidate list.
- **`constrained`**: the system prompt additionally includes the
  full list of CMS ACCESS-scope candidate codes (with display
  strings) and instructs the model to choose only from that list.
  Mechanical schema enforcement (Anthropic forced tool use, OpenAI
  structured outputs, Google `response_schema`) is applied across
  all modes; in `constrained` mode the menu itself is also passed
  in-prompt.
- **`rag`**: a frozen sentence-transformer (`all-MiniLM-L6-v2`)
  retrieves the top-k = 10 ACCESS-scope candidate codes most similar
  to the per-record input and injects only those into a
  constrained-style prompt.

The verbatim system and user prompts and per-provider tool-use and
schema configurations are reproduced in Supplementary §S1.

## Outcome taxonomy

Each per-record top-1 prediction is classified into one of six
mutually exclusive, exhaustive outcome buckets, ordered from best
to worst by deployment relevance:

1. **`exact_match`**: predicted code equals ground truth.
2. **`category_match`**: predicted code is in the same ICD-10-CM
   3-character category as ground truth (e.g., E11.0 vs. E11.9, both
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

The `no_prediction` and `hallucination` buckets capture
qualitatively different failure modes and are reported separately.
From a deployment-safety perspective, abstention is preferable to
fabrication, since an abstaining model emits no spurious code that
downstream systems silently mishandle. Earlier work [@Soroush2024]
collapses these into a single error bucket; the six-way split
surfaces the distinction that conflation hides.

## Cost tracking

For every model invocation in our matrix the per-prediction CSV
records token usage, latency, and `cost_usd` computed at runtime
against the versioned pricing snapshot in `configs/pricing.yaml`
[@OpenAIPricing2026; @GeminiPricing2026]. These infrastructure
metrics are recorded even when the prediction is wrong and serve
as the basis for the cost-per-correct normalization reported in
Results. Cost-per-correct (USD per exact-match outcome) collapses
per-call price and per-call accuracy into a single
deployment-relevant number; we report it alongside per-call cost
because the two diverge meaningfully when accuracy drops.

## Statistical analysis

Per-(model, mode) outcome rates are reported as point estimates
with 95% Wilson confidence intervals, which we prefer over
normal-approximation intervals because both small N and rates near
zero or one are common in our matrix (per-cell N = 125, and
constrained-mode hallucination rates approach 0%). Within-model
paired comparisons across zero-shot, constrained, and RAG modes on
identical inputs are reported as exact McNemar tests on discordant
prediction pairs. Across-model and across-mode comparisons are
reported descriptively with overlapping or non-overlapping CIs. We
deliberately do not perform null-hypothesis significance tests
across the full grid, because the intended interpretation is
comparative deployment relevance rather than detection of any
single effect.
