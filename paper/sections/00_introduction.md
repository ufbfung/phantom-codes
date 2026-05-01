# Introduction

> **Status:** Draft v0 (2026-05-01). Markdown for now; convert to LaTeX once
> structure stabilizes. Citations use `[Author Year]` keys that map to
> `paper/sections/references.md`.

---

Clinical concept normalization — the task of mapping unstructured mentions of
diagnoses, medications, labs, and procedures to identifiers in controlled
vocabularies such as ICD-10-CM, LOINC, RxNorm, and SNOMED CT — sits at the
foundation of healthcare data interoperability. Accurate normalization is
required for billing, quality reporting, cohort definition for research,
population-health analytics, and the increasing use of structured EHR data to
power downstream machine-learning models. Errors propagate: a misassigned
diagnosis code can shift a hospital's case-mix index, mis-stratify a research
cohort, or quietly corrupt the training data of the next generation of models.

For most of the last decade, the state of the art for automated coding came
from supervised neural classifiers trained on labeled clinical text — most
prominently CAML's hierarchical attention mechanism over MIMIC-III discharge
summaries [Mullenbach 2018], and more recent transformer-based extensions
that adapt domain-pretrained encoders such as PubMedBERT and RoBERTa-PM to
the multi-label classification setting [Huang 2022]. These systems achieve
strong micro-F1 on common codes but require large, labeled training corpora
and degrade sharply on the long tail of rare diagnoses.

The arrival of general-purpose large language models has changed the
deployment economics. Foundation models such as GPT-4, Claude Sonnet, and
Gemini Pro can in principle perform code prediction with no task-specific
fine-tuning, no labeled training data, and no infrastructure beyond an API
call. The question that motivated this work is not *whether* LLMs can predict
medical codes — they can — but how they fail when they do, and whether those
failures look qualitatively different from the failures of the systems they
might replace.

## The hallucination problem in clinical coding

A key concern with applying LLMs to coded clinical data is that they can
fabricate codes that do not exist in the target vocabulary. Unlike a
classifier trained on a closed label set, a generative model has no
mechanical guarantee that its output is even a member of the controlled
vocabulary it was asked to produce. This is the narrowest, most
mechanically-checkable instance of the broader phenomenon of medical
hallucination — defined by Kim et al. as model-generated output that is
"factually incorrect, logically inconsistent, or unsupported by authoritative
clinical evidence in ways that could alter clinical decisions"
[Kim 2025]. Survey work documents that hallucination is a pervasive and
under-measured failure mode across clinical NLP tasks [Ji 2023].

Empirical evidence on how often this happens for code prediction
specifically is sparse but uniformly cautionary. Soroush et al. benchmarked
GPT-3.5, GPT-4, Gemini Pro, and Llama-2-70B on direct querying of medical
codes and reported that the best model (GPT-4) achieved only 46% exact match
on ICD-9, 34% on ICD-10, and 50% on CPT — with a substantial fraction of
errors being non-existent codes rather than real-but-wrong assignments
[Soroush 2024]. Goel et al. found that LLMs systematically struggle to
distinguish real medical codes from plausibly-formatted fakes
[Goel 2024]. Adversarial-prompt studies push these numbers higher: LLMs
elaborate on planted fabrications in 50–82% of cases, with simple mitigation
prompts only halving the rate [Omiye 2025]. A growing literature now treats
LLM hallucination in clinical settings as a primary safety concern, not a
peripheral one [Kim 2025; Hatem 2025].

## Existing benchmarks and their gaps

The response from the field has largely been to constrain the LLM rather
than to characterize its unconstrained behavior. Two-stage architectures
combine an LLM proposer with a discriminative verifier [Yang 2023];
extract-retrieve-rerank pipelines (MedCodER) ground predictions in retrieved
candidates [Sahaj 2024]; generation-assisted vector search (GAVS) inverts
the standard RAG flow by generating clinical entities first and then
matching them against the coding ontology [Mahmoud 2025]; agentic systems
walk the ICD index sequentially in imitation of human coders
[Motzfeldt 2025]; and neuro-symbolic verifiers report driving Type-I
hallucination rates to zero by refusing to emit codes that fail an existence
check [Hybrid-Code v2 2025]. Recent benchmarks expand coverage across
vocabularies and clinical sub-domains: ICPC-2 in Brazilian Portuguese
primary care [Almeida 2025], full-spectrum ICD-11 mapped from MIMIC-III
[Bhatti 2025], and rationale-annotated MIMIC-IV/ICD-10 datasets for
explainability evaluation [Li 2025]. A recent systematic review of 35
LLM-ICD-coding studies through January 2025 finds that LLMs reliably handle
common codes but degrade on rare diagnoses and lack external validation
[Gershon 2025].

What is conspicuously missing across this body of work:

1. **Hallucination is rarely a first-class outcome.** Most benchmarks report
   precision, recall, F1, or weighted variants thereof. When a model
   predicts a non-existent code, that prediction is typically counted as
   "wrong" — collapsing two distinct failure modes (real-but-wrong code vs.
   fabricated code) into the same bucket and obscuring the pattern that
   motivates the entire constrained-decoding research direction.
2. **Zero-shot vs. constrained is rarely a controlled within-model
   comparison.** Constrained-decoding papers report performance with the
   constraint; unconstrained-LLM papers report performance without. Few
   isolate the effect by holding the LLM, the input, and the task fixed and
   varying only whether a candidate list is provided. Without that
   comparison, it is unclear whether constrained-mode gains come from
   eliminating hallucination, eliminating real-but-wrong predictions, or
   both.
3. **Inputs are taken as given.** Benchmarks evaluate models on clinical
   notes as they appear in the dataset. None systematically test what
   happens when the canonical lexical signals on which string-matching
   baselines depend (display tokens, full clinical names) are deliberately
   stripped while the underlying diagnosis is preserved.

## This work

We present **Phantom Codes**, a reproducible benchmark for clinical concept
normalization that addresses these three gaps. Our primary contributions are
methodological rather than architectural:

- **A degradation pipeline that includes an abbreviation-stress mode
  (D4_abbreviated)** which deliberately replaces canonical display tokens
  with clinical jargon (T2DM, HTN, CKD, etc.) while preserving the
  underlying diagnosis. D4 is the headline experiment: it is the condition
  under which string-matching baselines collapse and under which any
  remaining LLM advantage must come from genuine semantic retrieval rather
  than lexical overlap. Three lower-stress modes (D1_full, D2_no_code,
  D3_text_only) provide a controlled gradient of input information.
- **A 5-way outcome taxonomy with hallucination as an explicit, mutually
  exclusive bucket.** Every prediction lands in exactly one of:
  exact_match, category_match, chapter_match, out_of_domain (real ICD-10-CM
  code, no hierarchical relation to truth), or hallucination (predicted
  code does not exist in ICD-10-CM, mechanically checked against the
  CMS-published FY2026 tabular list). This replaces the standard
  precision/recall framing with a hierarchy-aware classification of
  failure modes.
- **A controlled within-model ablation between zero-shot and constrained
  prompting.** The same LLM, the same input, varying only whether a
  candidate list from the CMS ACCESS Model FHIR ValueSets is provided.
  This isolates "wandering off the menu" from "genuine ignorance" and
  quantifies the mechanism by which constrained decoding helps (or fails
  to help) in code prediction.

We evaluate this benchmark on FHIR Conditions from MIMIC-IV-FHIR v2.1,
restricted to ICD-10-CM codes in the CMS ACCESS Model implementation guide
(diabetes, atherosclerotic cardiovascular disease, CKD stage 3,
hypertension, dyslipidemia, prediabetes, obesity). Models compared include
three families of LLMs (Claude Sonnet 4.6 and Haiku 4.5; GPT-4-class;
Gemini 2.5 Pro and Flash) under both prompting modes; a sentence-transformer
retrieval baseline with FAISS; a PubMedBERT classification head; and three
constrained-only baselines (exact match, fuzzy token-set ratio, TF-IDF
cosine).

The empirical contribution we expect to surface is not a model-vs-model
leaderboard — those exist already and are increasingly saturated. It is the
shape of LLM failure under D4 conditions: how the hallucination bucket
fills relative to the other four outcomes, how that distribution shifts
between zero-shot and constrained prompting within the same model, and what
that pattern implies for the safe deployment of LLMs in workflows where the
output is consumed by downstream systems that assume code validity.

The framework, taxonomy, and evaluation runner are designed to extend
without architectural change to LOINC (laboratory observations) and RxNorm
(medication requests) in subsequent work, with the same five-way outcome
classes applied to the appropriate hierarchy and existence validator.
