# Phantom Codes

> *Hallucination in LLM-Based Clinical Concept Normalization.*

A reproducible benchmark for evaluating how well LLMs and trained models map degraded clinical data to standardized terminology codes — and how often they fabricate codes that don't exist.

**Clinical concept normalization** is the task of mapping mentions of clinical entities (diagnoses, labs, medications) to identifiers in controlled vocabularies (ICD-10-CM, LOINC, RxNorm, etc.). It's a foundational task for healthcare data integration, billing, research cohort construction, and AI-assisted documentation.

**Status:** Phase 0 complete for diagnoses. Degradation pipeline, 5-way outcome taxonomy, ICD-10-CM validator, LLM module (Claude + GPT + Gemini in zero-shot + constrained + RAG modes), bi-encoder retrieval baseline, constrained-only baselines (exact / fuzzy / TF-IDF), token + cost tracking, run-manifest sidecars for reproducibility, blinded `--infra-only` smoke-test mode, and end-to-end eval runner are implemented and tested. Real MIMIC ingestion + trained classifier head forthcoming.

**v1 scope:** diagnoses (ICD-10-CM). v2+ extends to labs (LOINC) and medications (RxNorm) using the same framework.

## What this is

Take a coded FHIR Condition (e.g., one bearing ICD-10 `E11.9` "Type 2 diabetes mellitus without complications"), strip or degrade the coding fields, and ask: can a model recover the original `(system, code)` — and when it can't, does it admit ignorance or fabricate a plausible-looking-but-wrong code?

The benchmark compares:

- **LLMs** — Claude (Opus 4.7, Sonnet 4.6, Haiku 4.5), GPT (5.5, 4-class), Gemini 2.5 (Pro, Flash). Three prompting modes: **zero-shot** (no menu, primary experiment), **constrained** (fixed candidate list, hallucination control), and **RAG** (per-record retrieved candidates).
- **Trained models** — sentence-transformer retrieval (FAISS) and PubMedBERT classification head.
- **Baselines** — exact, fuzzy (rapidfuzz), TF-IDF nearest neighbor.

across four degradation modes:

| Mode | What's removed | What remains |
|------|----------------|--------------|
| `D1_full` | nothing | full coded resource (control) |
| `D2_no_code` | `coding.code`, `coding.system` | display, text, status, category |
| `D3_text_only` | all coding | canonical `text` field only |
| `D4_abbreviated` | all coding; canonical text replaced with clinical jargon (T2DM, HTN, CKD, …) | one-sentence summary using abbreviations |

D4 is the headline mode: it strips the canonical display tokens that string-matching baselines depend on while preserving the underlying diagnosis. That's where LLMs are forced to actually *retrieve from semantic understanding* — and where hallucinations should surface.

## Five-way outcome taxonomy

Every prediction lands in exactly one bucket. Aligned with the literature: hierarchical-match terminology from [Mullenbach et al. 2018 (CAML)](https://aclanthology.org/N18-1100/), OOD framing from [Hendrycks & Gimpel 2017](https://arxiv.org/abs/1610.02136), hallucination/fabrication from [Ji et al. 2023](https://dl.acm.org/doi/10.1145/3571730).

- `exact_match` — predicted code equals truth
- `category_match` — same 3-char ICD-10 category (e.g., E11.0 vs E11.9)
- `chapter_match` — same ICD-10 chapter (E vs E), different category
- `out_of_domain` — real ICD-10-CM code, no hierarchical relation to truth
- `hallucination` — predicted code does NOT exist in ICD-10-CM (**the headline metric**)

The taxonomy generalizes naturally to LOINC and RxNorm in v2: same definitions with the appropriate hierarchy and validator file.

## Scope: CMS ACCESS Model conditions

We restrict the cohort to ICD-10-CM codes in the [CMS ACCESS Model](https://www.cms.gov/priorities/innovation/innovation-models/access) FHIR Implementation Guide:
- **CKM track**: diabetes (E08/E09/E11/E13), atherosclerotic CVD, CKD stage 3
- **eCKM track**: hypertension, dyslipidemia, prediabetes, obesity

ValueSets bundled at [src/phantom_codes/data/access_valuesets/](src/phantom_codes/data/access_valuesets/).

## Data

- **Primary:** [MIMIC-IV-FHIR v2.1](https://physionet.org/content/mimic-iv-fhir/2.1/) — credentialed access via PhysioNet. Cannot be redistributed; this repo only contains code and synthetic fixtures.
- **Open benchmark:** [Synthea](https://github.com/synthetichealth/synthea)-generated FHIR Bundles. Released as `benchmarks/synthetic_v1/` once that phase lands.

## Getting started (development)

```bash
uv sync --extra dev
uv run pytest tests/
uv run phantom-codes --help

# End-to-end smoke test on local fixtures (no network, no MIMIC):
uv run phantom-codes smoke-test

# With LLMs (~$3-5 per run with all API keys set):
#   ANTHROPIC_API_KEY (required) — Haiku, Sonnet, Opus 4.7
#   OPENAI_API_KEY  (optional)   — GPT-4o-mini, GPT-5.5
#   GEMINI_API_KEY  (optional)   — Gemini 2.5 Flash, Pro
uv run phantom-codes smoke-test --llms
```

The full MIMIC pipeline requires [PhysioNet credentialed access](https://physionet.org/about/citi-course/) for MIMIC-IV-FHIR v2.1 and a Google account linked to your PhysioNet profile (for GCS read access on the source bucket).

## Repository layout

```
src/phantom_codes/
  data/         # fhir_loader, code_set, degrade, abbreviate, disease_groups
                # icd10cm/  (bundled CMS FY2026 codes for hallucination detection)
                # access_valuesets/  (CMS ACCESS Model FHIR ValueSets)
  models/       # base ABC; llm (Anthropic, OpenAI, Google); rag_llm;
                #   retrieval (sentence-transformer + cosine); baselines;
                #   (classifier head — TBD)
  eval/         # metrics (5-way taxonomy), runner, infra (blinded
                #   smoke-test), cost (pricing → $), manifest (run sidecars)
  cli.py
configs/        # YAML configs (data.yaml, pricing.yaml)
tests/          # unit tests + synthetic fixtures (167 tests, ruff clean)
benchmarks/     # released open benchmark (Synthea — TBD)
paper/
  sections/     # markdown drafts: 00_introduction.md, references.md;
                #   future: methods, results, discussion (TBD)
  figures/      # generated figures (TBD)
  tables/       # generated tables (TBD)
```

## Roadmap

- **v1**: diagnoses (ICD-10-CM) — in progress
- **v2**: + labs (LOINC), via Observation resources
- **v3**: + medications (RxNorm), via MedicationRequest resources

The 5-way outcome taxonomy, eval runner, and prompt-caching infrastructure are designed to extend across vocabularies without architectural change.

## Inspirations and related work

Phantom Codes draws on five recent (2024-2026) lines of work in LLM-based clinical coding:

- **From Almeida et al. 2025 (ICPC-2 benchmark, NeurIPS GenAI4Health):** the multi-axis evaluation table — F1 paired with cost, latency, token usage, and format adherence — gives a more honest picture of LLM viability than accuracy alone. We adopt the same multi-axis framing for our headline results.
- **From Bhatti et al. 2025 (MAX-EVAL-11):** clinically-informed scoring with weighted reward by code relevance and diagnostic specificity. Our 5-way outcome taxonomy is a sibling of this idea — discrete buckets instead of continuous weights, but the same intuition that not every wrong answer is equally wrong.
- **From Motzfeldt et al. 2025 (Code Like Humans, EMNLP Findings):** the "agentic" alternative to single-shot prompting — sequential search/verify/predict over the ICD index. We treat their approach as a third prompting mode to add (alongside our `zeroshot` and `constrained`) and test whether agent decomposition reduces hallucination on D4 inputs.
- **From Kim et al. 2025 (Medical Hallucination in Foundation Models):** a strict definition of medical hallucination as "factually incorrect, logically inconsistent, or unsupported by authoritative clinical evidence in ways that could alter clinical decisions." Our `hallucination` bucket (code does not exist in ICD-10-CM) is a narrow, mechanically-checkable instance of their broader definition.
- **From Li et al. 2025 (ICD Coding Rationales):** the faithfulness/plausibility distinction for evaluating rationales, with a MIMIC-IV/ICD-10 rationale-annotated dataset. A natural v2 extension is to ask each LLM for a rationale alongside its code and evaluate both axes.

What we contribute on top of this prior work:

- **D4_abbreviated as a stress test** — deliberately strips the canonical display tokens that string-matching baselines depend on, forcing semantic retrieval. This is the experiment where LLM-vs-baseline differences should surface most cleanly, and the headline mode of the benchmark.
- **Hallucination as a first-class outcome** — "code does not exist in the controlled vocabulary" gets its own bucket, mechanically checked against a CMS-published validator. Replaces top-1 accuracy as the headline metric.
- **Zero-shot vs. constrained as a controlled within-model ablation** — isolates "wandering off the menu" from "genuine ignorance" using the same model and the same input, varying only the candidate-list constraint.

## References

The full curated bibliography (16 references organized into foundational landmarks, direct LLM-on-medical-coding evaluations, and clinical-hallucination studies) lives at [paper/sections/references.md](paper/sections/references.md). The five papers most directly inspiring this work:

1. **Soroush et al. 2024 (NEJM AI)** — *Large Language Models Are Poor Medical Coders.* GPT-4 achieves 46% / 34% / 50% exact match on ICD-9 / ICD-10 / CPT respectively; substantial hallucination of non-existent codes. [doi:10.1056/AIdbp2300040](https://ai.nejm.org/doi/full/10.1056/AIdbp2300040)
2. **Almeida et al. 2025 (NeurIPS GenAI4Health)** — *Large Language Models as Medical Code Selectors.* 33-LLM benchmark on ICPC-2; multi-axis evaluation (F1, cost, latency, format adherence). [arXiv:2507.14681](https://arxiv.org/abs/2507.14681)
3. **Bhatti et al. 2025** — *MAX-EVAL-11.* MIMIC-III mapped to full-spectrum ICD-11 with weighted scoring by code relevance and diagnostic specificity. [doi:10.1101/2025.10.30.25339130](https://www.medrxiv.org/content/10.1101/2025.10.30.25339130v1)
4. **Motzfeldt et al. 2025 (EMNLP Findings)** — *Code Like Humans.* Agentic ICD-10 coding traversing the alphabetic index sequentially. [arXiv:2509.05378](https://arxiv.org/abs/2509.05378)
5. **Kim et al. 2025** — *Medical Hallucinations in Foundation Models and Their Impact on Healthcare.* Defines medical hallucination broadly; benchmarks 11 foundation models. [arXiv:2503.05777](https://arxiv.org/abs/2503.05777)

Foundational references for the 5-way outcome taxonomy:

- Mullenbach et al. 2018 (CAML, NAACL) — hierarchical-match terminology. [aclanthology.org/N18-1100](https://aclanthology.org/N18-1100/)
- Hendrycks & Gimpel 2017 (ICLR) — OOD detection baseline. [arXiv:1610.02136](https://arxiv.org/abs/1610.02136)
- Ji et al. 2023 (ACM Computing Surveys) — hallucination survey. [doi:10.1145/3571730](https://dl.acm.org/doi/10.1145/3571730)

## License

Code in this repository is released under the [MIT License](LICENSE).

- **MIMIC-derived data is never distributed here** — PhysioNet's Credentialed Health Data License governs the data and prohibits redistribution. Obtain MIMIC-IV-FHIR via your own [PhysioNet credentialing](https://physionet.org/about/citi-course/).
- **Trained model weights are not released** from this repo. Weights derived from MIMIC are typically redistributed only via PhysioNet's "MIMIC-IV Models" channel under similar credentialing.
- **Released benchmark data** (Synthea-generated, planned for `benchmarks/synthetic_v1/`) will follow Synthea's Apache 2.0 license.
