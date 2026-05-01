# Phantom Codes

> *Hallucination in LLM-Based Clinical Concept Normalization.*

A reproducible benchmark for evaluating how well LLMs and trained models map degraded clinical data to standardized terminology codes — and how often they fabricate codes that don't exist.

**Clinical concept normalization** is the task of mapping mentions of clinical entities (diagnoses, labs, medications) to identifiers in controlled vocabularies (ICD-10-CM, LOINC, RxNorm, etc.). It's a foundational task for healthcare data integration, billing, research cohort construction, and AI-assisted documentation.

**Status:** Phase 0 complete for diagnoses. Degradation pipeline, 5-way outcome taxonomy, ICD-10-CM validator, LLM module (Claude + GPT, zero-shot + constrained), constrained baselines (exact / fuzzy / tfidf), and end-to-end eval runner are implemented and tested (111 tests, ruff clean). Real MIMIC ingestion + trained models forthcoming.

**v1 scope:** diagnoses (ICD-10-CM). v2+ extends to labs (LOINC) and medications (RxNorm) using the same framework.

## What this is

Take a coded FHIR Condition (e.g., one bearing ICD-10 `E11.9` "Type 2 diabetes mellitus without complications"), strip or degrade the coding fields, and ask: can a model recover the original `(system, code)` — and when it can't, does it admit ignorance or fabricate a plausible-looking-but-wrong code?

The benchmark compares:

- **LLMs** — Claude Sonnet 4.6, Claude Haiku 4.5, GPT-4-class. Both **zero-shot** (no menu, primary experiment) and **constrained** (with candidate list, hallucination control).
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

# With LLMs (requires ANTHROPIC_API_KEY in .env, ~$0.05):
uv run phantom-codes smoke-test --llms
```

The full MIMIC pipeline requires [PhysioNet credentialed access](https://physionet.org/about/citi-course/) for MIMIC-IV-FHIR v2.1 and a Google account linked to your PhysioNet profile (for GCS read access on the source bucket).

## Repository layout

```
src/phantom_codes/
  data/         # fhir_loader, code_set, degrade, abbreviate, disease_groups
                # icd10cm/  (bundled CMS FY2026 codes for hallucination detection)
                # access_valuesets/  (CMS ACCESS Model FHIR ValueSets)
  models/       # base ABC + llm, baselines, (retrieval, classifier — TBD)
  eval/         # metrics (5-way taxonomy), runner
  cli.py
configs/        # YAML configs
tests/          # unit tests + synthetic fixtures
benchmarks/     # released open benchmark (Synthea)
paper/          # LaTeX sources (TBD)
```

## Roadmap

- **v1**: diagnoses (ICD-10-CM) — in progress
- **v2**: + labs (LOINC), via Observation resources
- **v3**: + medications (RxNorm), via MedicationRequest resources

The 5-way outcome taxonomy, eval runner, and prompt-caching infrastructure are designed to extend across vocabularies without architectural change.

## License

Code: MIT. Released benchmark data: TBD (will follow Synthea's Apache 2.0). MIMIC-derived data is never distributed.
