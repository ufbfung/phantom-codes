# Phantom Codes

> *Hallucination, Accuracy, and Cost in LLM-Based Medical Concept Normalization.*

A reproducible benchmark for evaluating how well frontier LLMs map degraded clinical data to standardized terminology codes — how often they fabricate codes that don't exist, and at what cost-per-correct they actually deliver.

**Medical concept normalization** is the entity-linking task of mapping in-text mentions of biomedical concepts (diagnoses, labs, medications) to entries in a standardized ontology or controlled vocabulary (ICD-10-CM, LOINC, RxNorm, etc.). It's a foundational task for healthcare data integration, billing, research cohort construction, and AI-assisted documentation.

**Read the paper:** [paper/phantom_codes.pdf](paper/phantom_codes.pdf) (main, IEEE format, 11 pages) · [paper/phantom_codes_supplementary.pdf](paper/phantom_codes_supplementary.pdf) (supplement, 17 pages) · [paper/cover_letter.md](paper/cover_letter.md) · [paper/README.md](paper/README.md) for build details. Headline n=125 Synthea evaluation completed 2026-05-04; per-(model, mode) aggregates at [results/summary/n125_run_v2/](results/summary/n125_run_v2/).

**Status:** v1 complete for diagnoses. Degradation pipeline ([data/degrade.py](src/phantom_codes/data/degrade.py)), 6-way outcome taxonomy ([eval/metrics.py](src/phantom_codes/eval/metrics.py)), ICD-10-CM validator ([data/icd10cm/](src/phantom_codes/data/icd10cm/)), LLM module ([models/llm.py](src/phantom_codes/models/llm.py): Claude + GPT + Gemini in zero-shot + constrained + RAG modes), bi-encoder retrieval baseline ([models/retrieval.py](src/phantom_codes/models/retrieval.py)), string-matching baselines ([models/baselines.py](src/phantom_codes/models/baselines.py)), token + cost tracking ([eval/cost.py](src/phantom_codes/eval/cost.py)), run-manifest sidecars ([eval/manifest.py](src/phantom_codes/eval/manifest.py)), blinded `--infra-only` smoke-test mode, and end-to-end eval runner ([eval/runner.py](src/phantom_codes/eval/runner.py)) are implemented and tested. Manuscript submission-ready for **IEEE Journal of Biomedical and Health Informatics, LLM Special Issue Part II** (deadline 2026-06-01); 11 pages IEEE J-BHI two-column format, 3 figures, 4 tables, ~17-page supplement, accompanied by a companion arXiv tech report on the PubMedBERT fine-tuning.

**v1 scope:** diagnoses (ICD-10-CM). v2+ extends to labs (LOINC) and medications (RxNorm) using the same framework.

## What this is

Take a coded FHIR Condition (e.g., one bearing ICD-10 `E11.9` "Type 2 diabetes mellitus without complications"), strip or degrade the coding fields, and ask: can a model recover the original `(system, code)` — and when it can't, does it admit ignorance or fabricate a plausible-looking-but-wrong code?

The benchmark compares (all configured via [configs/models.yaml](configs/models.yaml); see [paper §III](paper/phantom_codes.pdf) for full results):

- **LLMs** ([models/llm.py](src/phantom_codes/models/llm.py), [models/rag_llm.py](src/phantom_codes/models/rag_llm.py)) — Claude (Opus 4.7, Sonnet 4.6, Haiku 4.5), GPT (5.5, 4o-mini), Gemini (2.5 Pro, 2.5 Flash, 3 Flash Preview), each in three prompting modes: zero-shot, constrained, RAG.
- **Trained models** ([models/retrieval.py](src/phantom_codes/models/retrieval.py), [models/classifier.py](src/phantom_codes/models/classifier.py)) — sentence-transformer retrieval (FAISS) and PubMedBERT classification head.
- **String-matching baselines** ([models/baselines.py](src/phantom_codes/models/baselines.py)) — exact, fuzzy (rapidfuzz), TF-IDF nearest neighbor.

across four degradation modes (implemented at [data/degrade.py](src/phantom_codes/data/degrade.py); D4 abbreviation rules at [data/abbreviate.py](src/phantom_codes/data/abbreviate.py) + [data/abbreviations.yaml](src/phantom_codes/data/abbreviations.yaml)):

| Mode | What's removed | What remains |
|------|----------------|--------------|
| `D1_full` | nothing | full coded resource (control) |
| `D2_no_code` | `coding.code`, `coding.system` | display, text, status, category |
| `D3_text_only` | all coding | canonical `text` field only |
| `D4_abbreviated` | all coding; canonical text replaced with clinical jargon (T2DM, HTN, CKD, …) | one-sentence summary using abbreviations |

D4 is the headline robustness probe: it strips the canonical display tokens that string-matching baselines depend on while preserving the underlying diagnosis. That's where LLMs are forced to actually *retrieve from semantic understanding* — and where hallucinations surface.

## Six-way outcome taxonomy

Every prediction lands in exactly one bucket (implemented in [eval/metrics.py](src/phantom_codes/eval/metrics.py); full rationale in [paper §II](paper/phantom_codes.pdf)). Aligned with prior literature on hierarchical-match terminology (Mullenbach et al. 2018), OOD framing (Hendrycks & Gimpel 2017), and the hallucination survey (Ji et al. 2023) — all cited in [paper/references.bib](paper/references.bib).

- `exact_match` — predicted code equals truth
- `category_match` — same 3-char ICD-10 category (e.g., E11.0 vs E11.9)
- `chapter_match` — same ICD-10 chapter (E vs E), different category
- `out_of_domain` — real ICD-10-CM code, no hierarchical relation to truth
- `no_prediction` — model returned no usable prediction (empty array, refusal, or transient API failure). Distinct from hallucination — nothing was fabricated; the model abstained.
- `hallucination` — predicted code does NOT exist in ICD-10-CM (*narrow* definition: fabrications only; mechanically validated against [data/icd10cm/](src/phantom_codes/data/icd10cm/))

Hallucination and no_prediction are reported separately to surface a downstream-error distinction: an abstaining model emits no spurious code that downstream systems silently mishandle, while a hallucinated code propagates into billing, research, and quality reporting. The taxonomy generalizes to LOINC and RxNorm in v2 with the appropriate hierarchy + validator file.

## Scope: CMS ACCESS Model conditions

We restrict the cohort to ICD-10-CM codes in the [CMS ACCESS Model](https://www.cms.gov/priorities/innovation/innovation-models/access) FHIR Implementation Guide:
- **CKM track**: diabetes (E08/E09/E11/E13), atherosclerotic CVD, CKD stage 3
- **eCKM track**: hypertension, dyslipidemia, prediabetes, obesity

ValueSets bundled at [src/phantom_codes/data/access_valuesets/](src/phantom_codes/data/access_valuesets/).

## Data

- **Primary:** [MIMIC-IV-FHIR v2.1](https://physionet.org/content/mimic-iv-fhir/2.1/) — credentialed access via PhysioNet. Cannot be redistributed; this repo only contains code and synthetic fixtures.
- **Open benchmark:** [Synthea](https://github.com/synthetichealth/synthea)-generated FHIR Bundles. Cohort lives at `benchmarks/synthetic_v1/conditions.ndjson` (deterministic given pinned Synthea SHA + `seed=42`); regenerable from `./scripts/generate_synthea_cohort.sh` + `phantom-codes prepare-synthea`.

### Compliance: PhysioNet's responsible-LLM-use policy

PhysioNet's [responsible-LLM-use policy](https://physionet.org/news/post/llm-responsible-use/) (effective 2025-09-24) prohibits sharing credentialed data with third parties, including commercial LLM APIs. Phantom Codes is **compliant by design** through a three-way data separation:

- **MIMIC-IV-FHIR** stays on the credentialed user's laptop (gitignored under `data/mimic/raw/`) and is consumed only by trained models fine-tuned locally on Apple Silicon (PyTorch MPS). MIMIC content never leaves the laptop.
- **Synthea-generated FHIR Bundles** are the universal benchmark for the LLM evaluation matrix — synthetic, freely redistributable, safe to send to any LLM endpoint.
- **Synthetic fixtures** (in `tests/fixtures/`) are hand-built and contain no MIMIC content; safe for smoke tests and CI.

Trained-model weights are MIMIC-derivative and never committed/pushed/uploaded; the training module ([`training/trainer.py`](src/phantom_codes/training/trainer.py)) sets defensive env vars at import time to disable wandb/mlflow/comet telemetry. Hard rules, prohibitions, and code patterns I (Claude) am bound by during development are enumerated in [CLAUDE.md](CLAUDE.md). If you reproduce this work, your own compliance with PhysioNet's DUA is your responsibility.

## Data setup

The MIMIC arm of the project (PubMedBERT classifier fine-tuning) requires PhysioNet [credentialed access](https://physionet.org/about/citi-course/) to MIMIC-IV-FHIR v2.1 and is local-only by design. The Synthea arm (full LLM evaluation matrix) needs no credentialing. Setup workflow (download paths, GCS / AWS variants, pinned Synthea SHA, expected file sizes) lives in [BENCHMARK.md](BENCHMARK.md).

## Getting started (development)

```bash
uv sync --extra dev
uv run pytest tests/
uv run phantom-codes --help

# End-to-end smoke test on local fixtures (no network, no MIMIC).
# Defaults to `smoke_test_set` from configs/models.yaml (3 baselines +
# retrieval + 1 cheap Haiku call ≈ <$0.10 per run if ANTHROPIC_API_KEY
# is set; LLM models with missing keys are silently skipped).
uv run phantom-codes smoke-test

# To exercise the full headline matrix on fixtures (much higher cost;
# pair with smaller fixtures or --max-records on `evaluate`):
uv run phantom-codes smoke-test --models-set headline_set
```

The full MIMIC pipeline requires [PhysioNet credentialed access](https://physionet.org/about/citi-course/) for MIMIC-IV-FHIR v2.1. See the [Data setup](#data-setup) section above for the manual download + upload workflow (PhysioNet doesn't mirror this dataset to GCS).

## Reproducing the headline benchmark

The full LLM evaluation matrix on Synthea is reproducible **without** PhysioNet credentialing. The four-step sequence (setup Synthea → generate cohort → smoke-validate → headline run → report) and the full reproducibility checklist (pinned Synthea SHA, generation seed, model registry, pricing snapshot, expected costs/runtimes, troubleshooting) live in [BENCHMARK.md](BENCHMARK.md). Headline cost: ~$50 in API spend, ~24–36 hr wall clock at provider rate limits.

What's reproducible without MIMIC credentialing: ✅ Synthea cohort, ✅ all LLM evaluation, ✅ all baseline + retrieval models, ✅ paper rebuild. What's not: ❌ the trained PubMedBERT classifier checkpoint (MIMIC-derivative; bring your own if you have MIMIC access).

## Repository layout

```
src/phantom_codes/
  data/         # fhir_loader, code_set, degrade, abbreviate, disease_groups
                # icd10cm/  (bundled CMS FY2026 codes for hallucination detection)
                # access_valuesets/  (CMS ACCESS Model FHIR ValueSets)
  models/       # base ABC; llm (Anthropic, OpenAI, Google); rag_llm;
                #   retrieval (sentence-transformer + cosine); baselines;
                #   classifier (PubMedBERT inference wrapper)
  training/     # PyTorch fine-tuning: dataset, trainer, devices, seeding;
                #   MPS-native, telemetry-disabled by default
  eval/         # metrics (6-way taxonomy), runner, infra (blinded
                #   smoke-test), cost (pricing → $), manifest (run sidecars)
  cli.py
configs/        # YAML configs (data.yaml, pricing.yaml, training.yaml)
docs/learning/  # PyTorch primers (foundations → data → models → loop → practical)
scripts/        # demo_minimal_training.py (synthetic-fixture end-to-end demo)
tests/          # unit tests + synthetic fixtures (205 tests, ruff clean)
benchmarks/     # released open benchmark (Synthea — TBD)
paper/                              # see paper/README.md for build details
  phantom_codes/    # IEEE J-BHI main manuscript + supplement (pure LaTeX, modular)
  pubmedbert/       # arXiv companion tech report (PubMedBERT fine-tuning)
  references.bib    # shared bibliography (biblatex, ieee style)
  Makefile          # `make phantom_codes` / `make pubmedbert` / `make snapshot-all`
  scripts/flatten_paper.py            # modular .tex → flat .tex (auto-run by snapshot)
  figures/                            # heatmap + cost frontier + D4 outcome stack (.py + .pdf)
  cover_letter.md                     # IEEE J-BHI cover letter
  phantom_codes.tex                   # GENERATED — single self-contained .tex for IEEE submission
  phantom_codes_supplementary.tex     # GENERATED — single self-contained supplement .tex
  phantom_codes.pdf                   # committed snapshot of the main manuscript
  phantom_codes_supplementary.pdf     # committed supplement snapshot
  pubmedbert_finetuning.pdf           # committed arXiv tech-report snapshot
```

The two GENERATED `.tex` files are auto-flattened from the modular
sources every time `make snapshot-phantom_codes` runs, so IEEE
reviewers see exactly the source we render. Edit the modular files
under `paper/phantom_codes/`; never edit the flat `.tex` directly.
Full paper-build documentation in [paper/README.md](paper/README.md).

## Roadmap

- **v1**: diagnoses (ICD-10-CM) — manuscript submission-ready for IEEE J-BHI LLM Special Issue Part II (deadline 2026-06-01)
- **v2**: + labs (LOINC), via Observation resources
- **v3**: + medications (RxNorm), via MedicationRequest resources

The 6-way outcome taxonomy, eval runner, and prompt-caching infrastructure are designed to extend across vocabularies without architectural change.

## Contributions and references

Phantom Codes contributes (1) **frontier-model recency** — 24 (model, prompting-mode) configurations across the most recent Claude / GPT / Gemini generations, several released within weeks of the evaluation date; (2) **cost as a first-class outcome alongside accuracy** — cost-per-correct as the deployment-decisive normalization, with the Pareto frontier in the accuracy-cost plane explicit; and (3) **methodological instruments** — a 6-way outcome taxonomy (fabrication and abstention as mutually-exclusive buckets), a within-model ablation across zero-shot / constrained / RAG, and a D4 abbreviation-stress robustness probe.

Full related-work narrative + literature positioning is in [paper §I (Background and Significance)](paper/phantom_codes.pdf) and the comparison tables in §IV. The complete curated bibliography (each entry annotated with a `% Why we cite:` comment block) lives at [paper/references.bib](paper/references.bib).

## License

Code in this repository is released under the [MIT License](LICENSE).
Third-party attributions required by upstream licenses (Synthea Apache
2.0; SNOMED CT IHTSDO posture propagated via Synthea; ICD-10-CM /
CMS ACCESS Model public-domain provenance) are reproduced in
[NOTICE](NOTICE).

- **MIMIC-derived data is never distributed here** — PhysioNet's Credentialed Health Data License governs the data and prohibits redistribution. Obtain MIMIC-IV-FHIR via your own [PhysioNet credentialing](https://physionet.org/about/citi-course/).
- **Trained model weights are not released** from this repo. Weights derived from MIMIC are typically redistributed only via PhysioNet's "MIMIC-IV Models" channel under similar credentialing.
- **Released benchmark data** (Synthea-generated, planned for `benchmarks/synthetic_v1/`) will follow Synthea's Apache 2.0 license; see [NOTICE](NOTICE) for the full Synthea + SNOMED CT attribution.
