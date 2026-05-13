# Phantom Codes

> *Hallucination, Accuracy, and Cost in LLM-Based Medical Concept Normalization.*

A reproducible benchmark for evaluating how well frontier LLMs map degraded clinical data to standardized terminology codes — how often they fabricate codes that don't exist, and at what cost-per-correct they actually deliver.

**Medical concept normalization** is the entity-linking task of mapping in-text mentions of biomedical concepts (diagnoses, labs, medications) to entries in a standardized ontology or controlled vocabulary (ICD-10-CM, LOINC, RxNorm, etc.). It's a foundational task for healthcare data integration, billing, research cohort construction, and AI-assisted documentation.

**Status:** v1 complete for diagnoses. Degradation pipeline, 6-way outcome taxonomy, ICD-10-CM validator, LLM module (Claude + GPT + Gemini in zero-shot + constrained + RAG modes), bi-encoder retrieval baseline, constrained-only baselines (exact / fuzzy / TF-IDF), token + cost tracking, run-manifest sidecars, blinded `--infra-only` smoke-test mode, and end-to-end eval runner are implemented and tested. PubMedBERT classifier fine-tuned locally on MIMIC-IV-FHIR via PyTorch MPS. Headline n=125 Synthea evaluation completed 2026-05-04 (per-(model, mode) aggregates in [results/summary/n125_run_v2/](results/summary/n125_run_v2/)). Manuscript submission-ready for **JAMIA Research and Applications** (~3,200 words, ≤250-word structured abstract, 3 figures, 4 tables, accompanied by supplementary materials and a companion arXiv tech report on the PubMedBERT fine-tuning).

**v1 scope:** diagnoses (ICD-10-CM). v2+ extends to labs (LOINC) and medications (RxNorm) using the same framework.

## What this is

Take a coded FHIR Condition (e.g., one bearing ICD-10 `E11.9` "Type 2 diabetes mellitus without complications"), strip or degrade the coding fields, and ask: can a model recover the original `(system, code)` — and when it can't, does it admit ignorance or fabricate a plausible-looking-but-wrong code?

The benchmark compares:

- **LLMs** — Claude (Opus 4.7, Sonnet 4.6, Haiku 4.5), GPT (5.5, 4o-mini), Gemini (2.5 Pro, 2.5 Flash, 3 Flash Preview). Three prompting modes: **zero-shot** (no menu, primary experiment), **constrained** (fixed candidate list, hallucination control), and **RAG** (per-record retrieved candidates).
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

## Six-way outcome taxonomy

Every prediction lands in exactly one bucket. Aligned with the literature: hierarchical-match terminology from [Mullenbach et al. 2018 (CAML)](https://aclanthology.org/N18-1100/), OOD framing from [Hendrycks & Gimpel 2017](https://arxiv.org/abs/1610.02136), hallucination/fabrication from [Ji et al. 2023](https://dl.acm.org/doi/10.1145/3571730).

- `exact_match` — predicted code equals truth
- `category_match` — same 3-char ICD-10 category (e.g., E11.0 vs E11.9)
- `chapter_match` — same ICD-10 chapter (E vs E), different category
- `out_of_domain` — real ICD-10-CM code, no hierarchical relation to truth
- `no_prediction` — model returned no usable prediction (empty array, refusal, or transient API failure). Distinct from hallucination — nothing was fabricated; the model abstained.
- `hallucination` — predicted code does NOT exist in ICD-10-CM (*narrow* definition: fabrications only)

Hallucination and no-prediction are reported separately to surface a downstream-error distinction: an abstaining model emits no spurious code that downstream systems silently mishandle, while a hallucinated code propagates into billing, research, and quality reporting. Both fail to produce a usable prediction, but their downstream costs differ. (Refined 2026-05-04 from a 5-bucket version that lumped abstention into hallucination — see metrics.py docstring for the rationale.)

The taxonomy generalizes naturally to LOINC and RxNorm in v2: same definitions with the appropriate hierarchy and validator file.

## Scope: CMS ACCESS Model conditions

We restrict the cohort to ICD-10-CM codes in the [CMS ACCESS Model](https://www.cms.gov/priorities/innovation/innovation-models/access) FHIR Implementation Guide:
- **CKM track**: diabetes (E08/E09/E11/E13), atherosclerotic CVD, CKD stage 3
- **eCKM track**: hypertension, dyslipidemia, prediabetes, obesity

ValueSets bundled at [src/phantom_codes/data/access_valuesets/](src/phantom_codes/data/access_valuesets/).

## Data

- **Primary:** [MIMIC-IV-FHIR v2.1](https://physionet.org/content/mimic-iv-fhir/2.1/) — credentialed access via PhysioNet. Cannot be redistributed; this repo only contains code and synthetic fixtures.
- **Open benchmark:** [Synthea](https://github.com/synthetichealth/synthea)-generated FHIR Bundles. Cohort lives at `benchmarks/synthetic_v1/conditions.ndjson` (deterministic given pinned Synthea SHA + `seed=42`); regenerable from `./scripts/generate_synthea_cohort.sh` + `phantom-codes prepare-synthea`.

### Compliance: PhysioNet's responsible-LLM-use policy

PhysioNet's [responsible-LLM-use policy](https://physionet.org/news/post/llm-responsible-use/) (effective 2025-09-24) **explicitly prohibits** sharing credentialed data with third parties, including sending it through commercial LLM APIs (Anthropic, OpenAI, Google, etc.). Phantom Codes is **compliant by design** through a deliberate three-way separation:

- **MIMIC-IV-FHIR** is downloaded directly to the credentialed user's laptop (gitignored under `data/mimic/raw/`) and consumed only by trained models (PubMedBERT classifier head, retrieval encoder) fine-tuned locally on Apple Silicon via PyTorch's MPS backend. **MIMIC content never leaves the laptop** — not to any cloud bucket, training service, or LLM endpoint. (Cloud-based training paths are supported by the same code if a future researcher prefers Vertex AI / AWS / Azure with appropriate compliance review, but our reference implementation is fully local.)
- **Synthea-generated FHIR Bundles** are the universal benchmark on which every model — frontier LLMs (Claude, GPT, Gemini), trained models, and baselines — is evaluated. Synthea data contains no real patient information, is freely redistributable, and can be sent to any LLM endpoint without policy concern.
- **Synthetic fixtures** (in `tests/fixtures/`) are hand-built and contain no MIMIC content; safe for smoke tests and CI.

This separation provides three simultaneous benefits: (1) full compliance with PhysioNet's DUA, (2) full reproducibility — anyone can replicate the headline benchmark on Synthea without PhysioNet credentialing, and (3) the strongest possible auditability story since MIMIC content never traverses any network beyond the initial download from PhysioNet.

**Trained-model weights are MIMIC-derivative.** Checkpoints under `models/checkpoints/` are gitignored and **never** committed to git, pushed to HuggingFace Hub, or uploaded to any cloud bucket. The training module ([`src/phantom_codes/training/trainer.py`](src/phantom_codes/training/trainer.py)) sets defensive environment variables at import time to disable wandb / mlflow / comet cloud telemetry, so accidental imports of those libraries can't leak training metrics either.

**If you reproduce this work**, you are responsible for ensuring your own use complies with PhysioNet's DUA and current policy. We do not redistribute MIMIC data or any model weights derived from it.

## Data setup

PhysioNet hosts MIMIC-IV-FHIR v2.1 via **HTTPS download and AWS S3 only — there is no GCS mirror** (despite PhysioNet's general support for Google account linking, this specific dataset isn't mirrored to GCS). Three supported paths depending on where you want the data to live; **Option 1 (local-only) is our reference setup** and the path used in this paper.

### Option 1 (recommended): Local-only — no cloud at all

The simplest path and the strongest compliance story: MIMIC content never touches any cloud infrastructure beyond the initial PhysioNet download. After completing PhysioNet credentialing and CITI training:

```bash
# 1. Download the files we need for v1 (~303 MB total) directly into the
#    repo's gitignored data/ directory:
mkdir -p data/mimic/raw && cd data/mimic/raw

wget --user YOUR_PHYSIONET_USERNAME --ask-password https://physionet.org/files/mimic-iv-fhir/2.1/fhir/MimicCondition.ndjson.gz
wget --user YOUR_PHYSIONET_USERNAME --ask-password https://physionet.org/files/mimic-iv-fhir/2.1/fhir/MimicPatient.ndjson.gz
wget --user YOUR_PHYSIONET_USERNAME --ask-password https://physionet.org/files/mimic-iv-fhir/2.1/fhir/MimicEncounter.ndjson.gz

cd ../../..

# 2. Run prepare against the local file (override --source flag):
uv run phantom-codes prepare \
  --source data/mimic/raw/MimicCondition.ndjson.gz \
  --local-out data/derived
```

`data/` is gitignored so MIMIC content never gets accidentally committed. PubMedBERT training (P2 in the project backlog) also reads from this directory locally, so the data never moves.

### Option 2: GCP-based (download → upload to your own GCS bucket)

If you need MIMIC accessible from cloud-hosted compute (e.g., a Vertex AI Workbench instance for distributed training), upload the locally-downloaded files to your own GCS bucket:

```bash
# After completing the wget step from Option 1:

# Create your own GCS bucket (matching us-central1 keeps egress free):
gcloud storage buckets create gs://YOUR_BUCKET/ \
  --project=YOUR_PROJECT --location=US-CENTRAL1 --uniform-bucket-level-access

# Authenticate gcsfs to use the right project for billing:
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT

# Upload at the path our pipeline reads from:
gcloud storage cp data/mimic/raw/*.ndjson.gz gs://YOUR_BUCKET/mimic/raw/

# Update configs/data.yaml: derived_bucket: gs://YOUR_BUCKET

# Verify the upload landed correctly:
uv run phantom-codes check-data
```

If using cloud-hosted training, the BAA / data-residency review of your cloud provider is your responsibility.

### Option 3: AWS-based pipeline

PhysioNet provides AWS S3 access for credentialed users — see the "Files → Access the files" section at [physionet.org/content/mimic-iv-fhir/2.1/](https://physionet.org/content/mimic-iv-fhir/2.1/) for the documented setup. Our pipeline reads from GCS today; using AWS S3 would require swapping `gcsfs` for `s3fs` in [src/phantom_codes/data/fhir_loader.py](src/phantom_codes/data/fhir_loader.py) and [src/phantom_codes/data/gcs_setup.py](src/phantom_codes/data/gcs_setup.py). Not currently supported in v1 but a contained change if you need it.

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

The full evaluation matrix on Synthea — every model in the lineup
evaluated on the same synthetic FHIR cohort — is fully reproducible
**without** PhysioNet credentialing. Synthea data is freely
redistributable and contains no real patient information.

Quick sequence (each step has a single command):

```bash
# One-time setup
brew install openjdk@17                 # Java 17+ for Synthea
./scripts/setup_synthea.sh              # clone + build Synthea v4.0.0

# Generate the 2,000-patient Synthea cohort (~5–10 min, ~30 GB peak)
./scripts/generate_synthea_cohort.sh

# Build the inference dataset (deduped, scope-filtered, 4 modes)
uv run phantom-codes prepare-synthea
rm -rf benchmarks/synthetic_v1/raw       # frees disk; bundles regenerable

# Smoke-validate before the headline run (<$1)
uv run phantom-codes evaluate \
    --models-set smoke_test_set --max-records 50 --max-cost-usd 5

# Headline run — reproduces the v1 paper's n=125 cohort
# (~24-36 hr wall clock at provider rate limits; ~$50 realized cost)
uv run phantom-codes evaluate \
    --models-set headline_set --max-records 500 --max-cost-usd 500

# Generate paper-ready tables
uv run phantom-codes report --csv results/raw/headline_*.csv
```

> **`--max-records` semantics:** The flag counts long-format cohort
> rows (one row per resource × degradation mode), not unique
> resources. `--max-records 500` therefore yields **125 unique
> resources × 4 modes = 500 EvalRecord items** — this matches the v1
> paper's headline. To run on n=500 unique resources, pass
> `--max-records 2000`. See [BENCHMARK.md](BENCHMARK.md) for the full
> reproduction guide.

**Full reproduction guide**: see [BENCHMARK.md](BENCHMARK.md) for
prerequisites, expected costs/runtimes, the SNOMED→ICD-10-CM
curation workflow, troubleshooting, and the reproducibility
checklist (Synthea SHA pin, generation seed, model registry, pricing
snapshot).

What's reproducible without MIMIC credentialing:
- ✅ All Synthea cohort generation + preparation
- ✅ All LLM evaluation (frontier APIs hit Synthea inputs only)
- ✅ All baseline + retrieval models
- ✅ All report tables + paper rebuild
- ❌ The trained PubMedBERT classifier checkpoint (MIMIC-derivative;
  not redistributed per PhysioNet's DUA — bring your own checkpoint
  if you have MIMIC access)

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
paper/
  phantom_codes/    # JAMIA main manuscript + supplement (pure LaTeX, modular)
    main.tex            # main manuscript master
    supplementary.tex   # supplement master
    sections/           # 00_title_page → 06_conclusion (.tex)
    supp_sections/      # S1, S2, S5 (.tex)
  pubmedbert/       # arXiv companion tech report (PubMedBERT fine-tuning)
    main.tex            # tech report master
    sections/           # 00_introduction → 06_appendix (.tex)
  references.bib    # shared bibliography (biblatex Vancouver)
  Makefile          # `make phantom_codes` / `make pubmedbert` / `make snapshot-all`
  scripts/flatten_paper.py     # modular .tex → flat .tex (auto-run by snapshot)
  figures/          # 3 figures (heatmap, cost frontier, D4 outcome stack)
  cover_letter.md   # submission cover letter
  phantom_codes.tex                 # GENERATED — single self-contained .tex for JAMIA
  phantom_codes_supplementary.tex   # GENERATED — single self-contained .tex for JAMIA
  phantom_codes.pdf                 # committed snapshot of the main manuscript
  phantom_codes_supplementary.pdf   # committed supplement snapshot
  pubmedbert_finetuning.pdf         # committed arXiv tech-report snapshot
```

The two GENERATED `.tex` files are auto-flattened from the modular
sources every time `make snapshot-phantom_codes` runs, so JAMIA
reviewers see exactly the source we render. Edit the modular files
under `paper/phantom_codes/`; never edit the flat `.tex` directly.

## Roadmap

- **v1**: diagnoses (ICD-10-CM) — manuscript submission-ready for JAMIA Research and Applications
- **v2**: + labs (LOINC), via Observation resources
- **v3**: + medications (RxNorm), via MedicationRequest resources

The 6-way outcome taxonomy, eval runner, and prompt-caching infrastructure are designed to extend across vocabularies without architectural change.

## Inspirations and related work

Phantom Codes draws on five recent (2024-2026) lines of work in LLM-based clinical coding:

- **From Almeida et al. 2025 (ICPC-2 benchmark, NeurIPS GenAI4Health):** the multi-axis evaluation table — F1 paired with cost, latency, token usage, and format adherence — gives a more honest picture of LLM viability than accuracy alone. We adopt the same multi-axis framing for our headline results.
- **From Bhatti et al. 2025 (MAX-EVAL-11):** clinically-informed scoring with weighted reward by code relevance and diagnostic specificity. Our 6-way outcome taxonomy is a sibling of this idea — discrete buckets instead of continuous weights, but the same intuition that not every wrong answer is equally wrong.
- **From Motzfeldt et al. 2025 (Code Like Humans, EMNLP Findings):** the "agentic" alternative to single-shot prompting — sequential search/verify/predict over the ICD index. We treat their approach as a third prompting mode to add (alongside our `zeroshot` and `constrained`) and test whether agent decomposition reduces hallucination on D4 inputs.
- **From Kim et al. 2025 (Medical Hallucination in Foundation Models):** a strict definition of medical hallucination as "factually incorrect, logically inconsistent, or unsupported by authoritative clinical evidence in ways that could alter clinical decisions." Our `hallucination` bucket (code does not exist in ICD-10-CM) is a narrow, mechanically-checkable instance of their broader definition.
- **From Li et al. 2025 (ICD Coding Rationales):** the faithfulness/plausibility distinction for evaluating rationales, with a MIMIC-IV/ICD-10 rationale-annotated dataset. A natural v2 extension is to ask each LLM for a rationale alongside its code and evaluate both axes.

What we contribute on top of this prior work:

- **Frontier-model recency** — 24 (model, prompting-mode) configurations spanning Claude Opus 4.7 / Sonnet 4.6 / Haiku 4.5, GPT-5.5 / GPT-4o-mini, and Gemini 2.5 Pro / 2.5 Flash / 3 Flash Preview. Several were released within weeks of the evaluation date; prior work reports findings on 2023–2024 vintage models.
- **Cost as a first-class outcome alongside accuracy** — every prediction records token usage and dollar cost at runtime; we report cost-per-correct (USD per exact-match outcome) as the deployment-decisive normalization and identify the small Pareto frontier in the accuracy-cost plane.
- **Methodological instruments that make the above two contributions interpretable** — a 6-way outcome taxonomy (fabrication and abstention as explicit, mutually-exclusive buckets), a within-model ablation across zero-shot / constrained / RAG, and a D4 abbreviation-stress mode that strips lexical signal so semantic-mapping behavior is testable separately from lexical lookup.

## References

The full curated bibliography (16+ references organized into foundational landmarks, direct LLM-on-medical-coding evaluations, and clinical-hallucination studies) lives at [paper/references.bib](paper/references.bib) — BibTeX entries with `% Why we cite:` annotation comments preserving the rationale for each. The five papers most directly inspiring this work:

1. **Soroush et al. 2024 (NEJM AI)** — *Large Language Models Are Poor Medical Coders.* GPT-4 achieves 46% / 34% / 50% exact match on ICD-9 / ICD-10 / CPT respectively; substantial hallucination of non-existent codes. [doi:10.1056/AIdbp2300040](https://ai.nejm.org/doi/full/10.1056/AIdbp2300040)
2. **Almeida et al. 2025 (NeurIPS GenAI4Health)** — *Large Language Models as Medical Code Selectors.* 33-LLM benchmark on ICPC-2; multi-axis evaluation (F1, cost, latency, format adherence). [arXiv:2507.14681](https://arxiv.org/abs/2507.14681)
3. **Bhatti et al. 2025** — *MAX-EVAL-11.* MIMIC-III mapped to full-spectrum ICD-11 with weighted scoring by code relevance and diagnostic specificity. [doi:10.1101/2025.10.30.25339130](https://www.medrxiv.org/content/10.1101/2025.10.30.25339130v1)
4. **Motzfeldt et al. 2025 (EMNLP Findings)** — *Code Like Humans.* Agentic ICD-10 coding traversing the alphabetic index sequentially. [arXiv:2509.05378](https://arxiv.org/abs/2509.05378)
5. **Kim et al. 2025** — *Medical Hallucinations in Foundation Models and Their Impact on Healthcare.* Defines medical hallucination broadly; benchmarks 11 foundation models. [arXiv:2503.05777](https://arxiv.org/abs/2503.05777)

Foundational references for the 6-way outcome taxonomy:

- Mullenbach et al. 2018 (CAML, NAACL) — hierarchical-match terminology. [aclanthology.org/N18-1100](https://aclanthology.org/N18-1100/)
- Hendrycks & Gimpel 2017 (ICLR) — OOD detection baseline. [arXiv:1610.02136](https://arxiv.org/abs/1610.02136)
- Ji et al. 2023 (ACM Computing Surveys) — hallucination survey. [doi:10.1145/3571730](https://dl.acm.org/doi/10.1145/3571730)

## License

Code in this repository is released under the [MIT License](LICENSE).
Third-party attributions required by upstream licenses (Synthea Apache
2.0; SNOMED CT IHTSDO posture propagated via Synthea; ICD-10-CM /
CMS ACCESS Model public-domain provenance) are reproduced in
[NOTICE](NOTICE).

- **MIMIC-derived data is never distributed here** — PhysioNet's Credentialed Health Data License governs the data and prohibits redistribution. Obtain MIMIC-IV-FHIR via your own [PhysioNet credentialing](https://physionet.org/about/citi-course/).
- **Trained model weights are not released** from this repo. Weights derived from MIMIC are typically redistributed only via PhysioNet's "MIMIC-IV Models" channel under similar credentialing.
- **Released benchmark data** (Synthea-generated, planned for `benchmarks/synthetic_v1/`) will follow Synthea's Apache 2.0 license; see [NOTICE](NOTICE) for the full Synthea + SNOMED CT attribution.
