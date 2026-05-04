# Reproducing the Phantom Codes Benchmark

This guide walks through reproducing the headline experiment from
*Phantom Codes: Hallucination in LLM-Based Clinical Concept
Normalization* on Synthea-generated synthetic data. No PhysioNet
credentialing required — Synthea data is freely redistributable
(Apache 2.0) and contains no real patient information.

For the broader project context (motivation, methods, model lineup),
see the top-level [README](README.md). For paper-rebuild instructions,
see [paper/README.md](paper/README.md).

---

## What this reproduces

The full evaluation matrix:

- **3 string-matching baselines** (exact, fuzzy, TF-IDF)
- **1 frozen sentence-transformer retrieval baseline**
- **24 frontier LLM configurations** — every LLM in all 3 prompting
  modes (zeroshot / constrained / rag) for full within-model ablation
  across providers:
  - Anthropic Claude × 3 modes each (Haiku 4.5, Sonnet 4.6, Opus 4.7) = 9
  - OpenAI GPT × 3 modes each (GPT-5.5, GPT-4o-mini) = 6
  - Google Gemini × 3 modes each (2.5 Pro, 2.5 Flash, 3 Flash Preview) = 9

> **Note on Gemini 3.1 Pro Preview**: omitted from the headline matrix
> due to preview-tier API quota constraints (Tier 1 cap of 250
> requests/day vs. the ~6,000 calls required for full coverage).
> Partial-coverage data from a 21-record archived run is presented
> separately as supplementary material; see `BACKLOG.md` for the
> follow-up analysis plan.

against **Synthea-generated FHIR Conditions** scoped to the [CMS ACCESS
Model](https://dsacms.github.io/cmmi-access-model/) condition set
(diabetes, hypertension, dyslipidemia, prediabetes, obesity, CKD-3,
ASCVD/cerebrovascular). Each condition is materialized into four
degradation modes (D1_full / D2_no_code / D3_text_only /
D4_abbreviated) and scored via the 5-way outcome taxonomy
(exact_match / category_match / chapter_match / out_of_domain /
hallucination).

The trained PubMedBERT classifier described in §Methods is omitted
from this guide — its checkpoint is MIMIC-derivative (per PhysioNet's
DUA, weights are not released). Reproducers can substitute their own
trained classifier checkpoint at `models/checkpoints/pubmedbert/best_*.pt`
if they have one; the registry will pick it up via the `pubmedbert:classifier`
entry in [`configs/models.yaml`](configs/models.yaml).

## Expected cost and runtime

| Stage | Time | Cost |
|---|---|---|
| Setup (one-time) | 5–10 min | $0 |
| Synthea cohort generation (one-time) | 5–10 min | $0 |
| Inference dataset preparation | 2–5 min | $0 |
| Smoke validation (recommended before headline) | 2–5 min | <$1 |
| **Headline evaluation run** | **24–36 hours** | **$80–300 typical, $500 hard cap** |
| Report generation | <1 min | $0 |

LLM API pricing fluctuates; figures above reflect 2026-Q2 rates with
Anthropic prompt caching active. The runner has a configurable hard
cost cap (`--max-cost-usd`) with soft warnings at 5/10/25/50/75/90%
of cap. If the cap fires mid-run, partial results stay durable on
disk via incremental CSV writes — no work is lost.

## Prerequisites

### System

- macOS or Linux
- ~50 GB free disk space (Synthea generation peak; the persistent
  footprint after `prepare-synthea` + raw-bundle cleanup is ~100 MB)
- Python 3.11+
- Java 17+ (Synthea v4.0.0 requirement; on macOS:
  `brew install openjdk@17`)

### Software

- [`uv`](https://github.com/astral-sh/uv) — project's Python package
  manager. Install via `pip install uv` or `brew install uv`.
- `git` (for cloning Synthea at the pinned SHA)

### API keys

LLM evaluation requires API keys for the providers in the headline
set. **Missing keys cause those models to be skipped silently with a
warning** — partial-set evaluation is supported. Set in `.env`
(gitignored) or your shell environment:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."
```

Verify keys before the headline run:

```bash
uv run phantom-codes verify-keys
```

This sends one cheap test call per provider (~$0.001 total) and
confirms auth + schema parsing.

## Reproduction sequence

Run these commands in order from the repo root.

### 1. Install Python dependencies

```bash
uv sync --extra dev
```

### 2. Install Synthea

```bash
./scripts/setup_synthea.sh
```

Clones Synthea v4.0.0 into `tools/synthea/` (gitignored), builds the
runnable JAR, and verifies the install. ~5 min.

### 3. Generate the Synthea cohort

```bash
./scripts/generate_synthea_cohort.sh
```

Reads [`configs/synthea.yaml`](configs/synthea.yaml) (population_size:
2000, seed: 42 by default), generates ~2000 synthetic patients, emits
one FHIR R4 Bundle JSON per patient to
`benchmarks/synthetic_v1/raw/fhir/`. Each Condition gets dual SNOMED
+ ICD-10-CM coding via the curated map at
[`data/synthea/snomed_to_icd10cm.json`](data/synthea/snomed_to_icd10cm.json).
~5–10 min wall-clock, ~30 GB peak disk.

### 4. (Optional) Verify cohort coverage

```bash
uv run python scripts/inventory_synthea_snomed.py \
    --bundles-dir benchmarks/synthetic_v1/raw/fhir \
    --out data/synthea/snomed_inventory_full.json \
    --compare-with data/synthea/snomed_to_icd10cm.json
```

Walks the generated cohort, enumerates SNOMED codes that appear, and
diffs against the curated map. Should report `✓ no new SNOMED codes
— curated map + exclusions cover full inventory`. If new codes
surface, see [`data/synthea/README.md`](data/synthea/README.md) for
the curation workflow.

### 5. Build the inference dataset

```bash
uv run phantom-codes prepare-synthea
```

Walks `benchmarks/synthetic_v1/raw/fhir/`, extracts Condition
resources, deduplicates per `(patient, ICD code)` pair, applies the 4
degradation modes, and writes a single ndjson at
`benchmarks/synthetic_v1/conditions.ndjson`. ~2–5 min.

After this step, the raw FHIR bundles are no longer needed:

```bash
rm -rf benchmarks/synthetic_v1/raw  # frees ~30 GB
```

The bundles are reproducible from the generation script + seed.

### 6. Smoke-validate the evaluation pipeline (recommended)

```bash
uv run phantom-codes evaluate \
    --models-set smoke_test_set \
    --max-records 50 \
    --max-cost-usd 5
```

Runs the 5 cheapest models (3 baselines + retrieval + 1 Haiku zeroshot
call) against 50 records. Cost: <$1. Validates:
- API keys are working
- Cohort flows cleanly through every model class
- Streaming CSV + manifest sidecar both write successfully
- Cost monitor reports nonzero spend (proving pricing.yaml lookup works)

If anything errors here, fix before proceeding to the headline run.

### 7. Run the headline evaluation

```bash
uv run phantom-codes evaluate \
    --models-set headline_set \
    --max-records 500 \
    --max-cost-usd 500
```

This is the paper's headline experiment — every model in the matrix
evaluated on 500 records × 4 modes (~2000 prediction rows per model;
29 entries in `headline_set` including the trained classifier if its
checkpoint is present, otherwise 28). 24–36 hours wall-clock at
typical provider rate limits (≈15–25 records/hr on the slowest LLM
when iterating through all 3 prompting modes per record); $80–300
typical cost.

Output (timestamped):
- `results/raw/headline_<utc>.csv` — per-prediction long-format CSV
- `results/raw/headline_<utc>.manifest.yaml` — run metadata sidecar
  (token totals, cost breakdown, models used, git SHA, timing)

Streaming writes mean partial results survive any interruption
(Ctrl+C, network failure, cost-cap abort, OOM kill). Just re-run with
the same `--max-records` and the appended writes pick up where they
left off.

#### Spot-check progress during the run (optional but recommended)

The headline run is multi-hour and not always cheap. Run the blinded
structural spot-check tool from another terminal to confirm models are
firing correctly without surfacing the actual results:

```bash
uv run python scripts/blinded_progress_check.py \
    --csv results/raw/headline_<utc>.csv \
    --models-config configs/models.yaml \
    --models-set headline_set \
    --max-records 500 --max-cost-usd 500
```

The script reads only structural / wiring fields (model coverage,
error tally per model, latency profile, cost trajectory, cache
behavior) — it never touches `outcome`, `pred_code`, or any
ground-truth column, so you can run it as many times as you want
without biasing the analysis.

Recommended cadence on a 500-record run:

| Stage | When | Posture |
|---|---|---|
| **EARLY** | ~10% in (~50 rec, ~1 hr) | Aggressive — abort if any model >50% errors or coverage broken. <$10 spent; aborting saves 90% of budget. |
| **MID** | ~50% in (~250 rec, ~6 hr) | Trend-oriented — abort if a NEW error type appears or projected cost > 80% of cap. Saves 50%. |
| **LATE** | ~75% in (~375 rec, ~9 hr) | Validation only — by here, finishing is cheaper than re-running. Just confirm shape matches the mid checkpoint. |

The script's exit code is `0` if all checks pass, `1` if any
WARN-level issue is flagged, or `2` if the CSV is missing. The
verdict block at the bottom lists every triggered issue with concrete
abort/continue recommendations. See the docstring in
[`scripts/blinded_progress_check.py`](scripts/blinded_progress_check.py)
for the full check list and trigger thresholds.

### 8. Generate paper-ready tables

```bash
uv run phantom-codes report --csv results/raw/headline_*.csv
```

Produces five tables in `results/summary/<run-id>/`:

- **`headline.csv`** — outcome distribution per (model, mode) — the §3
  Results centerpiece
- **`hallucination.csv`** — per-mode hallucination rate by model with
  Wilson 95% confidence intervals
- **`top_k_lift.csv`** — top-1 vs top-5 exact-match comparison
- **`cost_per_correct.csv`** — $ per correct prediction by model
- **`per_bucket_cost.csv`** — cost decomposition by outcome bucket

Plus a combined `headline.md` markdown report with all five tables
formatted side-by-side for direct paste into the paper.

## Reproducibility checklist

To verify your reproduction matches the published numbers:

1. **Synthea version**: pinned to `0185c09ea9d10a822c6f5f3ef9bdcbcbe960c813`
   (v4.0.0) in [`scripts/setup_synthea.sh`](scripts/setup_synthea.sh).
   Don't bump unless re-running the SNOMED inventory step
   (Section 4 above) confirms no new in-scope codes appear.
2. **Generation seed**: `42` in [`configs/synthea.yaml`](configs/synthea.yaml).
3. **SNOMED→ICD-10-CM map**: committed at
   [`data/synthea/snomed_to_icd10cm.json`](data/synthea/snomed_to_icd10cm.json).
4. **ACCESS scope filter**: applied via
   [`src/phantom_codes/data/access_valuesets/`](src/phantom_codes/data/access_valuesets/)
   ValueSets (CMS ACCESS Model FHIR IG v0.9.6).
5. **Model registry**: [`configs/models.yaml`](configs/models.yaml)
   `headline_set` defines the 29-entry matrix (28 if no trained
   PubMedBERT checkpoint is present locally) — every LLM in all 3
   prompting modes for full within-model ablation. Gemini 3.1 Pro
   Preview is intentionally excluded from the headline matrix
   pending Tier 2+ quota; see the `# gemini-3.1-pro-preview removed`
   block in `configs/models.yaml` and the partial-coverage analysis
   note in `BACKLOG.md`.
6. **Pricing snapshot**: [`configs/pricing.yaml`](configs/pricing.yaml)
   captures provider rates at the time of run. Update via vendor
   pricing pages if rerun much later.

The cohort manifest at `benchmarks/synthetic_v1/manifest.yaml`
records the Synthea SHA, seed, condition counts per ICD code, and a
SHA-256 checksum of the produced ndjson. Two reproducers should
generate byte-identical cohorts given the same Synthea version + seed.

## Troubleshooting

### "No space left on device" during Synthea generation

Synthea writes ~30 GB of patient bundles. Free space and re-run from
Section 3:

```bash
rm -rf benchmarks/synthetic_v1/raw  # if a prior run left corrupt output
df -h ~  # confirm >50 GB free
./scripts/generate_synthea_cohort.sh
```

If you're consistently disk-tight, edit
[`configs/synthea.yaml`](configs/synthea.yaml) and reduce
`population_size` from 2000 to e.g. 500 (yields ~1500 in-scope
conditions, ~10 GB peak disk).

### Malformed JSON warnings during inventory

If the inventory script reports `⚠️ skipping malformed JSON: …` on
many files, it usually means disk filled up mid-generation. Check
`df -h ~`, free space, regenerate from Section 3.

### LLM rate limits

The runner's per-call try/except (in
[`src/phantom_codes/eval/runner.py`](src/phantom_codes/eval/runner.py))
catches transient API failures (rate limits, timeouts, schema errors)
and writes them as `error_type` / `error_msg` columns rather than
killing the matrix. The provider clients
([`src/phantom_codes/models/llm.py`](src/phantom_codes/models/llm.py))
already retry with exponential backoff before giving up. If you see
high error counts on one provider, check that provider's status page;
otherwise just rerun with the same flags — the streaming write
appends and you can re-aggregate post-hoc.

### Cost overrun

The hard cap (`--max-cost-usd`) aborts the run before exceeding the
threshold. Soft warnings fire at 5/10/25/50/75/90% of cap so you can
react if the burn rate is unexpected. Defaults can be tightened by
passing a smaller `--max-cost-usd` for cautious initial runs.

### Missing / unexpected SNOMED codes

If `inventory_synthea_snomed.py --compare-with` flags new SNOMED
codes that aren't in the curated map, see the curation workflow in
[`data/synthea/README.md`](data/synthea/README.md). The map is
hand-curated against the SNOMED browser and ICD-10-CM tabular list;
adding new entries is a ~30-minute manual process per code.

## Citation

If you use this benchmark in published work, please cite:

> Fung, B. K. (2026). *Phantom Codes: Hallucination in LLM-Based
> Clinical Concept Normalization*. [paper venue + DOI when published]

And the underlying dependencies:

> Walonoski, J., Kramer, M., Nichols, J., et al. (2018). Synthea: An
> approach, method, and software mechanism for generating synthetic
> patients and the synthetic electronic health care record.
> *J Am Med Inform Assoc*, 25(3), 230–238.
> doi:10.1093/jamia/ocx079

> Centers for Medicare & Medicaid Services. (2026). *ACCESS Model FHIR
> Implementation Guide v0.9.6*.
> https://dsacms.github.io/cmmi-access-model/

## License

This repository is MIT-licensed (see [LICENSE](LICENSE)). Synthea
itself is Apache 2.0. The CMS ACCESS Model ValueSets are public-domain
US government work. The curated SNOMED→ICD-10-CM map is original work
under MIT — both SNOMED concept IDs and ICD-10-CM codes themselves are
public information; the lookup table is not a UMLS redistribution.

For MIMIC-derived components (the trained PubMedBERT classifier and
its training data), see PhysioNet's Credentialed Health Data License.
This repository never distributes MIMIC content or weights derived
from it.
