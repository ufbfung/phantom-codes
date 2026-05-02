# Results

> **Status:** Scaffold v0 (2026-05-02). Section structure is settled;
> numbers TBD until headline run completes. Each subsection notes the
> specific source data needed to populate it. Order follows
> NEJM AI / IMRaD convention: cohort first → trained-model performance
> → frontier-LLM evaluation matrix → outcome distribution by mode →
> deployment economics.

---

## Cohort and label space

Final cohort sizes derived from the prepare pipeline against MIMIC-IV-FHIR
v2.1 with ACCESS-scope filtering:

| Split | Resources | Rows (4 modes/resource) | Unique codes |
|-------|-----------|-------------------------|--------------|
| Train | 171,888   | 687,552                 | 178          |
| Val   | 24,568    | 98,272                  | 140          |
| Test  | 49,119    | 196,476                 | 153          |
| Total | 245,575   | 982,300                 | —            |

The classification head was sized for the top-50 most frequent codes,
which collectively account for [TBD: %]\% of train rows. Code-frequency
distribution follows the long-tail pattern characteristic of clinical
diagnoses, with the head dominated by hypertension (I10), type 2
diabetes (E11.x), and dyslipidemia (E78.x).

> **Source for fill-in:** `data/derived/conditions/{train,val,test}.parquet`
> aggregate counts (already shared in §Methods); add the top-50
> coverage percentage from `phantom-codes` aggregate output.

## Fine-tuned PubMedBERT classifier performance

> **Status:** Awaiting headline training run (in progress at submission
> draft time; ~15 hours wall-clock on M1 hardware as documented in
> §Methods).

Once the run completes, this subsection reports per-epoch training
and validation loss, val-loss-best-checkpoint epoch, and held-out
test-set accuracy. The headline metric is per-mode top-1 accuracy
on the test split:

| Mode | Top-1 accuracy | Top-5 accuracy | Mean confidence |
|---|---|---|---|
| D1\_full | [TBD] | [TBD] | [TBD] |
| D2\_no\_code | [TBD] | [TBD] | [TBD] |
| D3\_text\_only | [TBD] | [TBD] | [TBD] |
| D4\_abbreviated | [TBD] | [TBD] | [TBD] |

The expected ordering (D1 ≥ D3 ≥ D2 ≥ D4) follows from the
information-theoretic argument that explicit canonical text is more
informative than partially-degraded FHIR JSON, and that abbreviation
substitution removes the lexical signal string-matching baselines
depend on.

> **Source for fill-in:** `models/metrics/train_*.json` (aggregate
> per-epoch metrics, safe to share) and a separate test-evaluation
> pass against the trained checkpoint on the held-out test split.
> Per-record predictions remain local (compliance) — only aggregated
> per-mode accuracy goes into the manuscript.

## Frontier-LLM evaluation matrix on Synthea

> **Status:** Awaiting Synthea cohort generation + evaluation pass.

The headline experiment runs every model in the matrix (frontier
LLMs across three prompting modes; trained PubMedBERT classifier;
sentence-transformer retrieval baseline; three constrained-only
string-matching baselines) against the same Synthea cohort. Synthea
data is freely redistributable; results in this subsection are
fully reproducible by any researcher with Synthea installed.

The primary outcome is the per-model distribution across the five
outcome buckets (Table~[TBD]), reported separately for each
degradation mode (D1\_full / D2\_no\_code / D3\_text\_only /
D4\_abbreviated) so the reader can see how degradation interacts
with each model's behavior.

### Hallucination rate (primary outcome)

[TBD: hallucination-rate table by model × mode. Expected pattern:
near-zero hallucination for trained classifier and constrained-only
baselines (mechanically constrained to vocabulary); nontrivial
hallucination for zero-shot LLMs, especially under D4_abbreviated;
substantial reduction in constrained-mode and RAG-mode within the
same model family.]

### Top-1 accuracy (secondary outcome)

[TBD: top-1 accuracy table by model × mode.]

### Cost per correct prediction (secondary outcome)

Drawing on the cost framework in §Cost economics, this subsection
reports cost-per-correct-prediction (\$ per exact-match outcome) for
each LLM configuration. [TBD: numbers from the headline run; format
follows the table prepared in §Cost economics.]

> **Source for fill-in:** `results/raw/headline_*.csv` (aggregated
> per-prediction CSVs from the eval runner; safe to share since they're
> Synthea-derived) and the corresponding per-model summary stats.

## Outcome distribution under abbreviation stress (D4)

> **Status:** Headline finding; placement matches NEJM AI convention
> of leading the most-novel result early in §Results.

This subsection reports the full 5-way outcome distribution under
D4\_abbreviated specifically, since this is the condition under which
string-matching baselines collapse and any remaining LLM advantage
must come from genuine semantic retrieval. The distribution shape
across models — particularly how mass shifts between
\emph{out\_of\_domain} and \emph{hallucination} — is the empirical
contribution this paper expects to surface.

[TBD: stacked-bar figure or table showing 5-way outcome distribution
by model (rows) and bucket (columns) under D4\_abbreviated specifically.]

[TBD: 1-2 paragraphs of prose describing the headline pattern: which
models hallucinate at what rates, and how that distribution shifts
between zero-shot and constrained prompting within the same model.]

## Sensitivity analyses

> **Status:** Optional; include if results suggest sensitivity to
> these dimensions or if reviewers ask.

- [TBD] Per-disease-group breakdown (CKM vs.\ eCKM): does any model
  hallucinate more on one group than the other?
- [TBD] Per-vocabulary-position breakdown: does hallucination rate
  scale with code frequency (rare codes hallucinated more often)?
- [TBD] Robustness to prompt phrasing: do small variations in the
  zero-shot prompt change the hallucination rate by more than one
  standard error? (Brief check; full robustness study is v2 work.)
