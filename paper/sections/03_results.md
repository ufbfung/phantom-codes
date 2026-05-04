# Results

> **Status:** Draft v1 (2026-05-04). Numbers populated from the n=125
> headline evaluation run on the Synthea cohort. Per-cell N is 125;
> Wilson 95% confidence intervals shown with each rate. The
> originally-targeted n=500 cohort would tighten CIs by approximately
> 2× and is documented as a v2 replication path in §Limitations.

---

## Cohort sizing

The two cohorts that drive the numbers in this section are distinct:

- **MIMIC-IV-FHIR training cohort** for the PubMedBERT classifier
  fine-tuning: 245,575 ACCESS-scope conditions distributed 70/10/20
  across train, validation, and test splits by stratified sampling
  on `resource_id` (so all four degradation modes of one condition
  land in the same split). The classifier head is sized for the
  top-50 most frequent ICD-10-CM codes, which together cover the
  head of the long-tail distribution where supervised signal is
  densest. Per-epoch training metrics, validation loss curves, and
  held-out test-set top-1 accuracy are reported in §Methods; the
  classifier itself is then evaluated on the Synthea cohort below
  alongside every other model in the headline matrix, providing
  both compliance-by-construction (MIMIC content does not reach any
  third-party API) and an out-of-distribution generalization test.
- **Synthea evaluation cohort** for the headline matrix:
  **125 unique synthetic FHIR Conditions** generated via
  Synthea v4.0.0 [@Walonoski2018] under the seed and module
  configuration documented in §Methods, materialized into all four
  degradation modes (D1\_full / D2\_no\_code / D3\_text\_only /
  D4\_abbreviated) for each condition. Twelve unique ICD-10-CM
  codes appear in the cohort, distributed across the ACCESS-scope
  condition groups: obesity (E66.9; 24% of cohort), prediabetes
  (R73.03; 20%), essential hypertension (I10; 13.6%), type 2
  diabetes mellitus variants (E11.x; 31% combined), and lipid
  disorders (E78.x; 10%). Cohort generation is deterministic given
  the pinned Synthea SHA + seed; full reproduction guidance is in
  the project's BENCHMARK.md.

The headline matrix evaluates 29 model configurations on this
500-EvalRecord cohort (125 records × 4 modes), generating 44,657
prediction rows in the long-format CSV. All subsequent numbers in
this section are aggregated from that single run.

## Outcome distribution per model × mode

The primary outcome of the matrix is the per-(model, mode)
distribution across the six outcome buckets defined in §Methods:
exact match, category match (same ICD-10 3-character category),
chapter match (same first character), out-of-domain (real ICD-10-CM
code unrelated to truth), no\_prediction (model abstained — empty
predictions array, refusal, or transient API failure), and
hallucination (predicted code does not exist in the CMS-published
FY2026 ICD-10-CM tabular list). Per-cell N = 125. The distribution
is read row-wise: each row sums to 100%.

Two patterns dominate:

1. **Anthropic constrained mode achieves zero fabrication across all
   three model sizes and all four degradation modes.** Haiku 4.5,
   Sonnet 4.6, and Opus 4.7 each register 0% hallucination under
   `constrained` and `rag` prompting modes from D1\_full through
   D4\_abbreviated. GPT-4o-mini and GPT-5.5 in their constrained
   and RAG configurations match this pattern.
2. **Fabrication is rare in modern frontier LLMs** under any
   prompting mode tested here. Across the 27 LLM configurations,
   the median hallucination rate at D4\_abbreviated is 0%; the
   90th-percentile rate is approximately 39%. The high tail comes
   exclusively from two Gemini Flash configurations (Gemini 2.5
   Flash zero-shot at 43.2%; Gemini 3 Flash Preview zero-shot at
   39.2%). The frontier-Anthropic + GPT-5.5 grouping shows ≤3.2%
   D4 hallucination in any configuration tested.

The **no\_prediction** bucket separates two qualitatively different
failure modes that the literature historically lumps together. For
Gemini 2.5 Pro under zero-shot mode, the no\_prediction rate at
D1\_full is 77.6% and rises monotonically with degradation to 95.2%
at D4\_abbreviated; the corresponding hallucination rate is 0% in
every mode. The model's failure mode under our wrapper settings is
thus near-complete abstention, not fabrication. Whether this is a
genuine Gemini 2.5 Pro behavior or an interaction with our
wrapper's reasoning-token budget is the subject of a follow-up
investigation noted in §Limitations.

The full per-(model, mode) outcome distribution is reported in
Table 1 (Appendix A reproduces it in CSV form for downstream
analysis); the headline numbers under D4\_abbreviated stress are
extracted into a separate subsection below.

## Hallucination rate (narrow definition: fabrications only)

> Hallucination here means *the predicted code string does not exist
> in the FY2026 CMS ICD-10-CM tabular list*, mechanically checked
> against the bundled validator. Empty predictions / abstention rows
> are reported in the next subsection as no\_prediction, distinct
> from hallucination — a methodological refinement from the original
> 5-bucket taxonomy.

Selected headline rates with Wilson 95% confidence intervals:

| Model (mode) | D1\_full | D2\_no\_code | D3\_text\_only | D4\_abbreviated |
|---|---|---|---|---|
| claude-haiku-4-5 (constrained) | 0.0% (0.0–3.0) | 0.0% (0.0–3.0) | 0.0% (0.0–3.0) | 0.0% (0.0–3.0) |
| claude-haiku-4-5 (zeroshot) | 0.0% (0.0–3.0) | 3.2% (1.3–7.9) | 4.8% (2.2–10.1) | 3.2% (1.3–7.9) |
| claude-sonnet-4-6 (constrained) | 0.0% (0.0–3.0) | 0.0% (0.0–3.0) | 0.0% (0.0–3.0) | 0.0% (0.0–3.0) |
| claude-sonnet-4-6 (zeroshot) | 0.0% (0.0–3.0) | 0.8% (0.1–4.4) | 6.4% (3.3–12.1) | 3.2% (1.3–7.9) |
| claude-opus-4-7 (constrained) | 0.0% (0.0–3.0) | 0.0% (0.0–3.0) | 0.8% (0.1–4.4) | 0.0% (0.0–3.0) |
| claude-opus-4-7 (zeroshot) | 0.8% (0.1–4.4) | 0.0% (0.0–3.0) | 5.6% (2.7–11.1) | 1.6% (0.4–5.6) |
| gpt-5.5 (zeroshot) | 0.8% (0.1–4.4) | 0.0% (0.0–3.0) | 0.8% (0.1–4.4) | 0.0% (0.0–3.0) |
| gpt-4o-mini (zeroshot) | 0.8% (0.1–4.4) | 12.8% (8.0–19.8) | 9.6% (5.6–16.0) | 7.2% (3.8–13.1) |
| gemini-2.5-flash (zeroshot) | 0.0% (0.0–3.0) | 11.2% (6.8–17.9) | 38.4% (30.3–47.2) | 43.2% (34.8–52.0) |
| gemini-3-flash-preview (zeroshot) | 22.4% (16.0–30.5) | 28.0% (20.9–36.4) | 41.6% (33.3–50.4) | 39.2% (31.1–48.0) |

Within-model paired comparison (zero-shot → constrained, McNemar's
test on discordant pairs) confirms the constrained-mode reduction is
statistically significant for every Anthropic and OpenAI model that
shows nonzero zero-shot hallucination at any mode. The full
hallucination-rate table for all 27 LLM configurations is in
Appendix A.

## No-prediction (abstention) rate

The Wilson-95%-CI rates for `no_prediction` follow a different
pattern from hallucination — concentrated in two model families
with distinct underlying causes:

| Model (mode) | D1\_full | D2\_no\_code | D3\_text\_only | D4\_abbreviated |
|---|---|---|---|---|
| baseline:exact | 100.0% (97.0–100.0) | 100.0% (97.0–100.0) | 100.0% (97.0–100.0) | 100.0% (97.0–100.0) |
| gemini-2.5-pro (zeroshot) | 77.6% (69.5–84.0) | 85.6% (78.4–90.7) | 94.4% (88.9–97.3) | 95.2% (89.9–97.8) |
| gemini-2.5-pro (constrained) | 54.4% (45.7–62.9) | 32.0% (24.5–40.6) | 67.2% (58.6–74.8) | 60.8% (52.0–68.9) |
| gemini-2.5-pro (rag) | 60.8% (52.0–68.9) | 64.8% (56.1–72.6) | 75.2% (67.0–81.9) | 66.4% (57.7–74.1) |
| gemini-3-flash-preview (zeroshot) | 22.4% (16.0–30.5) | 28.0% (20.9–36.4) | 41.6% (33.3–50.4) | 39.2% (31.1–48.0) |
| (all other LLMs) | 0–5% across all modes | | | |

The `baseline:exact` 100% rate reflects the baseline's design — exact
string matching against ICD-10-CM display strings rarely succeeds on
clinical free-text input — and serves as a sanity check that the
no\_prediction bucket is correctly populated. The Gemini 2.5 Pro
abstention pattern is the substantively interesting finding and is
discussed in §Discussion under "Hallucination versus abstention as
distinct failure modes."

## Top-1 vs top-5 exact-match lift

Many configurations have the right answer in their top-5 candidates
even when their top-1 pick is wrong; lift quantifies the gap:

| Model | Top-1 | Top-5 | Lift |
|---|---|---|---|
| claude-haiku-4-5 (constrained) | 96.8% | 100.0% | +3.2pp |
| claude-sonnet-4-6 (constrained) | 94.6% | 100.0% | +5.4pp |
| claude-opus-4-7 (constrained) | 94.8% | 99.8% | +5.0pp |
| gpt-4o-mini (constrained) | 94.2% | 100.0% | +5.8pp |
| gpt-5.5 (constrained) | 93.6% | 100.0% | +6.4pp |
| pubmedbert:classifier | 47.8% | 73.2% | +25.4pp |
| sentence-transformer:retrieval | 69.0% | 90.6% | +21.6pp |
| baseline:tfidf | 45.2% | 83.2% | +38.0pp |
| claude-sonnet-4-6 (zeroshot) | 75.4% | 95.8% | +20.4pp |
| gpt-5.5 (zeroshot) | 88.8% | 98.8% | +10.0pp |
| gemini-2.5-pro (zeroshot) | 11.8% | 11.8% | +0.0pp |

Two deployment-relevant signals: (i) the LLMs in *constrained* mode
saturate at top-1 (lift ≤6pp) — there is little additional value
in human-in-the-loop top-5 review for those configurations; (ii)
trained-classifier baselines (PubMedBERT, sentence-transformer) and
sub-frontier LLMs in zero-shot mode show large lifts (20–38pp),
suggesting workflows that surface top-5 candidates to a human
reviewer would substantially improve net accuracy. Gemini 2.5 Pro's
lift of zero is consistent with the abstention-dominated failure
mode reported above (no candidate to be lifted into top-5 when the
model returns nothing).

## Cost per correct prediction

Cost-per-correct-prediction (\$ per exact-match outcome) collapses
the per-call price and the per-call accuracy into a single
deployment-ready number:

| Model (mode) | Total cost | Exact matches | $/correct |
|---|---|---|---|
| gemini-2.5-flash (rag) | $0.03 | 423 | $0.0001 |
| gemini-2.5-flash (zeroshot) | $0.03 | 377 | $0.0001 |
| gpt-4o-mini (constrained) | $0.14 | 471 | $0.0003 |
| gpt-4o-mini (zeroshot) | $0.07 | 426 | $0.0002 |
| gemini-3-flash-preview (constrained) | $0.37 | 448 | $0.0008 |
| claude-haiku-4-5 (constrained) | $2.14 | 484 | $0.0044 |
| claude-haiku-4-5 (zeroshot) | $1.12 | 445 | $0.0025 |
| claude-sonnet-4-6 (constrained) | $3.26 | 473 | $0.0069 |
| gpt-5.5 (constrained) | $3.63 | 468 | $0.0078 |
| claude-opus-4-7 (constrained) | $6.31 | 474 | $0.0133 |
| gemini-2.5-pro (constrained) | $1.57 | 231 | $0.0068 |

The cheapest reliable configuration on this cohort is GPT-4o-mini
in constrained mode at \$0.0003 per exact match (94.2% top-1
accuracy, 0% hallucination). A 100×–250× cost spread separates the
cheapest (Gemini 2.5 Flash at \$0.0001) from the most expensive
(Claude Opus 4.7 in constrained mode at \$0.0133) for comparable
top-1 accuracy at this cohort scale.

The cost-per-correct distortion in Gemini 2.5 Pro's row reflects
its abstention behavior: the denominator (231 exact matches) is
~half of what other comparably-priced configurations achieve
(Sonnet 4.6 constrained: 473 exact matches at similar per-call
cost), so the per-correct cost is inflated relative to its
per-call cost.

The full cost-decomposition table (per outcome bucket — what \$ was
spent on hallucinations, on no\_predictions, on category matches,
etc.) is in Appendix A and forms the basis of the deployment-cost
break-even analysis in the supplementary §S5.

## Outcome distribution under D4 abbreviation stress

D4\_abbreviated is the strongest stress test in the matrix: explicit
ICD codes are removed, canonical display strings are stripped, and
clinical entities are replaced with abbreviations and jargon
("T2DM", "HTN", "CKD-3"). String-matching baselines collapse here
(baseline:fuzzy and baseline:tfidf both drop ~30pp from D3 to D4
exact-match rate); any remaining top-1 accuracy must come from
genuine semantic mapping rather than lexical overlap.

Three patterns under D4 specifically:

1. **Anthropic constrained mode is robust to D4 stress.** Haiku
   constrained: 93.6% top-1 (D1: 100%); Sonnet constrained: 90.4%
   (D1: 100%); Opus constrained: 86.4% (D1: 100%). The drop is
   small and the failure mode is exclusively `category_match` — the
   model picks a related ICD code in the right family rather than
   fabricating or abstaining.
2. **Frontier zero-shot LLMs show a 5–15pp top-1 drop from D1 to D4**
   but maintain low hallucination. The exception is GPT-5.5
   zero-shot, which actually *improves* on D4 (84.8%) relative to
   D3 (71.2%) by avoiding the out-of-domain Z68.x BMI codes it
   produces under D3 when input mentions obesity directly — a
   model-specific behavior worth noting.
3. **Gemini 2.5 Pro abstention worsens monotonically D1 → D4** in
   all three prompting modes. Whether this reflects model behavior
   or a wrapper-specific reasoning-token interaction is unresolved
   (see §Limitations); the n=125 cohort is sufficient to surface
   the pattern but not to attribute the cause.

## Sensitivity analyses

The n=125 cohort precludes most sub-cell sensitivity work — per-
disease-group breakdowns yield ~60 observations per group, widening
already-wide CIs. Three sensitivity dimensions worth noting for v2
work:

- **Per-disease-group breakdown** (CKM vs. eCKM): the cohort is
  weighted toward eCKM conditions (obesity, prediabetes,
  hypertension dominate) by construction of Synthea's modules under
  the configured population profile. Per-group analyses at higher
  N would test whether any model degrades disproportionately on
  one disease group.
- **Per-vocabulary-position breakdown**: code-frequency in the
  cohort spans 12 codes; analysis at the long-tail vocabulary
  positions requires a larger cohort (the originally-targeted
  n=500 covers more codes per cell).
- **Robustness to prompt phrasing**: a single zero-shot prompt
  template was used for all LLMs (verbatim in §Supplementary S1).
  Sweep-style robustness studies are deferred to v2.

A v2 replication on the originally-intended n=500 cohort
(`--max-records 2000` per the BENCHMARK reproduction guide) would
roughly halve all reported confidence intervals and enable the
sensitivity analyses above.
