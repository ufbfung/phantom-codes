# S2 — Extended results

This supplement extends §3 of the main manuscript with the full
per-(model, mode) outcome distribution across all 29 configurations,
per-disease-group breakdowns, and the complete Wilson 95% CI tables
for hallucination, no\_prediction, and top-1 accuracy.

> **Status:** Skeleton — to be filled by reproducing the full
> tables from `results/summary/n125_run_v2/headline.md` (or by
> regenerating with `phantom-codes report --csv ...`). The headline-
> relevant subset of these tables is in main §3; the full tables
> reproduce all 29 (model, mode) cells for reviewer auditability.

## S2.1 — Full headline outcome distribution (all 29 configs × 4 modes)

[TODO: paste the full headline_table from
`results/summary/n125_run_v2/headline.md` — 116 rows (29 configs ×
4 modes) covering all 6 outcome buckets per cell.]

## S2.2 — Full hallucination-rate table with Wilson 95% CIs

[TODO: paste full hallucination_table — 116 rows.]

## S2.3 — Full no\_prediction-rate table with Wilson 95% CIs

[TODO: paste full no_prediction_table — 116 rows.]

## S2.4 — Full top-1 vs top-5 lift table (all 29 configs)

[TODO: paste full top_k_lift_table — 29 rows.]

## S2.5 — Full per-bucket cost decomposition

[TODO: paste full per_bucket_cost_table from
`results/summary/n125_run_v2/per_bucket_cost.csv` — for each
(config, outcome) cell, total \$ spent on that bucket and per-call
\$ within the bucket.]

## S2.6 — Per-disease-group breakdown (CKM vs. eCKM)

[TODO: regenerate the headline tables filtered by gt\_group to
report performance separately on CKM-track conditions (diabetes,
ASCVD, CKD-3) vs. eCKM-track conditions (hypertension,
dyslipidemia, prediabetes, obesity). Per-group N is ~60-65
records; CIs are correspondingly wider.]

## S2.7 — Per-vocabulary-position breakdown

[TODO: bucket the 12 unique cohort ICD codes by frequency tier and
report per-tier accuracy. At n=125 with 12 codes, per-tier N ranges
from ~5 (rarest codes) to ~30 (most-frequent code), so this is
exploratory rather than statistically authoritative.]

## S2.8 — Hallucinated-code listing

[TODO: enumerate every fabricated (non-existent) ICD-10-CM code
emitted by any model, with frequency and the model + mode that
produced it. From the n=125 data this should be a short list:
mostly Gemini Flash zeroshot fabrications, with rare Anthropic /
GPT-4o-mini fabrications under specific modes. Useful as a
qualitative supplement showing what fabrications look like in
practice.]
