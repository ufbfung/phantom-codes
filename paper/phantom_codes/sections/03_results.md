# Results

## Cohort

The Synthea evaluation cohort comprises 125 unique synthetic FHIR
Conditions [@Walonoski2018], materialized into all four degradation
modes for 500 EvalRecord items. Twelve unique ICD-10-CM codes
appear: obesity (E66.9; 24%), prediabetes (R73.03; 20%), type 2
diabetes variants (E11.x; 31% combined), essential hypertension
(I10; 14%), and lipid disorders (E78.x; 10%). The matrix evaluates
29 model configurations, generating 44,657 prediction rows.

## Outcome distribution

Two patterns dominate the per-(model, mode) outcome distribution
(per-cell N = 125; full table in Supplementary §S2.1). First,
Anthropic constrained mode achieves zero fabrication across all
three sizes (Haiku 4.5, Sonnet 4.6, Opus 4.7) and all four
degradation modes; GPT-4o-mini and GPT-5.5 constrained match.
Second, fabrication is rare across the 27 LLM configurations:
median D4\_abbreviated hallucination = 0%; 90th-percentile ≈ 39%,
high tail driven exclusively by two Gemini Flash configurations
(2.5 Flash zero-shot 43.2%; 3 Flash Preview zero-shot 39.2%). The
frontier-Anthropic + GPT-5.5 grouping shows ≤3.2% D4 hallucination
in any tested configuration.

## Failure-mode breakdown: hallucination and abstention

The two failure modes the literature historically conflates —
fabrication of non-existent codes (hallucination) and
empty-prediction (no\_prediction) — surface different patterns.
Table 1 reports both rates per (model, mode) cell as
hallucination % / no\_prediction %, with point estimates from per-cell
N = 125. Wilson 95% confidence intervals at these point estimates
are approximately ±3pp at 0%, ±5pp at 5%, and ±8–9pp at 50%; full
CIs are in Supplementary §S2.2–S2.3.

**Table 1.** Failure-mode breakdown by model and degradation mode.
Each cell reports hallucination % / no\_prediction %.

| Model (mode) | D1\_full | D2\_no\_code | D3\_text\_only | D4\_abbreviated |
|---|---|---|---|---|
| claude-haiku-4-5 (constrained) | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| claude-haiku-4-5 (zeroshot) | 0.0 / 0.0 | 3.2 / 0.0 | 4.8 / 0.0 | 3.2 / 0.0 |
| claude-sonnet-4-6 (constrained) | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| claude-sonnet-4-6 (zeroshot) | 0.0 / 0.0 | 0.8 / 0.0 | 6.4 / 0.0 | 3.2 / 0.0 |
| claude-opus-4-7 (constrained) | 0.0 / 0.0 | 0.0 / 0.0 | 0.8 / 0.8 | 0.0 / 0.0 |
| claude-opus-4-7 (zeroshot) | 0.8 / 0.0 | 0.0 / 0.0 | 5.6 / 0.0 | 1.6 / 1.6 |
| gpt-5.5 (constrained) | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| gpt-5.5 (zeroshot) | 0.8 / 0.8 | 0.0 / 0.0 | 0.8 / 0.8 | 0.0 / 0.0 |
| gpt-4o-mini (constrained) | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| gpt-4o-mini (zeroshot) | 0.8 / 0.0 | 12.8 / 0.0 | 9.6 / 0.0 | 7.2 / 0.0 |
| gemini-2.5-pro (zeroshot) | 0.0 / 77.6 | 0.0 / 85.6 | 0.0 / 94.4 | 0.0 / 95.2 |
| gemini-2.5-pro (constrained) | 0.0 / 54.4 | 0.0 / 32.0 | 0.0 / 67.2 | 0.0 / 60.8 |
| gemini-2.5-flash (zeroshot) | 0.0 / 0.0 | 11.2 / 11.2 | 38.4 / 37.6 | 43.2 / 42.4 |
| gemini-3-flash-preview (zeroshot) | 22.4 / 22.4 | 28.0 / 28.0 | 41.6 / 41.6 | 39.2 / 39.2 |

Within-model paired comparison (zero-shot → constrained, McNemar
tests on discordant pairs) confirms the constrained-mode reduction
is statistically significant for every Anthropic and OpenAI model
that shows nonzero zero-shot hallucination at any mode. Gemini 2.5
Pro under zero-shot exhibits a near-complete abstention pattern
(95.2% no\_prediction at D4) with 0% fabrication — the model's
failure mode under our wrapper settings is abstention, not
fabrication. Constrained and RAG prompting cut the Gemini 2.5 Pro
abstention rate substantially but do not eliminate it. Whether this
reflects model behavior or a wrapper-level interaction with the
reasoning-token budget is unresolved (see §Discussion limitations);
the n=125 cohort is sufficient to surface the pattern but not
attribute the cause. The Gemini 2.5 Flash and Gemini 3 Flash Preview
zero-shot rows show roughly equal hallucination and no\_prediction
rates because most of those rows are *both* — the model returns an
empty array of predictions which is counted as no\_prediction, and
when it does return a code it frequently fabricates one. The full
27-LLM tables are in Supplementary §S2.2–S2.3.

## Top-1 vs top-5 exact-match lift

Many configurations have the right answer in their top-5 candidates
even when their top-1 pick is wrong; lift quantifies the gap:

**Table 2.** Top-1 vs top-5 exact-match lift across selected configurations.

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

Two deployment-relevant signals: (i) LLMs in *constrained* mode
saturate at top-1 (lift ≤6pp) — there is little additional value in
human-in-the-loop top-5 review for those configurations; (ii)
trained-classifier baselines and sub-frontier LLMs in zero-shot mode
show large lifts (20–38pp), suggesting workflows that surface top-5
candidates to a human reviewer would substantially improve net
accuracy. Gemini 2.5 Pro's zero lift is consistent with the
abstention-dominated failure mode (no candidate to be lifted into
top-5 when the model returns nothing).

## Cost per correct prediction

Cost-per-correct-prediction (\$ per exact-match outcome) collapses
per-call price and per-call accuracy into a single deployment-ready
number:

**Table 3.** Cost per correct prediction (selected configurations).

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
constrained at \$0.0003 per exact match (94.2% top-1, 0%
hallucination). A 100×–250× cost spread separates the cheapest
(Gemini 2.5 Flash at \$0.0001) from the most expensive (Claude Opus
4.7 constrained at \$0.0133) for comparable top-1 accuracy at this
cohort scale. The Gemini 2.5 Pro row's cost-per-correct distortion
reflects its abstention behavior: 231 exact matches is roughly half
of comparably-priced configurations (Sonnet constrained: 473), so
per-correct cost is inflated relative to per-call cost. The full
per-bucket cost decomposition is in Supplementary §S2.5.

## Outcome distribution under D4 abbreviation stress

D4\_abbreviated is the strongest stress test: explicit ICD codes
are removed, canonical display strings are stripped, and clinical
entities are replaced with jargon ("T2DM", "HTN", "CKD-3").
String-matching baselines collapse (fuzzy and TF-IDF both drop
~30pp from D3 to D4 exact-match rate); any remaining top-1 accuracy
must come from semantic mapping rather than lexical overlap.
Anthropic constrained mode is robust under D4 (Haiku 93.6%, Sonnet
90.4%, Opus 86.4% top-1 vs.\ 100% at D1); the drop is small and the
failure mode is exclusively `category_match` rather than fabrication
or abstention. Frontier zero-shot LLMs show a 5–15pp top-1 drop
D1→D4 but maintain low hallucination; GPT-5.5 zero-shot actually
*improves* on D4 (84.8%) relative to D3 (71.2%) by avoiding the
out-of-domain Z68.x BMI codes it produces under D3. Gemini 2.5 Pro
abstention worsens monotonically D1→D4 in all three prompting modes.
