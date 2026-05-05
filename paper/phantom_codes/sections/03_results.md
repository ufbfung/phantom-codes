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

Two patterns dominate (per-cell N = 125; full table in
Supplementary §S2.1). First, **Anthropic constrained and RAG modes
achieve zero fabrication** across all sizes and all four
degradation modes; GPT-4o-mini and GPT-5.5 constrained / RAG match.
Second, **fabrication concentrates in zero-shot prompting and
varies by model tier**: flagship (Opus 4.7, GPT-5.5, Gemini 2.5
Pro) ranges 0–5.6% across all cells (max from Opus zero-shot D3);
sub-flagship (Sonnet 4.6, Haiku 4.5, GPT-4o-mini, Gemini Flash
variants) ranges 0–12.8% (max from GPT-4o-mini zero-shot D2). A
pooled median understates the spread (D1/D2 expose the source code;
constrained / RAG cells are 0% by design). Gemini 2.5 Pro's 0% is
from abstention (95.2% no\_prediction at D4 zero-shot), not safe
answering.

## Failure-mode breakdown: hallucination and abstention

The two failure modes the literature historically conflates —
fabrication of non-existent codes (hallucination) and abstention
(no\_prediction) — surface different patterns. Table 1 reports
both rates per cell as hallucination % / no\_prediction %; per-cell
N = 125, Wilson 95% CIs are ±3pp at 0%, ±5pp at 5%, ±8–9pp at 50%
(full CIs in §S2.2–S2.3).

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
with nonzero zero-shot hallucination. Gemini 2.5 Pro zero-shot
exhibits a near-complete abstention pattern (95.2% no\_prediction
at D4) with 0% fabrication — failure mode is abstention, not
fabrication; constrained and RAG cut but do not eliminate it.
Whether this reflects model behavior or a wrapper-level
interaction with the reasoning-token budget is unresolved (see
§Discussion limitations). Gemini Flash zero-shot rows show roughly
equal hallucination and no\_prediction rates because most rows are
*both* — empty arrays counted as no\_prediction, and when a code
is returned it is frequently fabricated. Full 27-LLM tables in
Supplementary §S2.2–S2.3.

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

Cost-per-correct (\$ per exact-match outcome) collapses per-call
price and accuracy into one deployment-ready number; the Top-1 and
Halluc columns surface all three deployment dimensions in one
row. Sorted by \$/correct ascending; n=500 per row.

**Table 3.** Cost per correct prediction. *constr.* = constrained.

| Model                          | Mode     | Top-1 | Halluc | Total | $/correct |
|:-------------------------------|:---------|------:|-------:|------:|----------:|
| Gemini 2.5 Flash               | rag      | 84.6% |   0.0% | $0.03 |   $0.0001 |
| GPT-4o-mini                    | zeroshot | 85.2% |   7.6% | $0.07 |   $0.0002 |
| GPT-4o-mini                    | rag      | 90.4% |   0.0% | $0.09 |   $0.0002 |
| Gemini 2.5 Flash               | constr.  | 88.4% |   0.0% | $0.09 |   $0.0002 |
| **GPT-4o-mini**                | **constr.** | **94.2%** | **0.0%** | **$0.14** | **$0.0003** |
| Gemini 3 Flash Preview         | constr.  | 89.6% |   0.0% | $0.37 |   $0.0008 |
| Claude Haiku 4.5               | rag      | 90.6% |   0.0% | $1.22 |   $0.0027 |
| Claude Haiku 4.5               | constr.  | 96.8% |   0.0% | $2.14 |   $0.0044 |
| Claude Sonnet 4.6              | constr.  | 94.6% |   0.0% | $3.26 |   $0.0069 |
| GPT-5.5                        | constr.  | 93.6% |   0.0% | $3.63 |   $0.0078 |
| Gemini 2.5 Pro                 | constr.  | 46.2% |   0.0% | $1.57 |   $0.0068 |
| Claude Opus 4.7                | constr.  | 94.8% |   0.0% | $6.31 |   $0.0133 |

Read top-down: cheaper rows are all 4–19pp lower on top-1 or carry
nonzero hallucination. **GPT-4o-mini constrained leads on
deployment-relevance**: 94.2% top-1, 0% hallucination, \$0.0003 per
correct — the cheapest config at ≥94% top-1. Claude Haiku 4.5
constrained adds 2.6pp accuracy at ~14× cost; Claude Opus 4.7
constrained's 0.6pp accuracy edge does not justify its 44× cost —
the largest frontier model is not the deployment-leader. Gemini
2.5 Pro's elevated \$/correct reflects abstention behavior.

## Outcome distribution under D4 abbreviation stress

D4\_abbreviated is the strongest stress test: explicit ICD codes
and canonical display strings are stripped, and clinical entities
are replaced with jargon ("T2DM", "HTN", "CKD-3"). String-matching
baselines collapse (fuzzy and TF-IDF both drop ~30pp from D3 to
D4); remaining top-1 accuracy must come from semantic mapping.
Anthropic constrained mode is robust under D4 (Haiku 93.6%,
Sonnet 90.4%, Opus 86.4% top-1 vs.\ 100% at D1); failure mode is
exclusively `category_match`. Frontier zero-shot LLMs show a
5–15pp top-1 drop D1→D4 but maintain low hallucination. Gemini
2.5 Pro abstention worsens monotonically D1→D4 in all three
prompting modes.
