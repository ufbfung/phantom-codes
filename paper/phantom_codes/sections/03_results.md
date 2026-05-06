# RESULTS

## Cohort

The Synthea evaluation cohort comprises 125 unique synthetic FHIR
Conditions [@Walonoski2018], materialized into all four degradation
modes for 500 EvalRecord items. Twelve unique ICD-10-CM codes
appear: obesity (E66.9; 24%), prediabetes (R73.03; 20%), type 2
diabetes variants (E11.x; 31% combined), essential hypertension
(I10; 14%), and lipid disorders (E78.x; 10%). The matrix evaluates
28 model configurations.

## Outcome distribution

Two patterns dominate (per-cell N = 125; full table in
Supplementary §S2.1). Anthropic constrained and RAG modes achieve
zero fabrication across all sizes and all four degradation modes,
and GPT-4o-mini and GPT-5.5 in constrained or RAG mode match.
Fabrication concentrates instead in zero-shot prompting and varies
by model tier. Flagship models (Opus 4.7, GPT-5.5, Gemini 2.5 Pro)
range from 0% to 5.6% across all cells, with the maximum from Opus
zero-shot D3. Sub-flagship models (Sonnet 4.6, Haiku 4.5,
GPT-4o-mini, Gemini Flash variants) range from 0% to 12.8%, with
the maximum from GPT-4o-mini zero-shot D2. A pooled median
understates the spread, because D1 and D2 inputs expose the source
code and constrained or RAG cells are 0% by design. Gemini 2.5
Pro's 0% fabrication rate comes from abstention (95.2%
no\_prediction at D4 zero-shot), not safe answering.

## Failure-mode breakdown: hallucination and abstention

The literature historically conflates two failure modes:
fabrication of non-existent codes (hallucination) and abstention
(no\_prediction). They surface different patterns (Figure 1).
Table 1 reports both rates per cell as hallucination % /
no\_prediction %. Per-cell N = 125; Wilson 95% CIs are ±3pp at 0%,
±5pp at 5%, and ±8 to ±9pp at 50% (full CIs in §S2.2 to §S2.3).

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
| gemini-2.5-flash (zeroshot) | 0.0 / 0.0 | 0.0 / 11.2 | 0.8 / 37.6 | 0.8 / 42.4 |
| gemini-3-flash-preview (zeroshot) | 0.0 / 22.4 | 0.0 / 28.0 | 0.0 / 41.6 | 0.0 / 39.2 |

: Failure-mode breakdown by model and degradation mode. Each cell reports hallucination % / no\_prediction %.

\begin{figure}[tbp]
\centering
\includegraphics[width=0.75\textwidth,keepaspectratio]{../figures/figure1_hallucination_heatmap.pdf}
\caption{Hallucination rate (\%) by model, prompting mode, and degradation mode. Each row is one (LLM, prompting mode) configuration; columns are the four degradation modes. Cell color encodes the percentage of D-mode predictions that were fabricated codes (predictions that do not exist in the FY2026 CMS-published ICD-10-CM tabular list). Constrained and RAG rows are uniformly zero across all Anthropic and OpenAI models. Zero-shot fabrication concentrates in sub-flagship models, peaking at 12.8\% for GPT-4o-mini under D2.}
\par\noindent\textit{Alt text: Heatmap with rows for 24 LLM and prompting-mode configurations grouped by provider (Anthropic, OpenAI, Google), and columns for four input degradation modes (D1 full, D2 no code, D3 text only, D4 abbreviated). Cell shading indicates hallucination percentage. All Anthropic and OpenAI constrained and RAG rows show 0\%. Zero-shot rows show measurable hallucination, peaking at 12.8\% for GPT-4o-mini at D2.}
\label{fig:halluc}
\end{figure}

Within-model paired comparison (zero-shot versus constrained,
McNemar tests on discordant pairs) confirms the constrained-mode
reduction is statistically significant for every Anthropic and
OpenAI model with nonzero zero-shot hallucination. Gemini 2.5 Pro
zero-shot exhibits a near-complete abstention pattern (95.2%
no\_prediction at D4) with 0% fabrication. The failure mode is
abstention, not fabrication, and constrained and RAG cut but do
not eliminate it. Whether this reflects model behavior or a
wrapper-level interaction with the reasoning-token budget is
unresolved (see §Discussion limitations). Gemini 2.5 Flash and
Gemini 3 Flash Preview zero-shot abstain rather than fabricate.
When these models do return a code under D2 to D4 stress, it is
almost always real (hallucination ≤0.8% across all cells); the
dominant failure is empty-prediction, which rises to 39 to 42%
under D4 abbreviation stress. Full 24-LLM tables in Supplementary
§S2.2 to §S2.3.

## Top-1 vs top-5 exact-match lift

Many configurations have the right answer in their top-5
candidates even when their top-1 pick is wrong. Lift quantifies
the gap.

| Model | Top-1 | Top-5 | Lift |
|---|---|---|---|
| claude-haiku-4-5 (constrained) | 96.8% | 100.0% | +3.2pp |
| claude-sonnet-4-6 (constrained) | 94.6% | 100.0% | +5.4pp |
| claude-opus-4-7 (constrained) | 94.8% | 99.8% | +5.0pp |
| gpt-4o-mini (constrained) | 94.2% | 100.0% | +5.8pp |
| gpt-5.5 (constrained) | 93.6% | 100.0% | +6.4pp |
| sentence-transformer:retrieval | 69.0% | 90.6% | +21.6pp |
| baseline:tfidf | 45.2% | 83.2% | +38.0pp |
| claude-sonnet-4-6 (zeroshot) | 75.4% | 95.8% | +20.4pp |
| gpt-5.5 (zeroshot) | 88.8% | 98.8% | +10.0pp |
| gemini-2.5-pro (zeroshot) | 11.8% | 11.8% | +0.0pp |

: Top-1 vs top-5 exact-match lift across selected configurations.

Two deployment-relevant signals emerge. LLMs in *constrained* mode
saturate at top-1 (lift ≤6pp), so there is little additional value
in human-in-the-loop top-5 review for those configurations.
Non-LLM baselines and sub-frontier LLMs in zero-shot mode show
much larger lifts (20 to 38pp), suggesting that workflows
surfacing top-5 candidates to a human reviewer would substantially
improve net accuracy. Gemini 2.5 Pro's zero lift is consistent
with the abstention-dominated failure mode, since there is no
candidate to be lifted into top-5 when the model returns nothing.

## Cost per correct prediction

Most prior LLM medical-coding benchmarks report accuracy in
isolation [@Soroush2024; @Almeida2025; @Singh2025]. The deployment
decision, however, hinges on the joint distribution of API cost
per call, per-call accuracy, and the rate of failure modes that
drive downstream QA cost. Selected configurations from the
headline matrix are shown below; the full 28-configuration
breakdown, per-bucket cost decomposition, break-even analysis
(LLM and QA versus a human coder), and annual cost projections at
deployment scale are reported in Supplementary §S5.

| Model × Mode | n | $ total | $ per call (mean) |
|---|---:|---:|---:|
| claude-opus-4-7:zeroshot | 500 | 6.77 | 0.01354 |
| claude-sonnet-4-6:zeroshot | 500 | 3.41 | 0.00682 |
| claude-haiku-4-5:constrained | 500 | 2.14 | 0.00428 |
| gpt-5.5:zeroshot † | 499 | 3.59 | 0.00720 |
| gpt-4o-mini:constrained | 500 | 0.14 | 0.00028 |
| gemini-2.5-pro:zeroshot ‡ | 424 | 0.41 | 0.00096 |
| gemini-2.5-flash:zeroshot | 500 | 0.03 | 0.00005 |

: Per-call API cost across selected (model, prompting-mode) configurations from the headline matrix. n = number of completed calls (out of 500 attempted). †, ‡ explained in the footnotes below.

† One of 500 attempted gpt-5.5 zero-shot calls exhausted the SDK
retry budget on a transient API error and is excluded from the cost
total. We report n=499 as-is rather than re-running, since transient
API failures are an inherent property of production LLM deployment
and re-running would obscure that signal.

‡ Gemini 2.5 Pro zero-shot completed 424 of 500 attempted calls
under our wrapper's `max_output_tokens=1024` setting. The remaining
76 returned empty responses consistent with reasoning-token budget
exhaustion (counted as `no_prediction` in §4.3, not as cost-bearing
calls). The 424 figure reflects calls that produced billable
output; cost normalization uses that denominator.

A 270× spread separates the cheapest configuration (Gemini 2.5
Flash zeroshot at \$0.00005/call) from the most expensive (Claude
Opus 4.7 zeroshot at \$0.01354/call). Per-call price varies more
dramatically across the matrix than per-call accuracy, so
cost-per-correct prediction is dominated by per-call price for any
configuration that achieves ≥80% top-1 accuracy. One caveat: the
cost numbers reported here are *automated-coding cost*, not the
broader role of clinical informaticists. Any break-even analysis
is *deployment cost guidance*, not an *automation feasibility
verdict*. Error tolerance is workflow-specific, and a hallucinated
code that flows into claims billing has different downstream cost
than the same hallucination in a research cohort definition. The
full caveats are in Supplementary §S5.

Cost-per-correct (\$ per exact-match outcome) collapses per-call
price and accuracy into one deployment-ready number; the Top-1
and Halluc columns surface all three deployment dimensions in one
row. Plotted as a Pareto frontier in Figure 2 and tabulated in
Table 4 below. Sorted by \$/correct ascending; n=500 per row.

\begin{figure}[tbp]
\centering
\includegraphics[width=0.7\textwidth,keepaspectratio]{../figures/figure2_cost_frontier.pdf}
\caption{Cost per 1,000 correct predictions (USD, log scale) versus top-1 accuracy for every (LLM, prompting mode) configuration achieving \(\geq\)75\% top-1 accuracy. Color encodes provider; marker shape encodes prompting mode. Pareto-optimal configurations are drawn with a heavy black border. GPT-4o-mini constrained and Claude Haiku 4.5 constrained sit on the deployment frontier; Claude Opus 4.7 constrained is dominated despite competitive accuracy because its per-correct cost is 30 to 40 times higher.}
\par\noindent\textit{Alt text: Scatter plot. The x-axis shows top-1 accuracy from 75 to 100 percent; the y-axis shows cost per 1,000 correct predictions in USD on a logarithmic scale from \$0.10 to \$10. Each point is one LLM and prompting-mode configuration; color indicates provider, marker shape indicates prompting mode. Pareto-optimal configurations have heavy black borders. Three labeled callouts identify GPT-4o-mini constrained at \$0.30 per 1,000 correct and 94.2\% accuracy, Claude Haiku 4.5 constrained at \$4.40 and 96.8\%, and Claude Opus 4.7 constrained at \$13.30 and 94.8\%, the last labeled as dominated by Haiku 4.5.}
\label{fig:cost}
\end{figure}

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
| Gemini 2.5 Pro\*               | constr.  | 46.2% |   0.0% | $1.57 |   $0.0068 |
| Claude Opus 4.7                | constr.  | 94.8% |   0.0% | $6.31 |   $0.0133 |

: Cost per correct prediction. *constr.* = constrained. \*Gemini 2.5 Pro constrained reflects abstention pattern, not error rate; see §4.3 (Failure-mode breakdown).

Reading the table top-down, the cheaper rows are all 4 to 19pp
lower on top-1 or carry nonzero hallucination. **GPT-4o-mini
constrained leads on deployment-relevance**: 94.2% top-1, 0%
hallucination, \$0.0003 per correct, the cheapest configuration
that achieves ≥94% top-1. Claude Haiku 4.5 constrained adds 2.6pp
accuracy at roughly 14× the cost. Claude Opus 4.7 constrained's
0.6pp accuracy edge does not justify its 44× cost: the largest
frontier model is not the deployment leader. Gemini 2.5 Pro's
elevated \$/correct reflects abstention behavior, not a normal
cost-accuracy tradeoff.

## Outcome distribution under D4 abbreviation stress

D4\_abbreviated is the strongest stress test in the matrix.
Explicit ICD codes and canonical display strings are stripped,
and clinical entities are replaced with jargon ("T2DM", "HTN",
"CKD-3"). String-matching baselines collapse (fuzzy and TF-IDF
both drop ~30pp from D3 to D4), so remaining top-1 accuracy must
come from semantic mapping. Anthropic constrained mode is robust
under D4 (Haiku 93.6%, Sonnet 90.4%, Opus 86.4% top-1, versus
100% at D1), and the failure mode is exclusively `category_match`.
Frontier zero-shot LLMs show a 5 to 15pp top-1 drop from D1 to D4
but maintain low hallucination. Gemini 2.5 Pro abstention worsens
monotonically from D1 to D4 across all three prompting modes.
Figure 3 shows the full six-bucket outcome distribution per
configuration under D4 stress.

\begin{figure}[tbp]
\centering
\includegraphics[width=0.75\textwidth,keepaspectratio]{../figures/figure3_d4_outcome_stack.pdf}
\caption{Outcome distribution per (model, prompting mode) configuration under D4 abbreviation stress. Each horizontal bar sums to 100\% across the six outcome buckets defined in \S2.5; bars are ordered by exact\_match share descending. Anthropic and OpenAI constrained configurations cluster at the top, dominated by exact\_match with a small category\_match remainder. Gemini 2.5 Pro at the bottom is dominated by no\_prediction (abstention), distinct from the hallucination-heavy string-matching baselines (\texttt{baseline:fuzzy}, \texttt{baseline:tfidf}) below it.}
\par\noindent\textit{Alt text: Horizontal stacked bar chart showing the six-bucket outcome distribution per configuration under D4 abbreviation stress, ordered by exact-match share descending. Top rows (Anthropic and OpenAI constrained configurations) are dominated by green segments (exact-match and category-match). The bottom row (Gemini 2.5 Pro zero-shot) is dominated by blue-grey segments (no-prediction, abstention). Two string-matching baselines near the bottom show large red segments (hallucination).}
\label{fig:d4stack}
\end{figure}
