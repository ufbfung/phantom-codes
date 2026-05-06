# S5 — Cost economics: extended analysis

This supplement extends §2 of the main manuscript. The headline
per-call costs and cost-per-correct numbers from the n=125 Synthea
run are reported in main §2; this section covers (a) the full
24-LLM × 4-mode cost table, (b) per-bucket cost decomposition, (c)
the original metric framework (M3 hallucination tax + M4
fixed-accuracy-floor cost), (d) projected production-deployment
costs at multiple organization scales, (e) the deployment break-even
analysis, (f) qualitative deployment considerations (caching
threshold quirks, reliability cost), and (g) the full version of the
framing caveats summarized briefly in main §2.

## S5.1 — Full cost table (24 LLM configurations × 4 modes)

The full per-(model, prompting-mode) cost breakdown is in
`results/summary/n125_run_v2/cost_per_correct.csv`. The headline
configurations are reproduced in main §2; the full set adds the 12
constrained and RAG configurations for the 9 LLMs other than Haiku
(Sonnet, Opus, GPT-5.5, GPT-4o-mini, Gemini 2.5 Pro/Flash, Gemini
3 Flash Preview), plus the 3 string-matching baselines and the
frozen sentence-transformer retrieval baseline (all zero LLM cost
by construction). Total run cost across the 28 configurations on
the n=125 cohort = \$49.65, dominated by Claude
Opus (\$18.18 across its 3 modes) and the GPT-5.5 / Sonnet 4.6
groupings.

## S5.2 — Per-bucket cost decomposition

For each (model, outcome bucket) cell, total \$ spent on that
bucket = count × per-call cost. The full table is in
`results/summary/n125_run_v2/per_bucket_cost.csv`. Headline pattern:
for the Anthropic constrained configurations, ~95% of total spend
went to exact-match outcomes, ~5% to category-match outcomes, and
~0% to hallucination/no\_prediction (mechanical consequence of the
0% hallucination rate observed in those cells). For Gemini 2.5 Pro
zeroshot, by contrast, ~32% of spend went to exact-match outcomes
and ~67% to no\_prediction (the model's empty-array responses still
incur input-token cost via the prompt, though output tokens are
zero or near-zero — see §S5.6 for the wrapper-level investigation
this surfaced). Spend on hallucination outcomes was negligible
across the matrix (consistent with the rare-fabrication finding in
§Results).

## S5.3 — Original metric framework (M3 + M4)

### M3: Hallucination tax

For each (model, mode), the marginal cost of a hallucinated
prediction is not just the API call — it's the API call plus the
downstream QA time required to catch and correct the bad code:

$$
\text{hallucination\_tax} = N_{\text{hallucinations}} \times (T_{\text{QA}} \times r_{\text{coder\_hr}} + \text{API\_cost\_per\_call})
$$

where $T_{\text{QA}}$ is QA time per flagged prediction (sweep
1–10 min), $r_{\text{coder\_hr}}$ is the human coder's hourly rate
(sweep \$30–\$80/hr), and the API cost component is from M1.

For the n=125 cohort, hallucination counts are low across most
configurations, so the headline-run absolute hallucination tax is
small in dollar terms (a few dollars at most across the matrix).
The tax surface becomes load-bearing at production deployment
scale (millions of encounters per year), where even a 1%
hallucination rate translates into tens of thousands of QA-hours
annually for a single deployment.

### M4: \$ per coded encounter at fixed accuracy floor

For each model, find the configuration (zero-shot vs. constrained
vs. RAG) that achieves accuracy ≥ X% and report cost. Allows
comparison of LLM zero-shot vs. RAG-LLM vs. trained-classifier at
iso-accuracy — the question a deployment team actually asks. From
the n=125 headline data:

- At the 90% top-1 accuracy floor: GPT-4o-mini constrained
  (94.2%, \$0.0003/correct) and Claude Haiku 4.5 constrained
  (96.8%, \$0.0044/correct) are the cost-frontier configurations.
- At the 95% top-1 accuracy floor: only GPT-4o-mini constrained
  (94.2%) and Claude Sonnet 4.6 constrained (94.6%) come close;
  Opus 4.7 constrained (94.8%) and GPT-5.5 constrained (93.6%)
  round out the top.
- At the 99% top-1 accuracy floor: no configuration on this cohort
  meets the floor; Claude Haiku constrained tops out at 96.8%.

These floors are interpretive only at n=125 with Wilson 95% CI
half-widths of ±5pp; replication at the originally-targeted n=500
would tighten the comparison.

## S5.4 — Production-deployment cost projections

A production deployment looks very different from the headline
benchmark because organizations choose ONE configuration (not the
full 28-config matrix) and run it on incoming records continuously.
We project costs for representative single-configuration
deployments at three production scales, using the per-call costs
observed in main §2 M1 with a conservative 1.4× scale-up for
real-clinical-text payloads (which carry more metadata than Synthea
Conditions). Average 5 coded conditions per healthcare encounter
(typical for inpatient discharges in cardiometabolic populations).

| Production model choice | $ per condition | $ per encounter (5 conditions) |
|---|---:|---:|
| Gemini 2.5 Flash | \$0.000053 | \$0.0003 |
| GPT-4o-mini | \$0.000199 | \$0.001 |
| Claude Haiku 4.5 (zeroshot) | \$0.0027 | \$0.013 |
| Claude Sonnet 4.6 | \$0.0087 | \$0.044 |
| GPT-5.5 | \$0.0060 | \$0.030 |
| Claude Opus 4.7 | \$0.0167 | \$0.083 |

Annual cost at three production volumes:

| Volume scenario | Conditions/year | Encounters/year | Haiku 4.5 (cheap, fast) | Opus 4.7 (premium) |
|---|---:|---:|---:|---:|
| Small clinic (single specialty) | 25,000 | 5,000 | **\$67** | **\$418** |
| Medium hospital | 250,000 | 50,000 | **\$668** | **\$4,175** |
| Health system (multi-hospital) | 2,500,000 | 500,000 | **\$6,675** | **\$41,750** |
| Population health platform | 25,000,000 | 5,000,000 | **\$66,750** | **\$417,500** |

Even at population-health platform scale, Haiku 4.5 deployment is
well under \$100k/year for the API spend portion. Opus 4.7 at the
same scale crosses into "real money" territory but is still small
compared to the salary cost of equivalent human coding capacity
(see §S5.5).

### Non-LLM baseline cost (negligible)

The string-matching and frozen sentence-transformer retrieval
baselines contribute essentially zero per-call cost: they run
locally with no API calls. The frozen sentence-transformer
(`all-MiniLM-L6-v2`) loads once into memory at startup; subsequent
candidate retrieval is sub-millisecond. These baselines are
included for accuracy comparison only and are excluded from the
LLM cost extrapolations above.

### Caveats to the production extrapolation

- Per-call cost is dominated by zero-shot mode in this projection.
  Constrained or RAG modes add ~1.5–2× per-call cost relative to
  zero-shot for the same model (per main §2 M1); production
  deployments using those modes would scale annual costs
  proportionally.
- Real-world prompt sizes vary by encounter type. ICU discharge
  summaries are denser than ED visits; cardiometabolic conditions
  specifically may have longer FHIR payloads due to ASCVD/CKD
  staging complexity. The 1.4× scale-up factor is an estimate, not
  a measured value (range 1.2–2.0× based on structured vs. legacy
  EHR formats).
- API pricing changes. The 2026-05-03 snapshot in
  `configs/pricing.yaml` will become stale. Cost numbers are
  reconstructible by recomputing the per-call cost from token
  counts in the per-prediction CSV under the reader's current
  pricing.
- Provider availability is non-uniform. The Gemini 2.5 Pro
  abstention pattern observed in our headline run (88% empty
  responses on zero-shot calls — see main §Results and
  §Limitations) implies a deployment-relevant overhead not modeled
  in this section.

## S5.5 — Comparison to human-coder cost (rough magnitude)

For order-of-magnitude grounding only — a precise break-even
analysis follows in §S5.6:

- Median U.S. medical-coder salary ≈ \$60,000/year fully loaded
  (≈ \$30/hr including benefits and overhead)
- Throughput ≈ 30–50 charts/day × ~250 working days/year =
  ~10,000 charts/year per coder
- Implied cost per coded encounter ≈ \$5–8 (human)
- Implied cost per coded encounter ≈ \$0.013 (Haiku 4.5) to
  \$0.083 (Opus 4.7) (LLM, automated)

LLM API spend alone is **60×–600× cheaper per encounter** than a
human coder's loaded cost, depending on model choice. This is NOT
the deployment break-even — actual deployment cost includes QA on
flagged predictions and exception handling. §S5.6 computes that
properly.

## S5.6 — Break-even analysis (LLM + QA vs. human coder)

> **Methods finalized; quantitative numbers from the n=125 headline
> run are limited by cohort size for some sensitivity dimensions.
> Replication at n=500 would sharpen the break-even surfaces.**

For each model configuration, total \$ per accurately-coded
encounter under four deployment scenarios:

1. **Human coder alone** — cost = $H/hr × (chart_throughput^-1)
2. **LLM, no QA** — cost = API cost per encounter; accuracy =
   top-1 rate (no review of model outputs)
3. **LLM, QA only on flagged predictions** — flag low-confidence
   outputs, hallucinated codes (mechanically detectable via the
   ICD-10-CM validator), and abstentions (no\_prediction);
   QA cost added per flag
4. **Trained classifier with QA** — comparable QA fraction;
   training cost is negligible per §S5.4

Sensitivity sweep dimensions: human-coder hourly rate (\$30–\$80/hr),
QA time per LLM-coded encounter (1–10 min), coder throughput
(20–60 charts/day), and per-(model, mode) failure rates (from §3
Results).

The open question this analysis contributes to: at what
hallucination + abstention rate does the LLM-with-QA pipeline
beat human-only on cost? At the failure rates observed in the
n=125 headline run (median per-cell hallucination = 0%; abstention
varies by model), LLM-with-QA pipelines using any of the
constrained-mode Anthropic, OpenAI, or Gemini Flash configurations
beat human-only on cost across all sensitivity-sweep parameter
ranges considered. Gemini 2.5 Pro under our wrapper settings is
the exception — its high abstention rate inflates the QA fraction
to the point where its per-correct cost approaches human-coder
parity at moderate QA rates.

## S5.7 — Real-world deployment considerations (qualitative)

### S5.7.1 Anthropic prompt-caching threshold

Anthropic's prompt caching only fires when the cacheable prefix
exceeds a model-specific minimum (Opus 4.7 = 4,096 tokens; Haiku
4.5 = 4,096; Sonnet 4.6 = 2,048). Below the threshold,
`cache_control: ephemeral` is silently no-opped — no error, no
warning, no cache hit on subsequent calls. For the v1 ACCESS-scope
candidate list (~85 codes, ~2.8k tokens of system prompt) prior to
the 2026-05-04 expansion to all 24 LLM configurations,
constrained-mode prompts sat below the Haiku 4.5 minimum and
Anthropic caching did not fire for any Anthropic configuration in
that earlier run. With the headline run's prompts, caching is
active for Opus 4.7 constrained (78% cache-read ratio) and Sonnet
4.6 constrained (80%) — see the `cache_read_tokens` columns in
the per-prediction CSV. Haiku 4.5 constrained still sits below the
threshold and shows no cache activity, consistent with the model-
specific minimum.

**Deployment implication:** organizations adopting LLM-based coding
need to audit whether their cacheable prefix actually exceeds the
threshold for their chosen model. A quick check
(`cache_creation_input_tokens == 0` and `cache_read_input_tokens
== 0` after the first call) reveals silent threshold misses. At
scale, missed caching can multiply cost 5–10× for constrained-mode
workloads.

OpenAI's automatic prefix caching has lower thresholds (~1,024
tokens) and fires more reliably. Google's implicit caching also
fires reliably.

### S5.7.2 Reliability is a cost factor too

The headline-run observed transient failures on `gemini-2.5-pro`
(~10% ServerError 503 + ClientError 429 across modes — see
§Results error breakdown) and a high empty-prediction rate
distinct from API failures. For a deployment relying on Gemini Pro
for code prediction, this implies either a retry/backoff layer
doubling the effective per-call latency, a second-line fallback
model (with its own cost profile) for failed calls, or accepting a
non-trivial fraction of records will need post-hoc retry. None of
these are free.

### S5.7.3 Pricing snapshots are reproducibility artifacts

API pricing changes. The cost numbers reported in main §2 and this
supplement are anchored to the pricing snapshot in
`configs/pricing.yaml` at the time of the headline run (2026-05-03,
date-stamped in the run manifest)
[@OpenAIPricing2026; @GeminiPricing2026]. A reader replicating the
experiment 6 months later may see different costs even with the
same token counts; the per-prediction CSV publishes token counts
alongside the dated cost-USD column so cost numbers are
reconstructible under any reader's current pricing.

## S5.8 — Framing limitations (full version)

A measured framing is essential for this section because the data
touches on workforce questions that go beyond what a benchmark can
answer:

- We measure the cost of *automated coding*, not the broader role
  of clinical informaticists or terminologists, who do far more
  than coding (workflow design, specialty content review, quality
  measure validation, data governance, etc.).
- Any break-even analysis is *deployment cost guidance*, not an
  *automation feasibility verdict*. The data informs the question;
  it doesn't answer it.
- Error tolerance is workflow-specific: a hallucinated code that
  flows into claims billing has different downstream cost than the
  same hallucination in a research cohort definition or
  population-health analytics dashboard.
- Cost numbers are based on public API pricing as of the run date.
  On-prem deployment (vLLM, dedicated endpoints, fine-tuned models
  served internally) has very different economics and is out of
  scope for v1.
- The cohort scale (n=125 unique conditions) limits the statistical
  precision of per-cell cost-per-correct comparisons (Wilson 95%
  CI half-widths of ±5pp on the underlying accuracy rates
  propagate into the cost-per-correct ratio); replication at n=500
  would tighten these.

## S5.9 — Historical: pre-headline-run smoke-test cost data

Preliminary per-call cost data from the 2026-05-02 Phase-0 wiring-
validation run on 6 fixture FHIR Conditions × 4 degradation modes
(24 records) × 9 LLM configurations (216 calls). Reported here for
historical reference; the headline-run numbers in main §2 supersede
them.

| Model × Mode (smoke test) | Successful calls | Mean in/out tokens | $ per call | p50 latency |
|---|---:|---:|---:|---:|
| `gemini-2.5-flash:zeroshot` | 24 | 346 / 41 | \$0.000038 | 1.7 s |
| `gpt-4o-mini:zeroshot` | 24 | 440 / 125 | \$0.000142 | 2.2 s |
| `gemini-2.5-pro:zeroshot` | 18 | 384 / 67 | \$0.001156 | 10.3 s |
| `claude-haiku-4-5:zeroshot` | 24 | 1,208 / 140 | \$0.001904 | 1.2 s |
| `claude-haiku-4-5:rag` | 24 | 1,710 / 151 | \$0.002467 | 1.3 s |
| `claude-haiku-4-5:constrained` | 24 | 3,356 / 137 | \$0.004042 | 1.2 s |
| `gpt-5.5:zeroshot` | 24 | 436 / 70 | \$0.004283 | 1.8 s |
| `claude-sonnet-4-6:zeroshot` | 24 | 1,209 / 174 | \$0.006246 | 2.8 s |
| `claude-opus-4-7:zeroshot` | 24 | 1,617 / 152 | \$0.011900 | 2.2 s |

Total smoke-test cost: \$0.7652 for 216 LLM calls. The smoke-test
per-call costs predicted the headline-run costs to within 2× on a
per-call basis, validating the projected-budget estimates that
informed the headline-run cost cap.

A finding worth surfacing from the smoke test that persisted into
the headline run: the same English prompt produces materially
different token counts across providers (~3× difference between
Anthropic and Google for the same content). This compounds with
per-token pricing to produce headline cost differences that are
NOT explained by accuracy or capability — they're partly
tokenizer-driven. Literature comparisons that don't normalize for
tokenization can mislead readers about relative deployment cost.
