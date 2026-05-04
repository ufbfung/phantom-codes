# Cost and deployment economics

> **Status:** Draft v0 (2026-05-02; minor update for compliance framing).
> Placeholder structure — numbers in this section will be filled in from
> the locked headline run on Synthea-generated FHIR Bundles (not MIMIC;
> see Methods § "Data and policy compliance"). The framing, methods, and
> qualitative observations below are settled.
>
> **Section placement:** Results subsection (not Discussion), since it's
> empirical. Discussion engages with deployment implications separately.
>
> **Citations:** Almeida et al. 2025 [@Almeida2025] is the strongest precedent
> for this multi-axis evaluation framing. MAX-EVAL-11 [@Bhatti2025] reports
> weighted-relevance scoring but no cost. Soroush et al. 2024 [@Soroush2024]
> doesn't address cost at all. Position our contribution as the first
> benchmark to operationalize *cost-per-correct-prediction* and a
> *hallucination-aware break-even analysis* for clinical concept
> normalization.

---

## Why cost belongs in this paper

Most LLM-coding benchmarks report accuracy metrics in isolation: top-1, F1,
hierarchical match rates, or weighted variants. None of these tell a healthcare
organization whether deploying an LLM-based coding pipeline is *economically
viable*. The literature treats cost as an afterthought; we treat it as a
first-class outcome alongside accuracy.

This matters because the deployment decision hinges on the joint distribution
of three quantities:

1. **API cost per call** — directly billable
2. **Accuracy** — what fraction of calls produce a correct top-1 code
3. **Hallucination rate** — what fraction produce a non-existent code that
   downstream systems will silently mishandle (driving QA cost or, worse,
   propagating into billing or research databases)

A model that's cheap per call but hallucinates heavily may cost MORE per
correct coded encounter than a more expensive model with cleaner output, once
you account for the human QA needed to catch hallucinations. We make this
trade-off explicit and quantitative.

## What we measure

For every model invocation in our eval matrix, we record (in the per-prediction
CSV):

- `input_tokens`, `output_tokens` — billable token counts, extracted directly
  from the provider's response object (Anthropic `usage`, OpenAI
  `prompt_tokens_details`, Google `usage_metadata`)
- `cache_read_tokens`, `cache_creation_tokens` — billable cache traffic where
  applicable
- `latency_ms` — wall-clock time for the API call
- `cost_usd` — computed at runtime from a versioned pricing snapshot
  (`configs/pricing.yaml`), date-stamped and persisted in the run manifest
  for reproducibility

These are infrastructure metrics, not performance metrics — they're recorded
even when the prediction is wrong.

## Preliminary cost data (Phase-0 smoke test, 2026-05-02)

The following per-model, per-call costs are from a Phase-0 wiring-validation
run on 6 in-scope synthetic FHIR Conditions × 4 degradation modes = 24
records, evaluated against 9 LLM configurations (216 total LLM
invocations, 210 successful, 6 transient `ServerError 503` failures on
Gemini 2.5 Pro). Costs are computed from the per-prediction CSV's
`cost_usd` column using the pricing snapshot in `configs/pricing.yaml`
(snapshot date: 2026-05-01).

> **These numbers are infrastructure validation, not findings.** Per the
> project's research-integrity discipline, we do not draw performance
> conclusions from smoke-test runs on synthetic fixtures. The cost
> figures here are reported because (a) cost is an infrastructure metric,
> not a performance metric, and (b) they ground the extrapolations in the
> next subsection.

### Per-call cost (smoke test, successful calls only)

| Model × Mode | Successful calls | Mean in / out tokens | $ per call | p50 latency |
|--------------|-----------------:|---------------------:|-----------:|------------:|
| `gemini-2.5-flash:zeroshot` | 24 | 346 / 41 | $0.000038 | 1.7 s |
| `gpt-4o-mini:zeroshot` | 24 | 440 / 125 | $0.000142 | 2.2 s |
| `gemini-2.5-pro:zeroshot` | 18 | 384 / 67 | $0.001156 | 10.3 s |
| `claude-haiku-4-5:zeroshot` | 24 | 1,208 / 140 | $0.001904 | 1.2 s |
| `claude-haiku-4-5:rag` | 24 | 1,710 / 151 | $0.002467 | 1.3 s |
| `claude-haiku-4-5:constrained` | 24 | 3,356 / 137 | $0.004042 | 1.2 s |
| `gpt-5.5:zeroshot` | 24 | 436 / 70 | $0.004283 | 1.8 s |
| `claude-sonnet-4-6:zeroshot` | 24 | 1,209 / 174 | $0.006246 | 2.8 s |
| `claude-opus-4-7:zeroshot` | 24 | 1,617 / 152 | $0.011900 | 2.2 s |

**Total smoke-test cost: $0.7652** for 216 LLM calls + 96 baseline calls
(baselines have $0 API cost). This validates that running the full
benchmark matrix at modest scale costs less than $1 — a useful upper
bound for iteration during scaffolding.

### Provider-tokenizer differences (Phase-0 observation)

A finding worth surfacing: the same English prompt produces materially
different token counts across providers. Our zero-shot system prompt is
identical across all three providers, yet:

- Anthropic Haiku: ~1,208 input tokens
- OpenAI (GPT-4o-mini, GPT-5.5): ~440 input tokens
- Google Gemini Flash/Pro: ~350–380 input tokens

That's roughly a 3× difference in tokenization between Anthropic and
Google for the same content. This compounds with per-token pricing to
produce headline cost differences that are NOT explained by accuracy or
capability — they're partly tokenizer-driven. We flag this because
literature comparisons that don't normalize for tokenization can mislead
readers about relative deployment cost.

### Caching observations (Phase-0)

- **Anthropic:** all five Anthropic configurations show
  `cache_read_input_tokens = 0` across 24 calls. As discussed in §
  *Real-world deployment considerations*, this is the model-specific
  threshold issue: our system prompts (max ~2,800 tokens of cacheable
  prefix in constrained mode) sit below Haiku 4.5 and Opus 4.7's 4,096
  minimum. Caching is correctly enabled in code; it just doesn't
  activate at our v1 prompt sizes.
- **OpenAI:** automatic prefix caching reported 0 cache reads at our
  prompt sizes (~440 tokens), below OpenAI's typical ~1,024-token
  minimum.
- **Google:** Gemini Flash showed implicit caching (`cache_read_tokens =
  479` aggregate across 24 calls) — Google appears to cache more
  aggressively at smaller prompt sizes. Pro did not.

The cache-miss observation has direct cost implications for the next
section's extrapolations: at our v1 prompt sizes, **none of the
extrapolated costs benefit from caching discounts**. If v2 expands the
candidate list above the Anthropic threshold, expected cost reductions
of 2–8× on constrained-mode workloads would apply (cache reads cost ~10%
of base input tokens).

## Extrapolation to headline MIMIC run

For the headline run, we plan to evaluate the full 9-configuration LLM
matrix against MIMIC-IV-FHIR Conditions filtered to ACCESS Model scope
(diabetes, ASCVD, CKD stage 3, hypertension, dyslipidemia, prediabetes,
obesity). Each in-scope condition produces 4 records (one per
degradation mode) and is evaluated by every LLM configuration, plus the
4 non-LLM models (3 string baselines + 1 retrieval baseline).

### Per-condition LLM cost

Aggregating Phase-0 per-call costs across all 9 LLM configurations × 4
degradation modes = 36 LLM invocations per condition:

| Configuration | Cost per condition (4 modes) |
|--------------|----------------------------:|
| `gemini-2.5-flash:zeroshot` | $0.0002 |
| `gpt-4o-mini:zeroshot` | $0.0006 |
| `gemini-2.5-pro:zeroshot` | $0.0046 |
| `claude-haiku-4-5:zeroshot` | $0.0076 |
| `claude-haiku-4-5:rag` | $0.0099 |
| `claude-haiku-4-5:constrained` | $0.0162 |
| `gpt-5.5:zeroshot` | $0.0171 |
| `claude-sonnet-4-6:zeroshot` | $0.0250 |
| `claude-opus-4-7:zeroshot` | $0.0476 |
| **Sum (full 9-config matrix)** | **$0.1288 per condition** |

### Real-Synthea scale-up factor

Smoke-test fixtures contain minimal FHIR Condition payloads. Synthea-
generated Conditions carry additional metadata (extensions, provenance
references, encounter linkages) that inflate input tokens for D1_full
and D2_no_code modes specifically. D3_text_only and D4_abbreviated modes
are dominated by short text and should be unaffected. Conservative
scale-up factor based on Synthea Bundle structure: **1.3–1.5× higher
per-call cost on real Synthea Conditions** versus our hand-built
fixtures. (Synthea is the headline-run input; MIMIC trains the trained
models in a separate flow that doesn't touch the LLM eval matrix or its
costs — see Methods.)

### Projected headline-run cost at three cohort sizes

> **Estimates assume a 1.4× scale-up applied to the full 9-configuration
> matrix. Synthea cohort size is a parameter we control via Synthea's
> generation config; we bracket plausible ranges below.**

| Cohort size (in-scope conditions) | Total records (×4 modes) | LLM invocations (×9 configs) | Estimated total cost |
|----------------------------------:|--------------------------:|------------------------------:|--------------------:|
| 500 (small pilot) | 2,000 | 18,000 | **$90** |
| 2,500 (moderate) | 10,000 | 90,000 | **$450** |
| 10,000 (large) | 40,000 | 360,000 | **$1,800** |
| 25,000 (full ACCESS scope, upper-bound estimate) | 100,000 | 900,000 | **$4,500** |

**Per-model cost contribution** for a 10,000-condition cohort
(illustrative — Opus 4.7 dominates the bill):

| Model | Share of total cost |
|-------|--------------------:|
| `claude-opus-4-7:zeroshot` | ~37% (~$667) |
| `claude-sonnet-4-6:zeroshot` | ~19% (~$346) |
| `gpt-5.5:zeroshot` | ~13% (~$237) |
| `claude-haiku-4-5:constrained` | ~13% (~$226) |
| `claude-haiku-4-5:rag` | ~8% (~$138) |
| `claude-haiku-4-5:zeroshot` | ~6% (~$107) |
| `gemini-2.5-pro:zeroshot` | ~4% (~$64) |
| `gpt-4o-mini:zeroshot` | <1% (~$8) |
| `gemini-2.5-flash:zeroshot` | <1% (~$3) |

### Cost-reduction levers we have

If headline-run cost becomes a bottleneck, we have several knobs:

1. **Drop Opus 4.7 from the matrix.** Single largest line item; v2 paper
   could include it. Reduces total by ~37%.
2. **Stratified sampling.** A representative ~2,500-condition subset
   gives statistically reliable outcome rates at ~$450 instead of $4,500
   for the full cohort.
3. **Prompt-size optimization.** Expand the candidate list above
   Anthropic's caching threshold to enable the ~10× cache-read
   discount on Anthropic constrained-mode calls. Methodology change —
   needs decision before lock.
4. **Skip RAG mode in v1.** Constrained alone tests menu-following;
   RAG adds per-record retrieval. Reduces total by ~8%.

## Extrapolation to production deployment scale

A production deployment looks very different from the headline benchmark
because organizations choose ONE configuration (not the full 9-config
matrix) and run it on incoming records continuously. We project costs
for representative single-configuration deployments at three production
scales.

### Per-encounter cost (single LLM configuration)

Production deployments would operate against real-world clinical text
(MIMIC-like or richer EHR notes), not against Synthea. Assumes 1.4×
real-clinical-text scale-up applied to per-call costs, and an average
of 5 coded conditions per healthcare encounter (typical for inpatient
discharges in cardiometabolic populations):

| Production model choice | $ per condition | $ per encounter (5 conditions) |
|-------------------------|----------------:|-------------------------------:|
| Gemini 2.5 Flash | $0.000053 | $0.0003 |
| GPT-4o-mini | $0.000199 | $0.001 |
| Claude Haiku 4.5 (zeroshot) | $0.0027 | $0.013 |
| Claude Sonnet 4.6 | $0.0087 | $0.044 |
| GPT-5.5 | $0.0060 | $0.030 |
| Claude Opus 4.7 | $0.0167 | $0.083 |

### Annual cost at three production volumes

| Volume scenario | Conditions/year | Encounters/year | Haiku 4.5 (cheap, fast) | Opus 4.7 (premium) |
|----------------|----------------:|----------------:|------------------------:|--------------------:|
| Small clinic (single specialty) | 25,000 | 5,000 | **$67** | **$418** |
| Medium hospital | 250,000 | 50,000 | **$668** | **$4,175** |
| Health system (multi-hospital) | 2,500,000 | 500,000 | **$6,675** | **$41,750** |
| Population health platform | 25,000,000 | 5,000,000 | **$66,750** | **$417,500** |

Even at population-health platform scale, Haiku 4.5 deployment is well
under $100k/year for the API spend portion. Opus 4.7 at the same scale
crosses into "real money" territory but is still small compared to the
salary cost of equivalent human coding capacity.

### Comparison to human-coder cost (rough magnitude)

For order-of-magnitude grounding only — a precise break-even analysis
follows in the next subsection:

- Median U.S. medical-coder salary ≈ $60,000/year fully loaded
  (≈ $30/hr including benefits and overhead)
- Throughput ≈ 30–50 charts/day × ~250 working days/year = ~10,000
  charts/year per coder
- Implied cost per coded encounter ≈ $5–8 (human)
- Implied cost per coded encounter ≈ $0.013 (Haiku) to $0.083 (Opus) (LLM)

So the LLM API spend alone is **60×–600× cheaper per encounter** than a
human coder's loaded cost, depending on model choice. **This is NOT the
break-even** — the actual deployment cost includes QA on flagged
predictions and exception handling. The next subsection computes that
properly.

### Trained-model training cost (negligible at v1 scale)

The trained PubMedBERT classifier and frozen sentence-transformer
retrieval baseline contribute essentially zero training cost at v1
scale: training runs on Brian's M1 MacBook Pro via PyTorch's MPS
backend (Apple Silicon GPU). Hardware is sunk cost; energy use over a
~5-hour training run is dominated by the device's idle floor and
rounds to pennies. We report this for honesty rather than impact —
locally-trained classifiers are essentially free relative to per-call
LLM API spend.

For reproducibility purposes: equivalent training on a Vertex AI A100
40 GB instance would be ~1-2 hours wall clock at ~$3-5/hr = roughly
$5-10 per training run. Comparable on AWS p4d.24xlarge or Azure
NCv3-series. Local Apple Silicon is the cheapest path; cloud GPU is
viable if you don't have suitable local hardware.

### Caveats to the production extrapolation

- Per-call cost is dominated by zero-shot mode in this projection.
  Production deployments using constrained or RAG modes (which our data
  shows are ~2× more expensive per call for Haiku) would see roughly 2×
  the costs above. The trade-off is the constrained/RAG modes likely
  have lower hallucination rates — quantified in headline results.
- Real-world prompt sizes vary by encounter type. ICU discharge
  summaries are denser than ED visits; cardiometabolic conditions
  specifically may have longer FHIR payloads due to ASCVD/CKD staging
  complexity. Our 1.4× scale-up factor is an estimate, not a measured
  value. Sensitivity range: 1.2× (cleanly structured EHRs) to 2.0×
  (legacy systems with verbose FHIR).
- API pricing changes. The 2026-05-01 snapshot in
  `configs/pricing.yaml` will become stale. We commit to publishing the
  pricing table alongside cost numbers so readers can recompute under
  current pricing.
- These projections assume stable provider availability. The Gemini Pro
  failure rate observed in our smoke test (~25% transient 503s during
  one window) implies a deployment-relevant overhead we're not modeling
  here.

## Headline cost metrics

The numbers below are populated from the n=125 Synthea headline
evaluation (2026-05-04). The full per-(model, prompting-mode)
breakdown across all 24 LLM configurations and the per-outcome-bucket
cost decomposition appear in §Supplementary S5; the headline-relevant
subset that anchors the deployment-cost discussion below is reported
here.

### M1: Total run cost by (model, prompting mode)

Selected configurations from the headline matrix; full table in
§Supplementary S5. "n" is the count of billed (cost > 0) rank-0 calls;
configurations with high abstention rates (e.g., gemini-2.5-pro) show
n < 500 because abstention API responses do not bill output tokens
even though the call counts toward our matrix.

| Model × Mode | n | $ total | $ per call (mean) | $ per call (median) |
|---|---:|---:|---:|---:|
| claude-opus-4-7:zeroshot | 500 | 6.77 | 0.01354 | 0.01315 |
| claude-sonnet-4-6:zeroshot | 500 | 3.41 | 0.00682 | 0.00639 |
| claude-haiku-4-5:zeroshot | 500 | 1.12 | 0.00223 | 0.00226 |
| claude-haiku-4-5:constrained | 500 | 2.14 | 0.00428 | 0.00424 |
| claude-haiku-4-5:rag | 500 | 1.22 | 0.00243 | 0.00250 |
| gpt-5.5:zeroshot | 499 | 3.59 | 0.00720 | 0.00586 |
| gpt-4o-mini:zeroshot | 500 | 0.07 | 0.00015 | 0.00016 |
| gemini-2.5-pro:zeroshot | 424 | 0.41 | 0.00096 | 0.00089 |
| gemini-2.5-flash:zeroshot | 500 | 0.03 | 0.00005 | 0.00006 |

The 270× spread in per-call mean cost across the matrix
(\$0.00005 for Gemini 2.5 Flash to \$0.0135 for Claude Opus 4.7)
sets the floor for the deployment-cost analysis below: per-call
price varies more dramatically than per-call accuracy in this
cohort, so cost-per-correct-prediction (M2 below) is dominated by
the per-call price for any model that achieves ≥80% top-1
accuracy.

### M2: Cost-per-correct-prediction

For each (model, mode), compute:
$$
\text{cost\_per\_correct} = \frac{\text{total\_cost}}{\text{exact\_match\_count}}
$$

This is the simplest unified metric — it normalizes for accuracy by asking
"how much did you pay per code you got right?" Lets us plot cost vs. accuracy
on a single 2D scatter where the headline trade-off becomes visually obvious.

### M3: Hallucination tax

For each (model, mode), the marginal cost of a hallucinated prediction is
*not* just the API call — it's the API call PLUS the downstream QA time
required to catch and correct the bad code. Define:
$$
\text{hallucination\_tax} = N_{\text{hallucinations}} \times (T_{\text{QA}} \times \text{rate}_{\text{coder\_hr}} + \text{API\_cost\_per\_call})
$$

Sweep over reasonable values of QA time per flagged prediction (1–10 min) and
human-coder hourly rate ($30–$80/hr) and report a sensitivity surface, not a
point estimate.

### M4: $ per coded encounter at fixed accuracy floor

For each model, find the configuration (zero-shot vs constrained vs RAG)
that achieves accuracy ≥ X% and report cost. Lets us compare LLM zero-shot
vs RAG-LLM vs trained-classifier at iso-accuracy — the question a deployment
team actually asks.

## Real-world deployment considerations (qualitative)

### Anthropic prompt caching has model-specific minimum block sizes

A finding from our Phase-0 smoke test that's worth calling out: Anthropic's
prompt caching only fires when the cacheable prefix exceeds a model-specific
minimum (Opus 4.7 = 4,096 tokens; Haiku 4.5 = 4,096; Sonnet 4.6 = 2,048).
Below the threshold, `cache_control: ephemeral` is silently no-opped — no
error, no warning, no cache hit on subsequent calls. This is documented in
the Anthropic API docs but easy to miss in production deployments.

For our v1 ACCESS-scope candidate list (~85 codes, ~2.8k tokens of system
prompt), constrained-mode prompts sit *below* the Haiku 4.5 minimum.
Anthropic caching does not fire for any of our Anthropic configurations.
This is the API behaving correctly; just a constraint at our prompt sizes.

**Deployment implication:** organizations adopting LLM-based coding need to
audit whether their cacheable prefix actually exceeds the threshold for
their chosen model. A quick check (`cache_creation_input_tokens == 0` and
`cache_read_input_tokens == 0` after the first call) reveals silent
threshold misses. At scale, missed caching can multiply cost 5–10× for
constrained-mode workloads.

OpenAI's automatic prefix caching has lower thresholds (~1,024 tokens) and
fires more reliably for our prompt sizes. Google's implicit caching also
fires reliably in our smoke test (Gemini Flash showed cache hits even on
small prompts).

### Reliability is a cost factor too

Our Phase-0 smoke test observed transient failures on `gemini-2.5-pro`
(~25% of calls returned 503 "model experiencing high demand," even on the
paid tier). For a deployment relying on Pro for code prediction, this
implies either:

- A retry/backoff layer doubling the effective per-call latency
- A second-line fallback model (with its own cost profile) for failed calls
- Accepting a non-trivial fraction of records will need post-hoc retry

None of these are free. We report observed failure rates per model as a
deployment-relevant data point, not a model-quality judgment.

### Pricing snapshots are reproducibility artifacts

API pricing changes. The cost numbers we report are anchored to the
pricing snapshot in `configs/pricing.yaml` at the time of the headline run
(date stamped in the run manifest). A reader replicating the experiment 6
months later may see different costs even with the same token counts.

We commit to publishing the pricing snapshot alongside the per-prediction
data so cost numbers are reconstructible.

## Break-even analysis: LLM + QA vs. human coder (TBD — populate from headline run)

> **Placeholder analysis — methods finalized, numbers TBD.**

For each model configuration, plot total $ per accurately-coded encounter
under four deployment scenarios:

1. **Human coder alone** — cost = $H/hr × (chart_throughput^-1)
2. **LLM, no QA** — cost = API cost per encounter; accuracy = top-1 rate
3. **LLM, QA only on flagged predictions** — flag low-confidence or
   non-existent-code predictions; QA cost added per flag
4. **Trained classifier with QA** — comparable QA fraction; cost dominated
   by amortized training cost

Sensitivity sweep over: human-coder hourly rate ($30–$80/hr), QA time per
LLM-coded encounter (1–10 min), coder throughput (20–60 charts/day),
hallucination rate (per-model from headline run).

Report break-even surfaces, not point estimates. **The open question we
contribute to:** at what hallucination rate does the LLM-with-QA pipeline
beat human-only on cost?

## Framing limitations (caveats for Discussion)

A measured framing is essential for this section because the data we
report touches on workforce questions that go beyond what a benchmark can
answer:

- We measure the cost of *automated coding*, not the broader role of
  clinical informaticists or terminologists, who do far more than coding
  (workflow design, specialty content review, quality measure validation,
  data governance, etc.)
- Any break-even analysis is *deployment cost guidance*, not an
  *automation feasibility verdict*. The data informs the question; it
  doesn't answer it
- Error tolerance is workflow-specific: a hallucinated code that flows into
  claims billing has different downstream cost than the same hallucination
  in a research cohort definition or population-health analytics dashboard
- Our cost numbers are based on public API pricing as of the run date.
  On-prem deployment (vLLM, dedicated endpoints, fine-tuned models served
  internally) has very different economics and is out of scope for this
  paper
