# Cost and deployment economics

> **Status:** Draft v0 (2026-05-02). Placeholder structure — numbers in this
> section will be filled in from the locked headline run on real MIMIC. The
> framing, methods, and qualitative observations below are settled.
>
> **Section placement:** Results subsection (not Discussion), since it's
> empirical. Discussion engages with deployment implications separately.
>
> **Citations:** Almeida et al. 2025 [Almeida 2025] is the strongest precedent
> for this multi-axis evaluation framing. MAX-EVAL-11 [Bhatti 2025] reports
> weighted-relevance scoring but no cost. Soroush et al. 2024 [Soroush 2024]
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

## Headline cost metrics (TBD — populate from headline run)

### M1: Total run cost by (model, prompting mode)

> **Placeholder table — fill in from headline run.**

| Model × Mode | Calls | $ total | $ per call (mean) | $ per call (median) |
|--------------|-------|---------|-------------------|---------------------|
| claude-opus-4-7:zeroshot | TBD | TBD | TBD | TBD |
| claude-sonnet-4-6:zeroshot | TBD | TBD | TBD | TBD |
| claude-haiku-4-5:zeroshot | TBD | TBD | TBD | TBD |
| claude-haiku-4-5:constrained | TBD | TBD | TBD | TBD |
| claude-haiku-4-5:rag | TBD | TBD | TBD | TBD |
| gpt-5.5:zeroshot | TBD | TBD | TBD | TBD |
| gpt-4o-mini:zeroshot | TBD | TBD | TBD | TBD |
| gemini-2.5-pro:zeroshot | TBD | TBD | TBD | TBD |
| gemini-2.5-flash:zeroshot | TBD | TBD | TBD | TBD |

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
