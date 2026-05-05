# Cost and deployment economics

> **Status:** Draft v1 (2026-05-04). Numbers populated from the n=125
> Synthea headline run. Most of the v0 framing — preliminary
> smoke-test data, projected-cost extrapolations, full per-bucket
> cost decomposition, break-even analysis, and qualitative
> deployment considerations — has moved to §Supplementary S5
> (`cost_economics_extended`). The main-text content here is
> limited to the per-call cost numbers from the headline run, the
> cost-per-correct framing, and a one-paragraph deployment-cost
> summary. Per [@Almeida2025], cost-aware reporting is increasingly
> expected for LLM medical-coding evaluations; this section
> operationalizes that for the matrix in §Methods.

---

## Why cost belongs in this paper

Most LLM medical-coding benchmarks report accuracy metrics in
isolation: top-1, F1, hierarchical match rates, or weighted variants
[@Soroush2024; @Almeida2025; @Bhatti2025]. None tell a healthcare
organization whether deploying an LLM-based coding pipeline is
*economically viable*. The deployment decision hinges on the joint
distribution of three quantities: API cost per call (directly
billable), per-call accuracy (what fraction produce a correct top-1
code), and the rate of failure modes that drive downstream QA cost
(hallucination, abstention). A model that's cheap per call but
fabricates heavily may cost more per correct coded encounter than a
more expensive model with cleaner output, once human QA is priced
in. We make this trade-off explicit and quantitative.

## What we measure

For every model invocation in the eval matrix, the per-prediction
CSV records `input_tokens`, `output_tokens`, `cache_read_tokens`,
`cache_creation_tokens` (extracted from the provider's response
object), `latency_ms`, and `cost_usd` (computed at runtime against
the versioned pricing snapshot in `configs/pricing.yaml`,
date-stamped and persisted in the run manifest for reproducibility
[@OpenAIPricing2026; @GeminiPricing2026]). These are infrastructure
metrics — recorded even when the prediction is wrong — and serve as
the basis for both the per-call cost table below and the
cost-per-correct-prediction normalization.

## M1: Total run cost by (model, prompting mode)

Selected configurations from the headline matrix; full 24-LLM
breakdown in §Supplementary S5. "n" is the count of billed
(cost > 0) rank-0 calls; configurations with high abstention rates
(notably gemini-2.5-pro) show n < 500 because abstention API
responses do not bill output tokens even though the call counts
toward our matrix.

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

A 270× spread in per-call mean cost separates the cheapest
configuration (Gemini 2.5 Flash zeroshot at \$0.00005/call) from
the most expensive (Claude Opus 4.7 zeroshot at \$0.01354/call).
Per-call price varies more dramatically across the matrix than
per-call accuracy, so cost-per-correct-prediction (M2 below) is
dominated by per-call price for any model that achieves ≥80% top-1
accuracy.

## M2: Cost-per-correct-prediction

For each (model, mode), cost-per-correct = total\_cost /
exact\_match\_count. This single metric collapses per-call price
and per-call accuracy into a deployment-ready number. From the
headline-run aggregates (full table in §Results):

- **Cheapest reliable**: GPT-4o-mini constrained at \$0.0003 per
  exact-match prediction (94.2% top-1, 0% hallucination on the
  n=125 cohort)
- **Cheapest period**: Gemini 2.5 Flash zeroshot at \$0.0001 per
  exact-match (75.4% top-1, 43.2% hallucination on D4 — high
  abstention-or-fabrication rate, so the cheap per-correct number
  comes with a large QA tail)
- **Most expensive reliable**: Claude Opus 4.7 constrained at
  \$0.0133 per exact-match (94.8% top-1, 0% hallucination)
- The 4-orders-of-magnitude spread in cost-per-correct mirrors the
  per-call spread; cost-per-correct is *not* primarily an
  accuracy story at this cohort scale

## Deployment-cost framing

Headline-run API spend totaled \$49.65 across the full 29-config
matrix on 125 unique conditions × 4 degradation modes = 500
EvalRecord items per LLM. Single-configuration production
deployments would operate against a single LLM rather than the
full matrix; per-condition cost for representative single-config
deployments ranges from approximately \$0.0001 (Gemini 2.5 Flash
zeroshot) to \$0.013 (Claude Opus 4.7 zeroshot), with constrained
and RAG modes adding 1.5–2× per-call cost relative to zeroshot for
the same model. Annual cost projections for healthcare
organizations operating at 25,000 to 25 million coded encounters
per year, the deployment-cost break-even analysis (LLM + QA vs.
human coder), the hallucination-tax sensitivity surfaces (M3 in
the original metric framework), and per-bucket cost decomposition
(what \$ was spent on hallucinations vs. exact matches vs.
abstentions) are reported in §Supplementary S5.

A measured caveat: the cost numbers here are *automated coding
cost*, not the broader role of clinical informaticists or
terminologists who do far more than coding (workflow design,
specialty content review, quality measure validation, data
governance). Any break-even analysis is *deployment cost guidance*,
not an *automation feasibility verdict*. Error tolerance is also
workflow-specific — a hallucinated code that flows into claims
billing has different downstream cost than the same hallucination
in a research cohort definition. The full version of these caveats
is in §Supplementary S5.
