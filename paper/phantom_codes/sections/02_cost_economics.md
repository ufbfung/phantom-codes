# Cost and deployment economics

Most LLM medical-coding benchmarks report accuracy in isolation
[@Soroush2024; @Almeida2025; @Bhatti2025], but the deployment
decision hinges on the joint distribution of API cost per call,
per-call accuracy, and the rate of failure modes that drive
downstream QA cost (hallucination, abstention). For every model
invocation in our matrix the per-prediction CSV records token usage,
latency, and `cost_usd` computed at runtime against the versioned
pricing snapshot in `configs/pricing.yaml` [@OpenAIPricing2026;
@GeminiPricing2026]. These are infrastructure metrics — recorded
even when the prediction is wrong — and serve as the basis for the
cost-per-correct-prediction normalization reported in §Results.
Selected configurations from the headline matrix are shown below;
the full 29-configuration breakdown, per-bucket cost decomposition,
break-even analysis (LLM + QA vs.\ human coder), and annual cost
projections at deployment scale are reported in Supplementary §S5.

| Model × Mode | n | $ total | $ per call (mean) |
|---|---:|---:|---:|
| claude-opus-4-7:zeroshot | 500 | 6.77 | 0.01354 |
| claude-sonnet-4-6:zeroshot | 500 | 3.41 | 0.00682 |
| claude-haiku-4-5:constrained | 500 | 2.14 | 0.00428 |
| gpt-5.5:zeroshot | 499 | 3.59 | 0.00720 |
| gpt-4o-mini:constrained | 500 | 0.14 | 0.00028 |
| gemini-2.5-pro:zeroshot | 424 | 0.41 | 0.00096 |
| gemini-2.5-flash:zeroshot | 500 | 0.03 | 0.00005 |

A 270× spread separates the cheapest configuration (Gemini 2.5 Flash
zeroshot at \$0.00005/call) from the most expensive (Claude Opus 4.7
zeroshot at \$0.01354/call). Per-call price varies more dramatically
across the matrix than per-call accuracy, so cost-per-correct
prediction (M2 in §Results) is dominated by per-call price for any
configuration that achieves ≥80% top-1 accuracy. A measured caveat:
the cost numbers here are *automated-coding cost*, not the broader
role of clinical informaticists. Any break-even analysis is
*deployment cost guidance*, not an *automation feasibility verdict* —
error tolerance is workflow-specific, and a hallucinated code that
flows into claims billing has different downstream cost than the same
hallucination in a research cohort definition. The full caveats are
in Supplementary §S5.
