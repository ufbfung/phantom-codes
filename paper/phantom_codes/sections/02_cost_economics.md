# Cost and deployment economics

Most LLM medical-coding benchmarks report accuracy in isolation
[@Soroush2024; @Almeida2025; @Singh2025]. The deployment decision,
however, hinges on the joint distribution of API cost per call,
per-call accuracy, and the rate of failure modes that drive
downstream QA cost (hallucination and abstention). For every model
invocation in our matrix the per-prediction CSV records token
usage, latency, and `cost_usd` computed at runtime against the
versioned pricing snapshot in `configs/pricing.yaml`
[@OpenAIPricing2026; @GeminiPricing2026]. These are infrastructure
metrics, recorded even when the prediction is wrong, and serve as
the basis for the cost-per-correct-prediction normalization
reported in §Results. Selected configurations from the headline
matrix are shown below; the full 28-configuration breakdown,
per-bucket cost decomposition, break-even analysis (LLM and QA
versus a human coder), and annual cost projections at deployment
scale are reported in Supplementary §S5.

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
cost-per-correct prediction (M2 in §Results) is dominated by
per-call price for any configuration that achieves ≥80% top-1
accuracy. One caveat: the cost numbers reported here are
*automated-coding cost*, not the broader role of clinical
informaticists. Any break-even analysis is *deployment cost
guidance*, not an *automation feasibility verdict*. Error
tolerance is workflow-specific, and a hallucinated code that flows
into claims billing has different downstream cost than the same
hallucination in a research cohort definition. The full caveats
are in Supplementary §S5.
