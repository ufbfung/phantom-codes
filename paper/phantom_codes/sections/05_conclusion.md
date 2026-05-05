# Conclusion

Across a 27-LLM-configuration evaluation matrix on n=125 Synthea-
generated FHIR Conditions, fabrication of non-existent ICD-10-CM
codes was rare (median per-cell hallucination = 0%; eliminated
entirely for every Anthropic and OpenAI model under constrained
prompting). Hallucination behaves as a controllable property of
the prompting setup at this scale rather than a fixed model
limitation; abstention — separated from fabrication via the 6-way
taxonomy — emerged as the more consequential failure mode for
some models.

Two caveats temper the headline. First, the n=125 cohort and the
narrow 12-code ACCESS-Model vocabulary limit per-cell precision
and bound the population that surfaces fabrication. Second, the
D4 abbreviation-substitution stress is only one proxy for
real-world hallucination triggers; multi-condition narratives,
ambiguous documentation, code-specificity ambiguity, and long-tail
rare diagnoses are not exercised here. A v2 cohort with broader
vocabulary and more realistic clinical text (e.g., discharge
summaries, ED notes) is needed to establish the true effect size
of fabrication risk in production-relevant inputs.

For deployment, the comparison among configurations that already
achieve near-zero hallucination is not "lowest hallucination" but
**cost-per-correct sustainability**. GPT-4o-mini constrained at
\$0.0003 per exact match dominates every more expensive
configuration tested at this cohort's accuracy ceiling; production
deployment selection at population-coding scale should be anchored
in this sustainable-cost framing rather than in marginal safety
differences between configurations that are already at the floor
on fabrication.
