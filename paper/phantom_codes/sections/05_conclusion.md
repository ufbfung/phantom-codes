# Conclusion

Across a 24-LLM-configuration evaluation matrix on n=125 Synthea-
generated FHIR Conditions, fabrication of non-existent ICD-10-CM
codes ranged 0–5.6% for flagship models (Opus 4.7, GPT-5.5, Gemini
2.5 Pro) and 0–12.8% for sub-flagship models, and was eliminated
entirely under constrained prompting for every Anthropic and
OpenAI model. Hallucination behaves as a controllable property of
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

For deployment, two prescriptions emerge. First,
**zero-shot LLM prompting is unsafe for clinical concept
normalization** — every fabrication observed occurred under
zero-shot, and constrained or retrieval-augmented grounding is
required to eliminate it. Second, among grounded configurations,
**cost-per-correct rather than accuracy alone should drive
selection**: clinical coding errors are downstream-costly, and the
cheaper model can win on the combined metric. The deployment
frontier: **GPT-4o-mini constrained** (94.2% top-1,
\$0.0003/correct — cheapest at ≥94% accuracy) and **Claude
Haiku 4.5 constrained** (96.8%, \$0.0044 — +2.6pp accuracy at
~14× cost). Both dominate the largest frontier model (Opus 4.7
constrained, 94.8% / \$0.0133). Supported workflow:
LLM-augmented coding with terminologist review.
