# Conclusions

Across a 24-LLM × 4-degradation-mode evaluation matrix on
n=125 Synthea-generated FHIR Conditions, fabrication of
non-existent ICD-10-CM codes is rare in modern frontier LLMs
(median per-cell hallucination = 0%; 90th-percentile = ~39%
driven by two Gemini Flash zero-shot configurations). Constrained
prompting eliminates fabrication entirely for every Anthropic and
OpenAI model in the matrix that shows any nonzero zero-shot
hallucination — confirming hallucination as a controllable
property of the prompting setup, not a fixed model limitation.
Abstention, separated from fabrication via a 6-way outcome
taxonomy, surfaces as the more consequential failure mode for
some models (notably Gemini 2.5 Pro, ~88% empty-prediction rate
on zero-shot D4) and is methodologically distinct because it
emits no spurious downstream artifact.

Three contributions for the literature: a 6-way outcome
taxonomy that splits abstention from fabrication; a within-model
zero-shot vs. constrained vs. RAG ablation that isolates the
prompting-mode contribution to hallucination; and an
abbreviation-stress evaluation mode (D4) that strips the lexical
signal string-matching baselines depend on, surfacing genuine
semantic-mapping behavior. All three generalize without
architectural change to LOINC and RxNorm in subsequent work, and
the compliance-by-construction (MIMIC-trained, Synthea-evaluated)
evaluation pattern is reproducible without PhysioNet credentialing.
