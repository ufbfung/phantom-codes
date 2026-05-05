# Conclusion

Across a 24-LLM × 4-degradation-mode evaluation matrix on n=125
Synthea-generated FHIR Conditions, fabrication of non-existent
ICD-10-CM codes is rare in modern frontier LLMs (median per-cell
hallucination = 0%; high tail driven by two Gemini Flash zero-shot
configurations). Constrained prompting eliminates fabrication
entirely for every Anthropic and OpenAI model in the matrix that
shows any nonzero zero-shot hallucination, confirming hallucination
as a controllable property of the prompting setup rather than a
fixed model limitation. Abstention, separated from fabrication via
the 6-way outcome taxonomy, surfaces as the more consequential
failure mode for some models — notably Gemini 2.5 Pro at 88%
empty-prediction rate on zero-shot D4 — and is methodologically
distinct because it emits no spurious downstream artifact. The
6-way taxonomy, the within-model zero-shot vs.\ constrained vs.\ RAG
ablation, and the abbreviation-stress evaluation mode generalize
without architectural change to LOINC and RxNorm in subsequent
work, and the compliance-by-construction (MIMIC-trained,
Synthea-evaluated) pattern is reproducible without PhysioNet
credentialing.
