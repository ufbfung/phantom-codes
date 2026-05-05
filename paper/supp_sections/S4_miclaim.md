# S4 — MI-CLAIM checklist (Minimum Information about Clinical AI Modeling)

This supplement populates the 16-item MI-CLAIM checklist for the
study reported in the main manuscript. NEJM AI strongly recommends
MI-CLAIM completion for AI-method submissions; we provide it as a
supplementary appendix per the journal's reporting-guidance
recommendations.

> **Status:** Skeleton — to be filled in Phase 5 of the paper-
> writing scope. Each of the 16 items below has a placeholder that
> references the relevant section of the main manuscript or this
> supplement; the final version provides a self-contained
> per-item paragraph so readers using the checklist as their
> primary entry point need not cross-reference back to the main
> text for each entry.

## Items

1. **Study purpose** — [TODO: 1-paragraph summary referencing main §0]
2. **Data sources** — [TODO: MIMIC-IV-FHIR for training, Synthea for evaluation; cite §1, §3]
3. **Cohort selection** — [TODO: ACCESS-scope filtering, n=125 evaluation cohort; cite §3.1]
4. **Ground truth** — [TODO: ICD-10-CM codes from Synthea-generated Conditions via curated SNOMED→ICD map; cite §3.1, S3.3]
5. **Models compared** — [TODO: 29 configurations across 5 model families; cite §1.LLM-evaluation-methodology]
6. **Evaluation procedure** — [TODO: 6-way outcome taxonomy, 4 degradation modes; cite §1, §3]
7. **Performance metrics** — [TODO: Wilson 95% CIs, McNemar paired tests; cite §1.statistical-analysis]
8. **Statistical analysis** — [TODO: per-cell N=125, no NHST across the matrix; cite §1.statistical-analysis]
9. **Error analysis** — [TODO: per-bucket failure-mode breakdown via 6-way taxonomy; cite §3, §S2]
10. **Validation strategy** — [TODO: out-of-distribution test (MIMIC-trained, Synthea-evaluated); cite §1, §3]
11. **Deployment considerations** — [TODO: cost-per-correct, deployment-scale projections; cite §2, §S5]
12. **Code availability** — [TODO: GitHub URL + MIT license; cite back-matter]
13. **Data availability** — [TODO: Synthea reproducible from pinned SHA + seed; MIMIC requires PhysioNet credentialing; cite §S3, back-matter]
14. **Software versions** — [TODO: capture from `uv export`; cite §S3.2]
15. **Hardware** — [TODO: M1 MacBook Pro for training, API endpoints for LLM eval; cite §1.hardware]
16. **Limitations** — [TODO: cohort scale, single-seed PubMedBERT, Gemini Pro abstention pattern; cite §4.limitations]
