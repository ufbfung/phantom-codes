# Conclusions

> **Status:** Scaffold v0 (2026-05-02). Two paragraphs, ~150 words
> total — NEJM AI conclusions section convention. Rewrite once
> §Results numbers are settled.

---

[TBD — paragraph one: state the principal empirical finding in
plain prose, naming the headline number (hallucination rate under
D4\_abbreviated, contrasted between zero-shot and constrained modes
within the same model). End with the one-line takeaway for clinical
deployment: when LLMs are deployed for clinical concept normalization,
hallucination is a controllable property of the prompting setup
rather than a fixed model limitation, and downstream pipelines should
treat hallucinated codes as a flag-and-route-for-review category
distinct from hierarchical near-misses.]

[TBD — paragraph two: restate the methodological contributions
(6-way outcome taxonomy with hallucination and abstention as explicit, distinct buckets;
within-model zero-shot vs.\ constrained ablation; D4\_abbreviated as
a stress test that strips the lexical signal string-matching
baselines depend on; compliance-by-construction train-on-MIMIC /
evaluate-on-Synthea separation). End by noting that the framework
extends without architectural change to LOINC and RxNorm in
subsequent work.]
