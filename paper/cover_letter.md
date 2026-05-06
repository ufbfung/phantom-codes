Dear Dr. Bakken and Editorial Team,

I am pleased to submit the enclosed manuscript, "Phantom Codes:
Hallucination in LLM-Based Clinical Concept Normalization," for
consideration as a Research and Applications article in the Journal
of the American Medical Informatics Association.

This work addresses a question that the field has, to date, treated
inconsistently: when large language models are applied to clinical
concept normalization, how do they fail? Existing benchmarks
typically conflate fabrication of non-existent codes with
abstention, and rarely examine whether failure modes are properties
of the model or of the prompting setup. We separate these issues
with three methodological contributions. We introduce a six-way
outcome taxonomy that treats hallucination and abstention as
explicit, mutually exclusive buckets. We run a within-model
ablation across zero-shot, constrained, and retrieval-augmented
prompting that isolates "wandering off the menu" from genuine
ignorance. We also include an abbreviation-stress mode that strips
lexical signal so semantic-mapping behavior can be tested
independently of lexical lookup.

Across a 24-LLM-configuration evaluation matrix on 125 unique
Synthea-generated FHIR Conditions, fabrication ranged from 0% to
5.6% across flagship models and from 0% to 12.8% across
sub-flagship models, and was eliminated entirely under constrained
prompting for every Anthropic and OpenAI model evaluated.
Abstention emerged as the more consequential failure mode for some
models. Most strikingly, Gemini 2.5 Pro abstained on 95% of
zero-shot D4-stress inputs while fabricating nothing.
Cost-per-correct varied by a factor of 270 across the matrix, with
GPT-4o-mini in constrained mode (94.2% top-1 accuracy at \$0.0003
per correct prediction) emerging as the deployment-cost frontier
and Claude Haiku 4.5 in constrained mode (96.8% top-1 accuracy at
\$0.0044 per correct prediction) as the higher-accuracy
alternative. Both dominate the largest frontier model tested
(Claude Opus 4.7 in constrained mode at \$0.0133 per correct
prediction).

We believe JAMIA is the right venue for this work for three
reasons. First, the contribution is methodological rather than
architectural, with direct implications for clinical informatics
practice. Our findings argue for LLM-augmented coding with
terminologist review rather than autonomous LLM coding, and we
provide cost-per-correct guidance to support that workflow.
Second, the entire evaluation runs against freely-redistributable
Synthea data, so the benchmark is reproducible by any reader
without credentialed-data access, a deliberate design choice that
aligns with JAMIA's reproducibility expectations. Third, the
six-way taxonomy and evaluation runner extend without
architectural change to LOINC and RxNorm via the same FHIR
resource framework, providing a foundation for future
clinical-vocabulary work in this venue.

This manuscript has not been published elsewhere and is not under
consideration by any other journal. The author is unaffiliated
with any institution and has no competing interests to disclose.
The work received no external funding; LLM API costs were
self-funded and are itemized in the manuscript. All source code,
configuration files, synthetic test fixtures, and the manuscript
source are publicly available at the project's GitHub repository
under the MIT License, allowing any researcher to reproduce the
LLM-evaluation arm end-to-end.

Should you find the manuscript suitable for review, I am happy to
assist with any clarifications. Thank you for your consideration.

Sincerely,

Brian K. Fung
Independent Researcher
brian@briankfung.com
