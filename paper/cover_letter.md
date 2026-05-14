Dear Editors,

I am pleased to submit the enclosed manuscript, "Phantom Codes:
Hallucination, Accuracy, and Cost in LLM-Based Medical Concept
Normalization," for consideration in the *IEEE Journal of Biomedical
and Health Informatics* Special Issue on "Large Language Models with
Applications in Bioinformatics and Biomedicine, Part II."

The special issue's call identifies hallucination — the production
of plausible but factually incorrect output — as a central barrier
to safe clinical deployment of LLMs. This manuscript contributes a
reproducible evaluation framework that operationalizes hallucination
as an explicit, mechanically-checked outcome, applies that framework
to the most recent frontier-model generation, and reports the
deployment economics that determine pipeline viability.

The primary contribution is methodological. The paper introduces a
**six-way outcome taxonomy** that splits fabrication of non-existent
codes (hallucination) from abstention (no_prediction) as mutually-
exclusive, deployment-distinct failure modes — a distinction that
prior work conflates and that matters for downstream data quality,
since a fabricated code silently corrupts a harmonized dataset while
an abstention surfaces a tractable gap. The taxonomy generalizes
across closed-vocabulary grounding tasks (ICD-10-CM, LOINC, RxNorm,
SNOMED CT). The paper additionally introduces a **D4
abbreviation-stress robustness probe** that strips lexical signal
to isolate genuine semantic-mapping behavior from lexical lookup,
a **within-model ablation across zero-shot, constrained, and
retrieval-augmented prompting** that isolates "wandering off the
menu" from genuine ignorance, and **per-prediction cost as a
first-class outcome** alongside accuracy with cost-per-correct
(USD per exact-match outcome) reported as the deployment-relevant
normalization.

The empirical demonstration spans 24 (model, prompting-mode)
configurations across Claude Opus 4.7, Sonnet 4.6, and Haiku 4.5;
GPT-5.5 and GPT-4o-mini; and Gemini 2.5 Pro, 2.5 Flash, and 3 Flash
Preview — several released within weeks of the evaluation date.
Headline findings: zero fabrication under constrained prompting for
every Anthropic and OpenAI model evaluated; cost-per-correct varies
by a factor of 270× across the matrix while top-1 accuracy varies
by far less; the deployment frontier collapses to two grounded
sub-flagship configurations (GPT-4o-mini constrained at \$0.0003
per correct, Claude Haiku 4.5 constrained at \$0.0044) that
dominate the largest frontier model tested. The full evaluation
runs against freely-redistributable Synthea-generated FHIR Bundles,
so the benchmark is reproducible by any reader without a data-use
agreement; the per-prediction CSV, the FY2026 CMS ICD-10-CM
validator, and the eval runner are released under the MIT License
alongside the manuscript.

The manuscript is entirely original; it has not been copyrighted,
published, submitted, or accepted for publication elsewhere, and no
preliminary version has appeared as a conference abstract or paper.
This is a first submission to *IEEE J-BHI*, not a revision or
resubmission. The author is unaffiliated and declares no competing
interests. The work received no external funding; LLM API costs were
self-funded. The author used Anthropic's Claude as a coding,
drafting, and figure-design assistant; this is disclosed in the
manuscript's Acknowledgments per the COPE position statement on
Authorship and AI Tools.

Sincerely,

Brian K. Fung, PharmD, MPH
brian@briankfung.com
