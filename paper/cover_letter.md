Dear Dr. Elkin,

I'm submitting "Phantom Codes: Hallucination, Accuracy, and Cost in
LLM-Based Medical Concept Normalization" for consideration as a
regular research article in the *IEEE Journal of Biomedical and
Health Informatics*.

The paper contributes a reproducible evaluation framework for
LLM-based clinical concept normalization, demonstrated on the most
recent frontier-model generation. The framework treats hallucination
as a measurable outcome rather than a caveat. A six-way outcome
taxonomy splits fabrication of non-existent codes from abstention as
separate failure modes. They have different downstream consequences:
a fabricated code silently corrupts a harmonized dataset; an
abstention surfaces a gap a reviewer can fill. The taxonomy applies
to any closed-vocabulary grounding task, including LOINC and RxNorm.

Three other pieces support that core. A within-model ablation across
zero-shot, constrained, and retrieval-augmented prompting separates
models that wander off a candidate list from those that lack the
knowledge to begin with. A D4 abbreviation-stress mode replaces
canonical display strings with clinical jargon (T2DM, HTN, CKD-3) to
expose models that rely on lexical lookup. And per-prediction cost
is reported as cost-per-correct (USD per exact match), so accuracy
and economics are visible on the same axis.

The matrix covers 24 (model, prompting-mode) configurations across
Claude Opus 4.7, Sonnet 4.6, Haiku 4.5; GPT-5.5, GPT-4o-mini; and
Gemini 2.5 Pro, 2.5 Flash, 3 Flash Preview. Constrained prompting
eliminated fabrication for every Anthropic and OpenAI model.
Cost-per-correct varied by 270x. The deployment frontier is
GPT-4o-mini constrained (\$0.0003 per correct) and Claude Haiku 4.5
constrained (\$0.0044); both dominate Opus 4.7 at \$0.0133. Inputs
are Synthea-generated FHIR Bundles, so the benchmark is reproducible
without a data-use agreement. The full evaluation pipeline,
prompts, and per-prediction CSV are released under the MIT License.

The work fits J-BHI's scope on two fronts: clinical informatics
methodology (an evaluation framework that generalizes across
controlled vocabularies) and AI for health (a head-to-head
comparison of frontier LLMs against trained baselines, with
deployment economics).

The manuscript is original. It has not been copyrighted, published,
submitted, or accepted for publication elsewhere, and no preliminary
version has appeared as a conference paper. This is a first
submission to J-BHI, not a revision or resubmission. I am an
unaffiliated independent researcher with no competing interests and
no external funding. I used Anthropic's Claude as a coding and
drafting assistant; this is disclosed in the Acknowledgments per
COPE guidance.

Sincerely,

Brian K. Fung, PharmD, MPH
brian@briankfung.com
