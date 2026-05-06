# BACKGROUND AND SIGNIFICANCE

Clinical concept normalization is the task of mapping unstructured
mentions of diagnoses, medications, labs, and procedures to
identifiers in controlled vocabularies such as ICD-10-CM, LOINC,
and RxNorm. It sits at the foundation of healthcare data
interoperability. Errors propagate downstream: a misassigned
diagnosis code can shift a hospital's case-mix index, mis-stratify
a research cohort, or quietly corrupt the training data of the
next generation of models. For most of the last decade, the state
of the art for automated coding came from supervised neural
classifiers trained on labeled clinical text [@Mullenbach2018;
@Huang2022]. The arrival of general-purpose large language models
has changed the deployment economics: foundation models can in
principle perform code prediction with no task-specific
fine-tuning. The question that motivated this work is not whether
LLMs can predict medical codes. They can. The question is how
they fail when they do.

A key concern with applying LLMs to coded clinical data is that
they can fabricate codes that do not exist in the target
vocabulary. Unlike a classifier trained on a closed label set, a
generative model has no mechanical guarantee that its output is
even a member of the controlled vocabulary it was asked to
produce. This is the narrowest, mechanically checkable instance
of the broader phenomenon of medical hallucination [@Kim2025;
@Ji2023]. Empirical evidence is sparse but cautionary. Soroush
et al. reported that GPT-4 achieved only 46% exact match on ICD-9
and 34% on ICD-10, with a substantial fraction of errors being
non-existent codes rather than real-but-wrong assignments
[@Soroush2024]. Lee and Lindsey found that LLMs systematically
struggle to distinguish real codes from plausibly-formatted fakes
[@Lee2024]. The field's response has largely been to constrain
the LLM rather than to characterize its unconstrained behavior,
through retrieval-grounded pipelines [@DasBaksi2024;
@Sarvari2025], agentic walkers [@Motzfeldt2025], neuro-symbolic
verifiers [@HybridCode2025], and broader benchmark expansions
[@Almeida2025; @Singh2025; @Li2025; @Gershon2025]. Across this
body of work, hallucination is rarely treated as a first-class
outcome, the within-model contrast between zero-shot and
constrained prompting is rarely controlled, and degraded-input
robustness is not systematically tested.

We present **Phantom Codes**, a reproducible benchmark whose
primary contributions are methodological rather than
architectural. We introduce a six-way outcome taxonomy in which
hallucination and abstention are explicit, mutually exclusive
buckets. Splitting `no_prediction` (the model returned nothing)
from `hallucination` (the model returned a non-existent code)
matters for deployment safety, since an abstaining model is
preferable to a confidently-wrong one. We then run a within-model
ablation across three prompting modes (zero-shot, constrained,
and retrieval-augmented) that holds the model and input fixed and
varies only whether a candidate list is provided, isolating
"wandering off the menu" from genuine ignorance. We also include
an abbreviation-stress mode (D4) that replaces canonical display
strings with clinical jargon (T2DM, HTN, CKD-3) while preserving
the underlying diagnosis, exposing the condition under which
string-matching baselines collapse and any remaining accuracy
must come from genuine semantic mapping. The headline evaluation
matrix runs entirely against Synthea-generated FHIR Bundles
[@Walonoski2018], a freely-redistributable synthetic patient
dataset that gives full reproducibility from a pinned Synthea
version and seed.
