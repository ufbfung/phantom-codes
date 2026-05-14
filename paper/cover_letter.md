Dear Dr. Elkin,

I'm submitting "Phantom Codes: Hallucination, Accuracy, and Cost in
LLM-Based Medical Concept Normalization" for consideration in *IEEE
Journal of Biomedical and Health Informatics*.

The paper asks how well frontier LLMs assign ICD-10-CM codes to
clinical text, and what each correct code costs. We tested eight
models — Claude Opus 4.7, Sonnet 4.6, and Haiku 4.5; GPT-5.5 and
GPT-4o-mini; Gemini 2.5 Pro, 2.5 Flash, and 3 Flash Preview — in
three prompting modes (zero-shot, constrained to a candidate list,
and retrieval-augmented), for 24 experiments.

The methodology has two distinguishing features. We separate
hallucinations (predicting codes that don't exist) from abstentions
(refusing to answer), because the consequences differ: hallucinated
codes look plausible and slip through into the data, while
abstentions are visible and reviewable. And we test robustness on
Synthea-generated records by stripping the ground-truth codes and
degrading the text — one mode replaces canonical disease names with
abbreviations like T2DM, HTN, and CKD-3 — then check whether each
model can recover the correct code.

Two findings stood out. Grounding (constrained prompting or
retrieval augmentation) eliminated hallucinations in every
Anthropic and OpenAI model. And cost per correct code ranged from
\$0.0003 (GPT-4o-mini, constrained) to \$0.0133 (Claude Opus 4.7),
so the deployment choice is a small grounded model, not the largest
frontier one.

The benchmark uses Synthea data and the full pipeline is released,
so the results are reproducible end-to-end.

The manuscript is original, not copyrighted, published, submitted,
or accepted elsewhere, and no preliminary conference version exists.
This is my first submission to J-BHI. I am an unaffiliated
independent researcher with no competing interests and no external
funding. Claude was used as a coding and drafting assistant, also
disclosed in the Acknowledgments.

Sincerely,

Brian K. Fung, PharmD, MPH
brian@briankfung.com
