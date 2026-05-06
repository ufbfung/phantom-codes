Dear Dr. Bakken and Editorial Team,

I am pleased to submit the enclosed manuscript, "Phantom Codes:
Hallucination, Accuracy, and Cost in Frontier-LLM Medical Concept
Normalization," for consideration as a Research and Applications
article in JAMIA.

The paper contributes two findings that we believe are
deployment-relevant for clinical informatics groups currently
evaluating LLMs for medical-coding workflows. First, across a
24-LLM-configuration evaluation matrix on 125 Synthea-generated
FHIR Conditions, every fabrication of a non-existent ICD-10-CM
code we observed occurred under zero-shot prompting; constrained
prompting eliminated it entirely for every Anthropic and OpenAI
model evaluated. This adds independent evidence that zero-shot
LLM coding is inferior to grounded prompting for production use,
and that the fabrication-free property is recoverable through
prompting structure rather than model choice. Second, when configurations are scored on
cost-per-correct-prediction rather than accuracy alone, the
deployment frontier collapses to two grounded sub-flagship models
(GPT-4o-mini constrained at \$0.0003 per correct, Claude Haiku 4.5
constrained at \$0.0044). The largest frontier model tested
(Claude Opus 4.7 constrained) does not appear on that frontier
despite competitive accuracy, because cost-per-correct varies by
roughly 270× across the matrix while top-1 accuracy varies by far
less. Reporting accuracy in isolation systematically misranks
deployment options.

The entire evaluation runs against freely-redistributable Synthea
data, so the benchmark is reproducible by any reader without a
data-use agreement.

The manuscript has not been published elsewhere and is not under
consideration by any other journal. The author is unaffiliated and
declares no competing interests. The work received no external
funding; LLM API costs were self-funded. The author used Anthropic's
Claude as a coding, drafting, and figure-design assistant; this is
disclosed in the manuscript's Acknowledgments per the COPE position
statement on Authorship and AI Tools.

Sincerely,

Brian K. Fung
brian@briankfung.com
