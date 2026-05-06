# Discussion

## Principal findings

The headline result is that **fabrication is bounded in modern
frontier LLMs and stratifies cleanly by model tier and prompting
mode**, while abstention is the more consequential failure mode
for some models — a framing the 6-way taxonomy makes visible.
Across the 24 LLM configurations, hallucination ranges 0–5.6% for
flagship models (Opus 4.7, GPT-5.5, Gemini 2.5 Pro) and 0–12.8%
for sub-flagship models, with the upper bound from GPT-4o-mini
zero-shot D2. Hallucination is a prompting-mode
property, not a fixed model property: the shift from zero-shot to
constrained prompting eliminates fabrication entirely for every
Anthropic and OpenAI model that shows any nonzero zero-shot
hallucination (within-model paired McNemar tests confirm
significance at n=125). The string-matching and frozen
sentence-transformer baselines (45–69% top-1) sit well below every
constrained-mode LLM on the safety-vs-accuracy frontier, confirming
that constrained frontier LLMs offer real gains over lexical and
embedding-only approaches at this cohort scale. Cost-per-correct
collapses the deployment choice to a small frontier: GPT-4o-mini
constrained at \$0.0003 and Claude Haiku 4.5 constrained at \$0.0044
dominate every more expensive configuration tested.

## Comparison with prior work

The closest direct precedent is Soroush et al.\ (2024)\autocite{Soroush2024},
who benchmarked four LLMs on direct ICD code querying. Our framework
differs in three ways: hallucination is an explicit mutually-
exclusive bucket and abstention is split out via the 6-way taxonomy;
we isolate the within-model zero-shot vs.\ constrained contribution;
and the D4\_abbreviated stress strips lexical signal so semantic-
mapping behavior is testable separately from lexical lookup. Our
matrix sits substantially above Soroush et al.'s GPT-4 numbers
(every Anthropic / OpenAI configuration exceeds 80% top-1 even on
D4), attributable to cohort-scope restriction, generational LLM
improvements, and structured-output prompting. Neuro-symbolic
alternatives [@Motzfeldt2025; @HybridCode2025] report
hallucination control through architectural constraints; our 0%
Anthropic constrained-mode rate is below their reported neural-
baseline band, suggesting modern frontier LLMs under structured-
output constraints have closed much of the fabrication gap those
systems were designed to fix. Full comparison in Supplementary §S2.

## Implications for clinical deployment

The 6-way taxonomy reframes the deployment question. Top-1-accuracy
framing treats every wrong answer equally, but the cost of a
hallucinated code (downstream systems silently mishandle it,
propagating into billing, research, and quality reporting) is
qualitatively different from a near-miss within the correct ICD
chapter (corrected by a reviewer in seconds) and from an abstention
(re-coding from scratch, but emitting no spurious artifact).

The empirically-supported deployment pattern is therefore
**LLM-augmented coding with terminologist or clinical-informaticist
review**, not autonomous LLM coding. Two findings concretely
support this. First, for constrained-mode Anthropic and OpenAI
configurations (0% hallucination, ~5% category-match, ~0%
abstention), the QA fraction is dominated by fast category-match
adjudication — a clinically straightforward task an experienced
coder can resolve in seconds — and beats human-only coding across
all sensitivity-sweep parameter ranges considered (Supplementary
§S5). Second, the top-1 vs top-5 lift table (§3) shows that
trained-classifier and sub-frontier zero-shot LLM workflows
(top-1 in the 45–75% range) benefit substantially from surfacing
top-5 candidates to a reviewer, lifting net accuracy 20–38
percentage points; constrained-mode frontier LLMs already saturate
top-1 (lift ≤6pp) and shift the reviewer's role from candidate
selection to category-match adjudication. Either architecture
keeps a clinical adjudicator in the decision loop where accuracy
matters most.

## Limitations

The v1 headline matrix evaluates n=125 unique Synthea Conditions;
per-cell Wilson 95% CIs on a 10% rate are correspondingly ±5pp, and
v2 replication at n=2000 will tighten intervals by approximately
2×. Gemini 2.5 Pro showed an 88% no\_prediction rate on zero-shot
D4 with most empty rows showing zero output tokens — consistent
with reasoning-token budget exhaustion under our wrapper's
`max_output_tokens=1024` setting; whether this is a fixable
wrapper interaction or a genuine model limitation is the subject
of a follow-up investigation blocked by Tier-1 daily quota at v1
submission. Gemini 3.1 Pro Preview was excluded for the same quota
reason. The benchmark covers English ICD-10-CM only; extension to
LOINC and RxNorm via the same taxonomy is planned for v2.

## Future directions

Two extensions follow directly: v2 vocabulary expansion to LOINC
and RxNorm (the 6-way taxonomy and eval runner already support
arbitrary FHIR resource types); and rationale evaluation following
Li et al.\ \autocite{Li2025}, asking each LLM for a per-prediction
rationale to support faithfulness/plausibility evaluation alongside
the outcome-bucket framework. A v2 cohort with broader vocabulary
and more realistic clinical text (e.g., discharge summaries, ED
notes) is also needed to establish the true effect size of
fabrication risk on production-relevant inputs.
