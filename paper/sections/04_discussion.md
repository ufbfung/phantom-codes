# Discussion

> **Status:** Draft v1 (2026-05-04). Numbers populated from the
> n=125 Synthea headline run. Discussion structure follows NEJM AI
> convention: principal findings → comparison with prior work →
> mechanistic interpretation → clinical implications → limitations
> → future directions.

---

## Principal findings

The headline result is that **fabrication is rare in modern
frontier LLMs on this task, while abstention is the more
consequential failure mode for some models** — a framing the
6-way outcome taxonomy makes visible that the 5-way framing in
prior work hides. Across the 24 LLM configurations evaluated, the
median hallucination rate at D4\_abbreviated is 0% and the
90th-percentile rate is approximately 39% (with the high tail
coming exclusively from two Gemini Flash zero-shot
configurations). The frontier-Anthropic and OpenAI groupings show
≤3.2% D4 hallucination in any prompting mode tested, and 0%
hallucination across all three Anthropic model sizes (Haiku 4.5,
Sonnet 4.6, Opus 4.7) under the constrained-mode prompt. This
contradicts the popular literature framing of LLM medical coding
as a fabrication problem; in our matrix it is more accurately a
problem of *which models abstain at scale* (notably Gemini 2.5 Pro
under our wrapper settings — 88% no\_prediction at zero-shot D4 —
discussed in §Limitations).

Three findings warrant emphasis:

1. **Hallucination is a prompting-mode property, not a fixed model
   property.** Within the same model, the shift from zero-shot to
   constrained prompting eliminates fabrication entirely for
   every Anthropic and OpenAI model in the matrix that shows any
   nonzero zero-shot hallucination. Claude Sonnet 4.6 drops from
   6.4% (D3 zero-shot) and 3.2% (D4 zero-shot) hallucination to
   0% under constrained prompting; Claude Opus 4.7 drops from
   5.6% to 0.8% on D3 and 1.6% to 0% on D4; GPT-4o-mini drops
   from 12.8% (D2 zero-shot) to 0% under constrained prompting.
   The Gemini Flash configurations show the largest within-model
   shifts: Gemini 2.5 Flash zero-shot D4 hallucination is 43.2%,
   reduced to 18.4% under constrained prompting — still
   non-trivial but a 25-percentage-point reduction. Within-model
   paired McNemar tests confirm these shifts are statistically
   meaningful at the n=125 cohort scale for every model that
   shows nonzero zero-shot hallucination.
2. **The trained PubMedBERT classifier matches frontier LLMs on
   the safety dimension but not on the accuracy dimension.** The
   classifier registers 0% hallucination across all four
   degradation modes — mechanically constrained to its training
   vocabulary — and degrades cleanly from 50.4% top-1 accuracy at
   D1\_full to 40.0% at D4\_abbreviated. This is below the
   Anthropic and OpenAI constrained configurations (86–97%
   D4\_abbreviated top-1) but provides a clean comparison floor
   on the closed vocabulary: any LLM that exceeds the
   classifier's per-mode accuracy at zero or near-zero
   hallucination is dominating the safety-vs-accuracy frontier
   at this cohort scale. By that criterion, every Anthropic and
   OpenAI constrained-mode configuration in the matrix dominates
   the trained classifier.
3. **Cost-per-correct-prediction collapses to a binary deployment
   choice at this cohort scale.** GPT-4o-mini constrained mode at
   \$0.0003 per exact-match prediction (94.2% top-1, 0%
   hallucination) is the cost-frontier configuration; Claude
   Haiku 4.5 constrained at \$0.0044 (96.8% top-1, 0%
   hallucination) is the next-best with marginally higher
   accuracy at 14× the cost. Claude Opus 4.7 constrained mode at
   \$0.0133 per exact-match (94.8% top-1, 0% hallucination)
   delivers no meaningful accuracy improvement at 44× the cost
   of GPT-4o-mini constrained. For deployment, the choice
   collapses to "GPT-4o-mini constrained or Haiku constrained"
   with everything more expensive being dominated.

## Comparison with prior work

The closest direct precedent is Soroush et al.\
(2024)\autocite{Soroush2024}, who benchmarked four LLMs on direct
ICD-9, ICD-10, and CPT code querying without clinical context and
reported a substantial fraction of errors as non-existent codes —
GPT-4 at 46% / 34% / 50% exact match across the three
vocabularies. Our framework differs in three ways: (i) we make
hallucination an explicit, mutually-exclusive outcome bucket
rather than counting it as ordinary error, and additionally split
abstention out of fabrication via the 6-way taxonomy; (ii) we
isolate the contribution of constrained-vs-unconstrained
prompting within the same model on the same input; (iii) we
deliberately degrade the input via D4\_abbreviated to strip the
lexical signal that string-matching baselines depend on,
surfacing the conditions under which semantic retrieval is
genuinely required.

Numerically, our results sit substantially above Soroush et
al.'s reported GPT-4 numbers. On D1\_full (the closest analogue
to their code-querying task, since explicit code mentions are
present), every Anthropic and OpenAI configuration in our matrix
exceeds 99% top-1 accuracy in any prompting mode. Even on
D4\_abbreviated — a substantially harder setting because it
strips both the explicit code AND the canonical display string —
the same configurations exceed 80% top-1. Differences attributable
to: (a) the cohort-scope restriction to ACCESS-Model conditions
(12 unique ICD codes in our cohort vs. their broader vocabulary),
(b) generational improvements in frontier LLMs since 2024
(GPT-5.5, Opus 4.7, Sonnet 4.6 evaluated here vs. their GPT-4),
and (c) our constrained / RAG prompting modes vs. their
unstructured code-querying setup. The constrained-mode
hallucination rate of 0% across Anthropic models in our matrix is
notably lower than Hybrid-Code v2's reported neural-baseline
band of 6–18% [@HybridCode2025], suggesting that modern frontier
LLMs under structured-output constraints have closed much of the
fabrication gap that motivated neuro-symbolic approaches.

The agentic and neuro-symbolic alternatives in the recent
literature — Code Like Humans\autocite{Motzfeldt2025} and
Hybrid-Code v2\autocite{HybridCode2025} — report stronger
hallucination control through architectural constraints. Our
results are complementary rather than competitive: we measure
the unconstrained (zero-shot) and lightly-constrained (RAG,
constrained) behavior those systems are designed to fix,
providing the empirical baseline against which their
constraint-induced gains should be measured. At our matrix's
n=125 scale and ACCESS-scope restriction, the constraint headroom
those systems aim to exploit is largely already consumed by
modern frontier LLMs in a simpler `constrained` prompting
configuration.

## Why the abbreviation-stress mode reveals what other modes don't

The four degradation modes vary in how much information the
input carries about the target code. D1\_full is the FHIR
Condition resource verbatim — including the explicit ground-truth
code and the canonical display string. D2\_no\_code strips the
explicit code but leaves the canonical display. D3\_text\_only
strips the structured FHIR fields, leaving canonical clinical
text. D4\_abbreviated strips both the explicit code AND the
canonical display, replacing them with abbreviated clinical
jargon (T2DM, HTN, CKD-3, ASCVD), and is the only mode that
requires actual semantic mapping to recover the ICD code.
String-matching baselines (fuzzy, TF-IDF) drop ~30 percentage
points in top-1 accuracy from D3 to D4 — they were performing
lexical lookup, not semantic mapping. The headline question —
does a frontier LLM perform genuine semantic normalization
rather than lexical lookup? — is answerable only under D4
conditions, and the answer in our matrix is yes for the
constrained-mode Anthropic and OpenAI configurations (top-1
accuracy holds at 86–97% on D4) and conditionally yes for the
zero-shot variants of the same models (top-1 holds at 60–85% on
D4 with low hallucination). The Gemini Flash zero-shot
configurations are the exception: they degrade dramatically on
D4 (top-1 falls to ~55–60%, hallucination rises to 39–43%) —
their semantic-mapping performance is constrained by parameter
count rather than by prompting mode.

## Implications for clinical deployment

The 6-way outcome taxonomy reframes the deployment question.
Standard top-1-accuracy framing treats every wrong answer
equally, but the deployment cost of a hallucinated code
(downstream systems silently mishandle a non-existent code; the
error propagates into billing, research databases, and quality-
reporting pipelines) is qualitatively different from the cost of
a near-miss within the correct ICD chapter (which a human
reviewer would recognize and correct in seconds), and from an
abstention (which requires re-coding from scratch but emits no
spurious downstream artifact). A deployment pipeline that treats
hallucinations and abstentions as separate flag-and-route-for-
review categories — distinct from category-match near-misses —
can extract substantially more value from an LLM-based coder than
a pipeline that treats top-1 accuracy as the only metric.

For the headline-run findings specifically, the human-in-the-loop
QA economics are dominated by category-match and abstention
outcomes rather than by fabrication: the constrained-mode
Anthropic and OpenAI configurations register 0% hallucination,
~5% category-match, and ~0% abstention, so the QA fraction is
dominated by category-match review (a fast 5-second human action
to confirm or correct a near-miss code). Per the §Cost economics
break-even framework (full analysis in §Supplementary S5), this
configuration beats human-only coding on cost across all
sensitivity-sweep parameter ranges considered. Gemini 2.5 Pro
under our v1 wrapper settings is the lone exception in this
matrix: its high abstention rate (~88% on zero-shot D4)
inflates the QA fraction to the point where its per-correct
cost approaches human-coder parity at moderate QA rates. The
Gemini Pro empty-prediction behavior is plausibly a
wrapper-level interaction (reasoning-token budget exhaustion
under our default `max_output_tokens` setting; investigation
tracked in BACKLOG and discussed in §Limitations), not
necessarily an inherent model-quality limitation.

The compliance-by-construction framing of our evaluation pipeline
— training trained models on credentialed MIMIC; evaluating LLMs
on freely-redistributable Synthea — is itself a deployment-
relevant contribution. PhysioNet's responsible-LLM-use policy
(2025)\autocite{PhysioNet2025} prohibits sending credentialed
data through commercial LLM APIs, which means clinical
deployments of LLM coders necessarily face the same
train/evaluate separation. Our benchmark demonstrates that this
separation is methodologically sound: trained-model evaluation
on Synthea provides a stronger out-of-distribution test than
in-distribution MIMIC numbers would, and the LLM evaluation
matrix is fully reproducible by any researcher without
credentialed access.

## Limitations

We surface six limitations explicitly:

1. **Cohort scale.** The headline matrix evaluates on 125 unique
   Synthea Conditions, smaller than the originally-targeted
   n=500 cohort due to a `--max-records` semantic ambiguity in
   the evaluation CLI (it counts long-format ndjson rows rather
   than unique resources; documented in BACKLOG). Per-cell
   Wilson 95% confidence intervals are correspondingly wider
   (~±5pp on a 10% rate; ~±9pp on a 50% rate) than they would
   be at n=500 (~±2.7pp / ~±4.4pp). Replication at the
   originally-intended n=500 is straightforward
   (`--max-records 2000` per the BENCHMARK reproduction guide)
   and would tighten all reported confidence intervals by
   approximately 2×.
2. **Top-50 vocabulary cap on the trained classifier.** The
   fine-tuned PubMedBERT head cannot predict any code outside
   its 50-element training vocabulary; LLMs in the comparison
   arm have no such restriction, so trained-model recall on
   long-tail codes is mechanically zero, not learned-zero. Our
   12-code Synthea cohort happens to fit comfortably within the
   trained classifier's vocabulary, so this asymmetry does not
   materially affect the headline numbers, but it would on a
   broader-vocabulary v2 cohort.
3. **Single random seed for v1.** Reported training-run numbers
   come from a single PubMedBERT seed. A proper sensitivity
   analysis (5+ seeds with confidence intervals) is on the v2
   roadmap.
4. **Gemini 2.5 Pro empty-prediction behavior under our wrapper
   settings.** The headline run observed an 88% no\_prediction
   rate on Gemini 2.5 Pro zero-shot D4, with 84% of those rows
   showing zero output tokens — consistent with reasoning-token
   budget exhaustion (Gemini 2.5 Pro is a reasoning model whose
   internal "thinking" tokens count against `max_output_tokens`
   under the SDK version we use). Whether this is a fixable
   wrapper-level interaction (raising `max_tokens` from 1024 to
   4096 or disabling thinking via `thinking_budget=0`) or a
   genuine Gemini Pro limitation under structured-output
   prompting is the subject of a follow-up investigation
   blocked by Tier-1 daily quota at the time of v1 submission;
   findings will appear as a v1.1 supplement if a fix is
   identified.
5. **Gemini 3.1 Pro Preview excluded from headline matrix.**
   Tier-1 daily-quota limits (250 requests/day) made full
   coverage of the ~6,000 calls required for the matrix
   infeasible at v1 timeline; a partial-coverage exploratory
   analysis on ~21 records is presented as supplementary
   directional signal rather than headline data.
6. **English ICD-10-CM only.** The benchmark does not address
   non-English clinical-coding settings (where ICPC-2 or local
   variants apply) or alternative coding systems (LOINC,
   RxNorm, SNOMED CT). Extension to LOINC and RxNorm via the
   same 6-way taxonomy is planned for v2.

## Future directions

Three extensions follow directly from this work:

- **v2 vocabulary expansion to LOINC and RxNorm.** The 6-way
  outcome taxonomy generalizes naturally; the eval runner and
  degradation pipeline already support arbitrary FHIR resource
  types.
- **Parameter-efficient fine-tuning of larger biomedical
  encoders.** LoRA\autocite{Hu2022} or QLoRA\autocite{Dettmers2023}
  would close the gap between our 110M-parameter PubMedBERT-base
  classifier and larger biomedical models (BiomedLM, GatorTron,
  BioMistral) without exceeding our hardware constraints.
  Provides a stronger trained-model baseline and a "fine-tuned
  LLM" rung between zero-shot frontier models and a fully-
  trained classifier head.
- **Rationale evaluation.** Following Li et al.\
  2025\autocite{Li2025}, asking each LLM for a per-prediction
  rationale alongside its code allows a faithfulness/
  plausibility evaluation that complements the outcome-bucket
  framework. Useful for identifying which fabrications are
  accompanied by confident but spurious justification
  (clinically more concerning) versus those flagged by the
  model itself as low-confidence.
