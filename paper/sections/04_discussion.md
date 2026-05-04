# Discussion

> **Status:** Scaffold v0 (2026-05-02). Discussion structure follows
> NEJM AI convention: principal findings → comparison with prior
> work → mechanistic interpretation → clinical implications →
> limitations → future directions. Each subsection notes what
> evidence from §Results it should ground itself in.

---

## Principal findings

[TBD — populate from headline numbers in §Results once the run
completes. The paragraph should restate, in plain prose, the
distribution of outcomes across the five buckets under
D4\_abbreviated, and quantify how that distribution shifts between
zero-shot and constrained prompting within the same model. Length:
one paragraph.]

The three findings we expect to surface, in order of importance:

1. **Hallucination is not a fixed model property; it is a
   prompting-mode property.** [TBD: specify the magnitude of shift
   in hallucination rate when the same LLM moves from zero-shot to
   constrained mode.]
2. **The trained classifier head establishes a clean comparison
   floor on the closed vocabulary.** [TBD: report the gap between
   the classifier's per-mode accuracy and the best zero-shot LLM,
   under D4\_abbreviated specifically.]
3. **Cost-per-correct-prediction reveals deployment trade-offs that
   accuracy alone obscures.** [TBD: identify which model-mode
   combination minimizes cost per correctly-coded encounter at a
   fixed accuracy floor.]

## Comparison with prior work

The closest direct precedent for this work is Soroush et al.\
(2024)\autocite{Soroush2024}, who benchmarked four LLMs on direct
ICD-9, ICD-10, and CPT code querying and reported a substantial
fraction of errors as non-existent codes. Our framework differs in
three ways: (i) we make hallucination an explicit, mutually-exclusive
outcome bucket rather than counting it as ordinary error; (ii) we
isolate the contribution of constrained-vs-unconstrained prompting
within the same model on the same input; (iii) we deliberately
degrade the input via D4\_abbreviated to strip the lexical signal
that string-matching baselines depend on, surfacing the conditions
under which semantic retrieval is genuinely required.

The agentic and neuro-symbolic alternatives in the recent
literature — including Code Like Humans\autocite{Motzfeldt2025} and
Hybrid-Code v2\autocite{HybridCode2025} — report stronger
hallucination control through architectural constraints. Our results
are complementary rather than competitive: we measure the
unconstrained behavior those systems are designed to fix, providing
the empirical baseline against which their constraint-induced gains
should be measured.

[TBD: 1-2 paragraphs comparing our specific outcome distributions to
the most directly comparable precedent numbers, once results are in.
The pattern to expect is that our zero-shot LLM hallucination rates
fall within Hybrid-Code v2's reported neural-baseline band of 6-18%.]

## Why the abbreviation-stress mode reveals what other modes don't

[TBD: paragraph interpreting the per-mode result pattern. The
information-theoretic argument is that D1\_full and D2\_no\_code
inputs contain enough lexical signal that even a string-matching
baseline can recover the correct code in a substantial fraction of
cases, masking whether the LLM is doing semantic work or merely
lexical lookup. D3\_text\_only retains the canonical clinical name.
D4\_abbreviated strips both the structured code AND the canonical
display, leaving only abbreviated clinical jargon (T2DM, HTN, CKD)
that requires actual semantic understanding to map back to ICD-10-CM.
The headline question — does a frontier LLM perform genuine semantic
normalization? — is answerable only under D4 conditions.]

## Implications for clinical deployment

The 6-way outcome taxonomy reframes the deployment question. Standard
top-1-accuracy framing treats every wrong answer equally, but the
deployment cost of a hallucinated code (downstream systems silently
mishandle a non-existent code; the error propagates into billing,
research databases, and quality-reporting pipelines) is qualitatively
different from the cost of a near-miss within the correct ICD chapter
(which a human reviewer would recognize and correct in seconds). A
deployment pipeline that treats hallucinations as a separate,
flag-and-route-for-review category can extract substantially more
value from an LLM-based coder than a pipeline that treats top-1
accuracy as the only metric.

[TBD: paragraph on what hallucination rates within our reported range
mean for human-in-the-loop QA economics. Reference to break-even
analysis in §Cost economics.]

The compliance-by-construction framing of our evaluation pipeline —
training trained models on credentialed MIMIC; evaluating LLMs on
freely-redistributable Synthea — is itself a deployment-relevant
contribution. PhysioNet's responsible-LLM-use policy
(2025)\autocite{PhysioNet2025} prohibits sending credentialed data
through commercial LLM APIs, which means clinical deployments of LLM
coders necessarily face the same train/evaluate separation. Our
benchmark demonstrates that this separation is methodologically
sound: trained-model evaluation on Synthea provides a stronger
out-of-distribution test than in-distribution MIMIC numbers would,
and the LLM evaluation matrix is fully reproducible by any researcher
without credentialed access.

## Limitations

We surface five limitations explicitly:

1. **Top-50 vocabulary cap on the trained classifier.** The fine-tuned
   PubMedBERT head cannot predict any code outside its 50-element
   training vocabulary; LLMs in the comparison arm have no such
   restriction (with corresponding hallucination cost). This is a
   fair asymmetry — the LLMs are constrained only when explicitly
   given a candidate menu in \emph{constrained} mode — but it means
   trained-model recall on long-tail codes is mechanically zero.
2. **Single random seed for v1.** Reported training-run numbers come
   from a single PubMedBERT seed. A proper sensitivity analysis (5+
   seeds with confidence intervals) is on the v2 roadmap.
3. **D1/D2 sequence-length truncation.** Setting
   \texttt{max\_seq\_length=128} for hardware feasibility truncates
   D1\_full and D2\_no\_code inputs that exceed 128 tokens. The
   trained model may underperform on D1/D2 relative to a longer-context
   training run. The headline experiment (D4\_abbreviated, ~5-30
   tokens) is unaffected.
4. **MIMIC-to-Synthea distribution shift.** Trained models are fit on
   MIMIC-IV and evaluated on Synthea, which differs systematically in
   clinical-text style and code distribution. We report this as an
   intentional out-of-distribution test rather than a confound, but
   acknowledge that in-distribution MIMIC test-set numbers would be
   higher.
5. **English ICD-10-CM only.** The benchmark does not address
   non-English clinical-coding settings (where ICPC-2 or local
   variants apply) or alternative coding systems (LOINC, RxNorm,
   SNOMED CT). Extension to LOINC and RxNorm via the same five-way
   taxonomy is planned for v2.

## Future directions

Three extensions follow directly from this work:

- **v2 vocabulary expansion to LOINC and RxNorm.** The 5-way outcome
  taxonomy generalizes naturally; the eval runner and degradation
  pipeline already support arbitrary FHIR resource types.
- **Parameter-efficient fine-tuning of larger biomedical encoders.**
  LoRA\autocite{Hu2022} or QLoRA\autocite{Dettmers2023} would close
  the gap between our 110M-parameter PubMedBERT-base classifier and
  larger biomedical models (BiomedLM, GatorTron, BioMistral) without
  exceeding our hardware constraints. Provides a stronger
  trained-model baseline and a "fine-tuned LLM" rung between
  zero-shot frontier models and a fully-trained classifier head.
- **Rationale evaluation.** Following Li et al.\
  2025\autocite{Li2025}, asking each LLM for a per-prediction
  rationale alongside its code allows a faithfulness/plausibility
  evaluation that complements the outcome-bucket framework. Useful
  for identifying which hallucinations are accompanied by confident
  but spurious justification (clinically more concerning) versus
  those flagged by the model itself as low-confidence.
