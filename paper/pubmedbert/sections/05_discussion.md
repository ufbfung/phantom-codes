# Discussion

## Compliance-by-construction is reachable on consumer hardware

The pipeline described here demonstrates that PhysioNet-compliant
fine-tuning of a research-grade biomedical classifier is reachable
on a single consumer Apple Silicon laptop without cloud GPU
access, institutional compute budget, or BAA negotiation. The
~15-hour wall-clock per training run is slow relative to A100
cloud GPUs (~3–5× slower), but the cost is one-time per
checkpoint and the data-residency story is dramatically simpler.
For credentialed researchers who do not have institutional cloud
infrastructure pre-approved for credentialed clinical data, the
local-MPS path is the path of least friction.

The same code path supports CUDA backends; researchers with
appropriate cloud-compute authorization can swap
`torch.device("mps")` for `torch.device("cuda")` and gain the
wall-clock speedup. The compliance posture in that case becomes a
function of the cloud environment's BAA and data-residency
review, not the training code itself.

## In-distribution vs. out-of-distribution gap

The 99%-validation / 40–50%-Synthea gap is the most
deployment-relevant single finding from this work. It reflects a
common pattern in biomedical NLP that in-distribution validation
numbers can dramatically overstate real-world performance when
the deployment distribution differs in surface form or cohort
composition from the training distribution. Reporting *only*
in-distribution validation numbers is therefore misleading; we
report both, and we argue that any clinical deployment claim made
on the basis of in-distribution validation accuracy alone should
be regarded with skepticism.

Synthea is not the only out-of-distribution test that could be
run. Other reasonable choices include: cross-institution MIMIC
holdouts where available, pediatric or non-English clinical
corpora, or institution-specific notes from the deploying
hospital. Any such test would surface a gap of similar character;
our headline number is the one that the companion paper
[@FungPhantomCodes2026] uses to anchor its non-LLM baseline.

## Limitations

- **Top-50 vocabulary cap by construction.** The classifier
  cannot predict any code outside the 50 most-frequent training
  codes. This is a deliberate trade-off (signal density over
  coverage), but it bounds applicability to deployments where
  long-tail codes are not central. A v2 variant with a larger
  head or with hierarchical decomposition is straightforward to
  build and is on the roadmap.
- **Single random seed for v1.** Reported numbers come from a
  single seed. A proper sensitivity analysis (5+ seeds with
  confidence intervals) is on the v2 roadmap; for v1 the
  single-seed result is reported alongside this explicit
  acknowledgment.
- **D1/D2 sequence-length truncation.** Setting
  `max_seq_length=128` for hardware feasibility truncates the
  longest D1\_full and D2\_no\_code FHIR-JSON inputs. The
  D3\_text\_only and D4\_abbreviated headline inputs (10–50
  tokens) are unaffected; the truncation matters most for the
  D1/D2 modes that include serialized JSON payloads.
- **MIMIC-IV training cohort scope.** ACCESS-scope conditions
  are weighted toward cardiometabolic disease; performance on
  conditions outside that scope is not directly addressable from
  this trained model.
- **No comparison against larger biomedical encoders.** BiomedLM,
  GatorTron, BioMistral, and Med-PaLM (via parameter-efficient
  fine-tuning) are plausible competitors; we deferred those
  comparisons to a v2 ablation arm rather than crowd v1.

## Future directions

- **Parameter-efficient fine-tuning** of larger biomedical encoders
  via LoRA [@Hu2022] or QLoRA [@Dettmers2023] would close the gap
  between this 110M-parameter classifier and larger biomedical
  models without exceeding our hardware constraints.
- **Vocabulary expansion to LOINC and RxNorm** for laboratory
  observations and medications, using the same architecture and
  training recipe.
- **Multi-seed sensitivity analysis** with confidence intervals
  reported alongside point estimates.
- **Cross-institution out-of-distribution testing** where
  credentialed cohorts from additional institutions are
  available.
