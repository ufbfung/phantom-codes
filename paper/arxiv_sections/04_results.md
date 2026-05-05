# Results

## Training convergence (in-distribution MIMIC validation)

Validation top-1 accuracy at convergence reached **0.992–0.993**
across three training epochs on the held-out MIMIC-IV validation
split. This high in-distribution value reflects the long-tail
concentration of the ACCESS-scope cohort (the top-50 codes cover
the bulk of training rows; see *Cohort and label space*) combined
with effective fine-tuning of a pre-trained biomedical encoder.
Validation loss decreased monotonically across the early-stopping
window, consistent with healthy fine-tuning rather than
memorization.

Aggregate per-epoch metrics (train loss, validation loss,
validation top-1 accuracy) are persisted to
`models/metrics/train_*.json` after training completes. Per-record
predictions on MIMIC validation are deliberately not persisted
beyond the in-memory loop — both to limit checkpoint size and to
preserve the compliance posture that no MIMIC-derived per-record
data leaves the laptop.

## Out-of-distribution performance (Synthea evaluation cohort)

The same checkpoint, evaluated on a Synthea-generated cohort of
125 unique synthetic FHIR Conditions (the headline cohort of
[@FungPhantomCodes2026]), yields a substantially lower per-mode
top-1 accuracy in the **40–50% range**. Hallucination rate is 0%
across all four degradation modes by construction — the
classification head emits one of 50 known ICD-10-CM codes and
cannot fabricate non-existent codes.

The gap between in-distribution validation accuracy (~99%) and
out-of-distribution Synthea accuracy (~40–50%) is substantial and
deployment-relevant. Two factors plausibly drive it:

1. **Lexical surface differs systematically.** Synthea's clinical
   text is generated from rule-based modules and differs in vocabulary,
   register, and sentence structure from MIMIC's source notes. The
   classifier learned to map MIMIC-style surface forms to codes;
   Synthea's surface forms are out-of-distribution by construction.
2. **Cohort distribution differs.** The MIMIC training cohort is
   weighted toward conditions that present in critical care; the
   Synthea cohort is weighted toward population-health conditions
   (obesity, prediabetes, hypertension) by virtue of Synthea's
   default module configuration.

This out-of-distribution gap is the most important reported
finding for would-be deployers: a classifier that achieves 99% on
its source distribution may achieve substantially less on
clinically-realistic but compositionally-different inputs. The
companion paper [@FungPhantomCodes2026] uses this gap as the
non-LLM comparison floor against frontier LLMs evaluated on the
same Synthea inputs.

## Reproducibility

A single integer seed (`seed: 42`) is propagated to Python's
`random`, NumPy, and all PyTorch random number generators (CPU
and MPS) before model construction or data shuffling. Combined
with the deterministic stratified split (also seeded) and full
pinning of dependencies via `uv` lockfile, this is sufficient to
reproduce training outcomes within floating-point noise on the
same hardware. Bit-exact reproducibility on GPU/MPS additionally
requires `torch.use_deterministic_algorithms(True)`, which incurs
a 10–30% performance penalty; we do not enable it for v1.

The full set of training hyperparameters is captured in
`configs/training.yaml` and persisted into each saved checkpoint,
so inference code reads the configuration that produced the
weights rather than relying on a separately-tracked config file.

A credentialed reproducer with PhysioNet access can train a
byte-similar checkpoint by following the recipe documented in the
project repository's `BENCHMARK.md`. We do not redistribute the
trained weights; reproduction requires running the full pipeline
locally.
