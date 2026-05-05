# Appendix: Reproduction Recipe

This appendix provides everything a credentialed reader needs to
reproduce the trained checkpoint described above from a clean clone
of the project repository. Reproduction requires PhysioNet
credentialing (the MIMIC-IV-FHIR dataset is not redistributed via
this repository).

## A.1 — Reproduction sequence

```bash
# 1. Install dependencies
uv sync --extra dev

# 2. Download MIMIC-IV-FHIR v2.1 from PhysioNet (requires credentials)
mkdir -p data/mimic/raw && cd data/mimic/raw
wget --user YOUR_PHYSIONET_USERNAME --ask-password \
    https://physionet.org/files/mimic-iv-fhir/2.1/fhir/MimicCondition.ndjson.gz
wget --user YOUR_PHYSIONET_USERNAME --ask-password \
    https://physionet.org/files/mimic-iv-fhir/2.1/fhir/MimicPatient.ndjson.gz
wget --user YOUR_PHYSIONET_USERNAME --ask-password \
    https://physionet.org/files/mimic-iv-fhir/2.1/fhir/MimicEncounter.ndjson.gz
cd ../../..

# 3. Build the parquet derivative splits
uv run phantom-codes prepare \
    --source data/mimic/raw/MimicCondition.ndjson.gz \
    --local-out data/derived

# 4. Fine-tune PubMedBERT on the local data (~15 hr on M1 MPS)
uv run phantom-codes train \
    --config configs/training.yaml \
    --output models/checkpoints/pubmedbert
```

The full reproduction recipe is documented in the project's
`BENCHMARK.md` reference file.

## A.2 — Pinned dependencies and software versions

A complete `requirements.txt` produced by `uv export
--format requirements-txt` at the headline-checkpoint commit SHA
is committed alongside this paper for full reproducibility. Pin to
specific package versions so future readers get byte-similar
behavior from the wrapper code.

## A.3 — Deterministic configurations

The following configuration files fully specify the trained-model
arm:

- `configs/training.yaml` — training hyperparameters (peak LR,
  warmup steps, batch size, max sequence length, epochs, patience,
  random seed)
- `configs/data.yaml` — data paths, value-set scope filter
- `src/phantom_codes/data/access_valuesets/` — bundled CMS ACCESS
  Model FHIR ValueSets (v0.9.6) used as the scope filter

All three are committed at the checkpoint commit SHA recorded in
the checkpoint manifest sidecar.

## A.4 — What's NOT redistributed

Per PhysioNet's responsible-LLM-use policy [@PhysioNet2025],
fine-tuned weights derived from MIMIC are themselves credentialed-
derivative material and are not redistributed via this repository,
HuggingFace Hub, or any cloud bucket. A credentialed reader can
reproduce a byte-similar checkpoint from the recipe in §A.1; we
do not provide a downloadable checkpoint.

The companion paper [@FungPhantomCodes2026] documents the LLM
evaluation arm, which runs entirely on Synthea-generated data and
is fully reproducible without PhysioNet credentialing.
