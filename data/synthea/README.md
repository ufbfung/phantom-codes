# `data/synthea/` — Synthea cohort SNOMED→ICD curation artifacts

This directory holds the curated SNOMED-CT → ICD-10-CM mapping that
Synthea's exporter applies during cohort generation, plus the
companion files that document what was considered and why.

All three files are committed to the repo as the curation audit
trail. They're small (a few KB each) and human-readable.

## Files

| File | Purpose | Format | Consumed by |
|---|---|---|---|
| `snomed_to_icd10cm.json` | The active map: SNOMED concept ID → ICD-10-CM code that gets emitted alongside the SNOMED coding on every Synthea-generated Condition | Synthea's `code_map.icd10-cm` JSON spec (keyed by source SNOMED ID; values are arrays of `{code, description, ...}` objects) | Synthea (CLI flag `--exporter.code_map.icd10-cm`) AND `tests/test_snomed_to_icd_map.py` AND `scripts/inventory_synthea_snomed.py --compare-with` |
| `snomed_excluded.json` | SNOMED codes observed in pilot output but deliberately NOT added to the active map (their canonical ICD targets are out of ACCESS Model scope) | Custom JSON: `{"exclusions": {"<snomed-id>": {snomed_display, would_map_to, reason}}}` | `scripts/inventory_synthea_snomed.py --compare-with` (treats these as "already evaluated", not "new") |
| `snomed_inventory_pilot.json` | Snapshot of unique SNOMED codes the 100-patient pilot cohort surfaced (input to the curation step) | Custom JSON: `{entries: {"<snomed-id>": {snomed_display, n_observed}}}` | Auditability only — regenerable via `scripts/inventory_synthea_snomed.py` |

The inventory file is regenerable but committed for transparency:
shows future contributors / reviewers exactly which SNOMED codes
Synthea v4.0.0 emits for ACCESS-scope conditions at 100-patient
scale.

## Why `snomed_to_icd10cm.json` has no `_README` field

Synthea's `CodeMapper.java` deserializes the file as
`HashMap<String, List<Map<String, String>>>` — every top-level value
must be an array. A string-valued `_README` key crashes the parser.
This file therefore sticks strictly to Synthea's schema; this README
holds the prose documentation instead.

`snomed_excluded.json` is NOT consumed by Synthea (only by our
`inventory_synthea_snomed.py` diff feature), so its `_README` field
is harmless and retained.

## Curation workflow

The full handoff protocol lives in the project plan (gitignored
locally). High-level:

1. Generate a small pilot cohort (100 patients) with no code map
   applied: `./scripts/generate_synthea_cohort.sh --pilot`
2. Run the inventory script to enumerate ACCESS-relevant SNOMED
   codes: `uv run python scripts/inventory_synthea_snomed.py`
3. Curate `snomed_to_icd10cm.json` (in-scope mappings) and
   `snomed_excluded.json` (out-of-scope codes with rationale)
4. Validate: `uv run pytest tests/test_snomed_to_icd_map.py -q`
5. Generate the full cohort with the map applied:
   `./scripts/generate_synthea_cohort.sh`
6. Re-inventory the full cohort and diff against the curated files:
   ```bash
   uv run python scripts/inventory_synthea_snomed.py \
       --bundles-dir benchmarks/synthetic_v1/raw/fhir \
       --out data/synthea/snomed_inventory_full.json \
       --compare-with data/synthea/snomed_to_icd10cm.json
   ```
   If new codes surface, repeat steps 3-5 for the deltas only.

## Updating the map

When Synthea adds new modules or new SNOMED codes appear in larger
cohorts, the curation cycle re-runs. Each entry should record:

- `snomed_display` — Synthea's display string (verbatim from output)
- `access_group` — `ckm` | `eckm` | `OUT_OF_SCOPE`
- `confidence` — `high` | `needs_review` | `verified`
- `source` — provenance for the mapping decision (e.g., "SNOMED browser
  canonical mapping + ACCESS CKM ValueSet")
- `notes` — clinical rationale or methodology notes
- `n_observed_pilot` — count from the inventory (informational)

The `confidence` field gates the validator: `needs_review` causes the
`test_no_remaining_needs_review_after_signoff` test to xfail, signaling
that curation is incomplete.
