"""Phantom Codes — SNOMED-CT inventory from a Synthea pilot cohort.

Walks a directory of Synthea-generated FHIR Bundles, extracts every
unique SNOMED code that appears in `Condition.code.coding[]`, and
filters the result to codes whose display strings clinically suggest
membership in the ACCESS Model scope (diabetes, hypertension, CKD,
ASCVD, dyslipidemia, prediabetes, obesity).

Output: `data/synthea/snomed_inventory_pilot.json` — a JSON object
keyed by SNOMED concept ID, valued with the Synthea-generated display
string and a count of how many times the code appeared in the pilot.

This script feeds the curation step that produces the canonical
`data/synthea/snomed_to_icd10cm.json` map used by Synthea's exporter
on the full cohort generation. See the project plan's "Curation
handoff protocol" section for the full workflow.

Usage:
    # 1. Generate the pilot cohort first (100 patients, no ICD map)
    ./scripts/generate_synthea_cohort.sh --pilot

    # 2. Run this inventory
    uv run python scripts/inventory_synthea_snomed.py

    # 3. Output appears at data/synthea/snomed_inventory_pilot.json
    #    Hand off to the curation step.

This script is intentionally a standalone runnable Python file (not
part of the `phantom-codes` CLI) because it's a one-time tooling step,
not part of the user-facing benchmark workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────
# Heuristic keyword filter — scoping by display-string substrings.
# These keywords are intentionally conservative; we want to err toward
# capturing more codes for the curator to review (drop irrelevant ones)
# rather than missing in-scope codes (would silently exclude them from
# the eventual map).
# ─────────────────────────────────────────────────────────────────────────
ACCESS_KEYWORDS = (
    # Diabetes
    "diabetes", "diabetic", "hyperglycaemia", "hyperglycemia",
    "prediabet", "pre-diabet", "metabolic syndrome",
    # Hypertension
    "hypertensi", "blood pressure", "htn",
    # CKD
    "kidney", "renal", "ckd", "nephropathy", "glomerular",
    # ASCVD / cardiovascular
    "atherosclero", "coronary", "myocardial", "ischemic heart",
    "ischaemic heart", "ascvd", "stroke", "cerebrovascular",
    "peripheral vascular", "peripheral arter",
    # Dyslipidemia
    "hyperlipid", "dyslipid", "hypercholesterol", "hypertriglyc",
    "cholesterol",
    # Obesity
    "obes", "morbid obesity", "body mass index",
)

DEFAULT_BUNDLES_DIR = Path("benchmarks/synthetic_v1/pilot/fhir")
DEFAULT_OUTPUT_PATH = Path("data/synthea/snomed_inventory_pilot.json")
SNOMED_SYSTEM = "http://snomed.info/sct"


def is_access_relevant(display: str) -> bool:
    """Heuristic: does this display string look like an ACCESS-scope diagnosis?"""
    if not display:
        return False
    needle = display.lower()
    return any(kw in needle for kw in ACCESS_KEYWORDS)


def iter_bundles(directory: Path):
    """Yield (filename, bundle_dict) for every .json file in the directory."""
    if not directory.exists():
        raise FileNotFoundError(
            f"Bundles directory not found: {directory}\n"
            "Generate a cohort first:\n"
            "    ./scripts/generate_synthea_cohort.sh --pilot   # for inventory pilot\n"
            "    ./scripts/generate_synthea_cohort.sh           # for full cohort"
        )
    for path in sorted(directory.glob("*.json")):
        try:
            yield path.name, json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"  ⚠️  skipping malformed JSON: {path.name} ({e})", file=sys.stderr)


def extract_conditions(bundle: dict[str, Any]):
    """Yield Condition resources from a Synthea Bundle."""
    for entry in bundle.get("entry", []) or []:
        resource = entry.get("resource") or {}
        if resource.get("resourceType") == "Condition":
            yield resource


def collect_snomed_inventory(bundles_dir: Path) -> dict[str, dict[str, Any]]:
    """Walk Bundles, aggregate unique SNOMED codes with counts + displays.

    Returns a dict keyed by SNOMED concept ID, with display + count.
    Only codes whose display matches the ACCESS keyword filter are kept.
    """
    inventory: dict[str, dict[str, Any]] = {}
    counter: Counter[str] = Counter()
    n_bundles = 0
    n_conditions = 0
    n_in_scope_conditions = 0

    for _name, bundle in iter_bundles(bundles_dir):
        n_bundles += 1
        for cond in extract_conditions(bundle):
            n_conditions += 1
            code_obj = cond.get("code") or {}
            for coding in code_obj.get("coding") or []:
                if coding.get("system") != SNOMED_SYSTEM:
                    continue
                snomed_id = str(coding.get("code") or "")
                display = str(coding.get("display") or "")
                if not snomed_id:
                    continue
                if not is_access_relevant(display):
                    continue
                counter[snomed_id] += 1
                if snomed_id not in inventory:
                    inventory[snomed_id] = {
                        "snomed_display": display,
                        "n_observed": 0,
                    }
                # Update count on every observation
                inventory[snomed_id]["n_observed"] = counter[snomed_id]
                n_in_scope_conditions += 1

    print(f"[inventory] bundles processed: {n_bundles}")
    print(f"[inventory] total Conditions seen: {n_conditions}")
    print(f"[inventory] in-scope (per keyword filter): {n_in_scope_conditions}")
    print(f"[inventory] unique in-scope SNOMED codes: {len(inventory)}")
    return inventory


def write_inventory(
    inventory: dict[str, dict[str, Any]],
    out_path: Path,
    bundles_dir: Path,
) -> None:
    """Write the inventory to JSON, sorted by SNOMED ID for stability."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generator": "scripts/inventory_synthea_snomed.py",
        "bundles_dir": str(bundles_dir),
        "n_unique_snomed_codes": len(inventory),
        "scoping_keywords": list(ACCESS_KEYWORDS),
        "entries": dict(sorted(inventory.items())),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[inventory] wrote {out_path}")


def diff_against_curated_map(
    inventory: dict[str, dict[str, Any]],
    map_path: Path,
    excluded_path: Path | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Compare the inventory against the curated SNOMED→ICD map.

    The "known" set is the union of:
      - SNOMED IDs present in the curated map (in-scope, mapped)
      - SNOMED IDs present in the optional exclusion file (out-of-scope,
        deliberately not mapped)

    A SNOMED ID in inventory that's in NEITHER known set is a genuinely
    new code requiring curation.

    Returns (new_codes, excluded_codes_seen, missing_codes):
      - new_codes: in inventory, NOT in map, NOT in excluded list
                   → require curation
      - excluded_codes_seen: in inventory AND in excluded list
                             → already evaluated, intentionally dropped
      - missing_codes: in map but NOT in inventory
                       → curated entries this cohort didn't surface
                         (informational; usually fine)
    """
    curated = json.loads(map_path.read_text())
    curated_keys = {k for k in curated.keys() if not k.startswith("_")}

    excluded_keys: set[str] = set()
    if excluded_path is not None and excluded_path.exists():
        excluded_payload = json.loads(excluded_path.read_text())
        excluded_keys = set(excluded_payload.get("exclusions", {}).keys())

    inventory_keys = set(inventory.keys())
    known_keys = curated_keys | excluded_keys

    new_codes = sorted(inventory_keys - known_keys)
    excluded_codes_seen = sorted(inventory_keys & excluded_keys)
    missing_codes = sorted(curated_keys - inventory_keys)
    return new_codes, excluded_codes_seen, missing_codes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory the unique SNOMED concept IDs that appear in "
            "Synthea-generated FHIR Bundles for ACCESS-scope conditions. "
            "Defaults run against the 100-patient pilot used to seed map "
            "curation; pass --bundles-dir to re-run against the full cohort."
        )
    )
    parser.add_argument(
        "--bundles-dir",
        type=Path,
        default=DEFAULT_BUNDLES_DIR,
        help=(
            "Directory containing Synthea FHIR Bundle JSON files. "
            f"Default: {DEFAULT_BUNDLES_DIR} (the pilot output)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Output path for the inventory JSON. "
            f"Default: {DEFAULT_OUTPUT_PATH}. When running against the "
            "full cohort, suggest data/synthea/snomed_inventory_full.json."
        ),
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        help=(
            "Optional: path to the curated SNOMED→ICD-10-CM map "
            "(typically data/synthea/snomed_to_icd10cm.json). When set, "
            "the script reports SNOMED IDs in the inventory that are NOT "
            "yet in the map (need curation) and IDs in the map that did "
            "NOT appear in this inventory (informational)."
        ),
    )
    args = parser.parse_args()

    try:
        inventory = collect_snomed_inventory(args.bundles_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    if not inventory:
        print(
            "⚠️  No in-scope SNOMED codes found. Either the cohort is too small, "
            "or the keyword filter needs adjustment in this script "
            "(ACCESS_KEYWORDS).",
            file=sys.stderr,
        )
        return 1

    write_inventory(inventory, args.out, args.bundles_dir)

    # ─── Optional: diff against curated map ──────────────────────────
    if args.compare_with is not None:
        if not args.compare_with.exists():
            print(
                f"⚠️  --compare-with path not found: {args.compare_with}",
                file=sys.stderr,
            )
            return 1

        # Auto-locate the exclusion file alongside the map (same directory,
        # canonical name). Quietly skip if it doesn't exist.
        excluded_path = args.compare_with.parent / "snomed_excluded.json"
        if not excluded_path.exists():
            excluded_path = None

        new_codes, excluded_seen, missing_codes = diff_against_curated_map(
            inventory, args.compare_with, excluded_path
        )

        print()
        print(f"[inventory] diff vs. {args.compare_with}")
        if excluded_path is not None:
            print(f"[inventory]      and exclusions file {excluded_path}:")
        else:
            print("[inventory]      (no exclusions file found):")

        if not new_codes:
            print("[inventory] ✓ no new SNOMED codes — curated map + exclusions cover full inventory")
        else:
            print(
                f"[inventory] ⚠️  {len(new_codes)} new SNOMED codes need curation:"
            )
            for snomed_id in new_codes:
                entry = inventory[snomed_id]
                print(
                    f"          - {snomed_id} ({entry['n_observed']}× observed): "
                    f"{entry['snomed_display']}"
                )

        if excluded_seen:
            print(
                f"[inventory] (deliberately excluded: {len(excluded_seen)} SNOMED codes "
                "appeared again — already evaluated and dropped per snomed_excluded.json)"
            )

        if missing_codes:
            print(
                f"[inventory] ℹ️  {len(missing_codes)} SNOMED codes in the map "
                "did NOT appear in this cohort (likely fine — usually means "
                "Synthea didn't emit them at this seed/scale):"
            )
            for snomed_id in missing_codes:
                print(f"          - {snomed_id}")

        if new_codes:
            print()
            print("[inventory] NEXT: curate the new entries (Path A — Claude drafts,")
            print("[inventory]       Brian reviews); regenerate cohort; re-run this")
            print("[inventory]       diff to confirm zero new codes; then prepare-synthea.")
            return 2  # exit code 2 signals "diff has new codes"

    print()
    print("[inventory] ✓ done.")
    if args.compare_with is None:
        print("[inventory] NEXT: curate the SNOMED→ICD-10-CM map, then regenerate cohort.")
    else:
        print("[inventory] NEXT: cohort matches map; safe to run prepare-synthea.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
