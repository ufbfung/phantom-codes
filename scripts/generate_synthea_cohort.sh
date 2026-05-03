#!/usr/bin/env bash
# Phantom Codes — Synthea cohort generation.
#
# Reads configs/synthea.yaml, translates the YAML keys into Synthea CLI
# flags, runs the runnable JAR built by setup_synthea.sh. Output: a
# directory of FHIR R4 Bundle JSON files, one per synthetic patient.
#
# Usage:
#   ./scripts/generate_synthea_cohort.sh                # full cohort (uses configs/synthea.yaml)
#   ./scripts/generate_synthea_cohort.sh --pilot        # 100-patient pilot for SNOMED inventory
#
# The --pilot flag overrides population_size to 100 and writes to
# benchmarks/synthetic_v1/pilot/ instead of the configured output_dir.
# Pilot mode does NOT use the SNOMED→ICD-10-CM map — its purpose is to
# enumerate the SNOMED codes Synthea actually emits in our scope, which
# feeds the curation step that produces the map in the first place.

set -euo pipefail

JAR_PATH="tools/synthea/build/libs/synthea-with-dependencies.jar"
CONFIG_PATH="configs/synthea.yaml"
PILOT_MODE=false
PILOT_SIZE=100

# ─── Parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pilot)
            PILOT_MODE=true
            shift
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,17p' "$0" | sed 's/^# //; s/^#//'
            exit 0
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

# ─── Pre-flight ────────────────────────────────────────────────────────
if [ ! -f "${JAR_PATH}" ]; then
    echo "❌ ${JAR_PATH} not found. Run ./scripts/setup_synthea.sh first."
    exit 1
fi
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "❌ Config file ${CONFIG_PATH} not found."
    exit 1
fi

# ─── Parse YAML config (poor-man's YAML — we control the file format) ──
# Only handles flat key:value pairs and simple list entries. Don't
# extend without also extending this parser.
get_yaml_value() {
    local key="$1"
    grep -E "^${key}:" "${CONFIG_PATH}" | head -1 | awk -F': *' '{print $2}' | tr -d '"'
}
get_yaml_list() {
    local key="$1"
    awk -v key="${key}:" '
        $0 == key { in_list=1; next }
        in_list && /^[a-z_]+:/ { in_list=0 }
        in_list && /^  - / { gsub(/^  - /, ""); print }
    ' "${CONFIG_PATH}"
}

POPULATION_SIZE=$(get_yaml_value "population_size")
SEED=$(get_yaml_value "seed")
GENDER_RATIO=$(get_yaml_value "gender_ratio")
MIN_AGE=$(get_yaml_value "min_age")
MAX_AGE=$(get_yaml_value "max_age")
US_CORE_VERSION=$(get_yaml_value "fhir_us_core_version")
BULK_DATA=$(get_yaml_value "fhir_bulk_data")
CODE_MAP=$(get_yaml_value "icd10_code_map")
OUTPUT_DIR=$(get_yaml_value "output_dir")
DISABLED_MODULES=$(get_yaml_list "disabled_modules")

# ─── Apply --pilot overrides ───────────────────────────────────────────
if [ "${PILOT_MODE}" = true ]; then
    echo "[generate-synthea] PILOT mode: ${PILOT_SIZE} patients, no ICD-10-CM map"
    POPULATION_SIZE="${PILOT_SIZE}"
    OUTPUT_DIR="benchmarks/synthetic_v1/pilot"
    CODE_MAP=""  # pilot generates SNOMED-only — that's what we're inventorying
fi

mkdir -p "${OUTPUT_DIR}"

# ─── Build Synthea command ─────────────────────────────────────────────
# Synthea's CodeMapper resolves --exporter.code_map.* paths via the JVM
# *classpath* (Utilities.readResource → Guava Resources.getResource),
# NOT the filesystem. We need to add the directory containing our map
# to the classpath so Synthea can find the file by basename.
#
# We use `-cp` + main-class instead of `-jar` because `-jar` ignores
# any -cp additions. The fat JAR's manifest names `App` as Main-Class.
EXTRA_CP=""
CODE_MAP_RESOURCE=""
if [ -n "${CODE_MAP}" ]; then
    if [ ! -f "${CODE_MAP}" ]; then
        echo "❌ ICD-10-CM code map ${CODE_MAP} not found."
        echo "Run ./scripts/inventory_synthea_snomed.py first to build it."
        exit 1
    fi
    CODE_MAP_DIR=$(dirname "${CODE_MAP}")
    CODE_MAP_RESOURCE=$(basename "${CODE_MAP}")
    EXTRA_CP="${CODE_MAP_DIR}:"
    echo "[generate-synthea] using SNOMED→ICD-10-CM map: ${CODE_MAP}"
    echo "[generate-synthea] (added ${CODE_MAP_DIR} to classpath; Synthea resolves the map as resource ${CODE_MAP_RESOURCE})"
fi

CMD=(java -cp "${EXTRA_CP}${JAR_PATH}" App)
CMD+=(-p "${POPULATION_SIZE}")
CMD+=(-s "${SEED}")
CMD+=(-a "${MIN_AGE}-${MAX_AGE}")
CMD+=(--exporter.baseDirectory="${OUTPUT_DIR}")
CMD+=(--exporter.fhir.export=true)
CMD+=(--exporter.fhir.us_core_version="${US_CORE_VERSION}")
CMD+=(--exporter.fhir.bulk_data="${BULK_DATA}")
CMD+=(--exporter.csv.export=false)
CMD+=(--exporter.text.export=false)
CMD+=(--exporter.cpcds.export=false)
CMD+=(--exporter.years_of_history=0)  # full lifespan history (default)

if [ -n "${CODE_MAP_RESOURCE}" ]; then
    CMD+=(--exporter.code_map.icd10-cm="${CODE_MAP_RESOURCE}")
fi

# Disabled modules — Synthea expects comma-separated module IDs
if [ -n "${DISABLED_MODULES}" ]; then
    DISABLED_CSV=$(echo "${DISABLED_MODULES}" | tr '\n' ',' | sed 's/,$//')
    CMD+=(--generate.only_dead_patients=false)
    # Synthea uses --module_override OR per-module flags. The clean
    # approach is to point at an exclude list via --modules_dir and
    # only include the desired modules; alternatively, the simpler
    # per-module disable uses the exporter.fhir.included_resources
    # mechanism. For v1 we accept the default module set and rely on
    # post-hoc scope filtering in WS2's loader.
    echo "[generate-synthea] (disabled_modules in YAML — applied via post-hoc filter, not Synthea CLI)"
fi

# ─── Run ───────────────────────────────────────────────────────────────
echo "[generate-synthea] writing ${POPULATION_SIZE} patients → ${OUTPUT_DIR}"
echo "[generate-synthea] command: ${CMD[*]}"
echo

START_TIME=$(date +%s)
"${CMD[@]}"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ─── Verify output ─────────────────────────────────────────────────────
N_BUNDLES=$(find "${OUTPUT_DIR}/fhir" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
echo
echo "[generate-synthea] ✓ generated ${N_BUNDLES} Bundle JSON files in ${ELAPSED}s"
echo "[generate-synthea] output: ${OUTPUT_DIR}/fhir/"

if [ "${PILOT_MODE}" = true ]; then
    echo "[generate-synthea] next: uv run python scripts/inventory_synthea_snomed.py"
else
    echo "[generate-synthea] next: uv run phantom-codes prepare-synthea --bundles-dir ${OUTPUT_DIR}/fhir"
fi
