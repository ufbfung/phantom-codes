#!/usr/bin/env bash
# Phantom Codes — one-time Synthea install + build.
#
# Clones Synthea at a pinned commit SHA into ./tools/synthea/ (gitignored)
# and builds the runnable JAR. Idempotent — safe to re-run.
#
# Usage:
#   ./scripts/setup_synthea.sh
#
# Pre-requisites:
#   - git
#   - Java 17 or newer (Synthea v4.0.0 requires JDK 17+)
#       macOS: brew install openjdk@17
#       Ubuntu: sudo apt install openjdk-17-jdk
#
# Output:
#   tools/synthea/build/libs/synthea-with-dependencies.jar
#       — the runnable JAR; consumed by scripts/generate_synthea_cohort.sh
#
# Pinned to Synthea v4.0.0 for reproducibility. Future contributors can
# bump SYNTHEA_SHA below to a newer release after re-running the SNOMED
# inventory step (see scripts/inventory_synthea_snomed.py) and confirming
# the curated SNOMED→ICD-10-CM map (data/synthea/snomed_to_icd10cm.json)
# still covers the in-scope conditions.

set -euo pipefail

SYNTHEA_REPO="https://github.com/synthetichealth/synthea.git"
SYNTHEA_SHA="0185c09ea9d10a822c6f5f3ef9bdcbcbe960c813"  # v4.0.0 (2026-03-05)
SYNTHEA_DIR="tools/synthea"
JAR_PATH="${SYNTHEA_DIR}/build/libs/synthea-with-dependencies.jar"

# ─── Pre-flight: Java version check ────────────────────────────────────
if ! command -v java >/dev/null 2>&1; then
    echo "❌ java not found on PATH."
    echo
    echo "Synthea v4.0.0 requires Java 17 or newer. Install:"
    echo "  macOS:  brew install openjdk@17"
    echo "  Ubuntu: sudo apt install openjdk-17-jdk"
    echo
    echo "After install on macOS, you may need to symlink the JDK:"
    echo "  sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk \\"
    echo "              /Library/Java/JavaVirtualMachines/openjdk-17.jdk"
    exit 1
fi

JAVA_MAJOR=$(java -version 2>&1 | head -1 | awk -F '"' '{print $2}' | awk -F . '{print $1}')
if [ "${JAVA_MAJOR}" -lt 17 ]; then
    echo "❌ Java ${JAVA_MAJOR} found, but Synthea v4.0.0 requires Java 17+."
    echo "Upgrade via your package manager and re-run."
    exit 1
fi

echo "[setup-synthea] Java ${JAVA_MAJOR} ✓"

# Defensive: Gradle reads JAVA_HOME independently of PATH-based `java`.
# A stale JAVA_HOME (e.g., pointing at an old Intel-Mac brew location
# after migrating to Apple Silicon) will crash Gradle even though
# `java -version` works fine. Unset if dangling so Gradle falls back
# to PATH lookup.
if [ -n "${JAVA_HOME:-}" ] && [ ! -d "${JAVA_HOME}" ]; then
    echo "[setup-synthea] ⚠️  JAVA_HOME=${JAVA_HOME} doesn't exist; unsetting for Gradle"
    echo "[setup-synthea]    (consider fixing ~/.zshrc to use \"\$(/usr/libexec/java_home)\")"
    unset JAVA_HOME
fi

# ─── Clone or fetch Synthea at pinned SHA ──────────────────────────────
if [ -d "${SYNTHEA_DIR}/.git" ]; then
    echo "[setup-synthea] ${SYNTHEA_DIR} exists; fetching latest refs"
    git -C "${SYNTHEA_DIR}" fetch --quiet origin
else
    echo "[setup-synthea] cloning ${SYNTHEA_REPO} → ${SYNTHEA_DIR}"
    git clone --quiet "${SYNTHEA_REPO}" "${SYNTHEA_DIR}"
fi

CURRENT_SHA=$(git -C "${SYNTHEA_DIR}" rev-parse HEAD)
if [ "${CURRENT_SHA}" != "${SYNTHEA_SHA}" ]; then
    echo "[setup-synthea] checking out pinned SHA ${SYNTHEA_SHA:0:12}..."
    git -C "${SYNTHEA_DIR}" checkout --quiet "${SYNTHEA_SHA}"
fi

# ─── Build the runnable JAR ────────────────────────────────────────────
if [ -f "${JAR_PATH}" ]; then
    echo "[setup-synthea] ${JAR_PATH} already built ✓"
else
    echo "[setup-synthea] building Synthea (this takes ~3-5 min on first run)…"
    pushd "${SYNTHEA_DIR}" > /dev/null
    ./gradlew --quiet build -x test
    popd > /dev/null
    if [ ! -f "${JAR_PATH}" ]; then
        echo "❌ build completed but ${JAR_PATH} not found. Inspect ${SYNTHEA_DIR}/build/libs/."
        exit 1
    fi
    echo "[setup-synthea] built ${JAR_PATH} ✓"
fi

# ─── Smoke-test the JAR ────────────────────────────────────────────────
echo "[setup-synthea] smoke-testing the JAR…"
java -jar "${JAR_PATH}" -h > /dev/null 2>&1 || {
    echo "❌ JAR runs but -h flag failed; smoke check inconclusive"
    exit 1
}

echo
echo "[setup-synthea] ✓ Synthea v4.0.0 ready at ${JAR_PATH}"
echo "[setup-synthea] next: ./scripts/generate_synthea_cohort.sh"
