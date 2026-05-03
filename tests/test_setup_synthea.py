"""Smoke tests for the Synthea setup script.

These tests are intentionally lightweight: we don't actually invoke
the setup script (cloning + building Synthea takes 3-5 minutes and
requires Java 17). We just verify the script exists, is executable,
and parses cleanly. The full integration is verified manually when
the user runs `./scripts/setup_synthea.sh`.

If Java 17 is available *and* the JAR has been built, we additionally
verify the JAR is launchable. Skipped otherwise.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_synthea.sh"
JAR = Path(__file__).resolve().parents[1] / "tools" / "synthea" / "build" / "libs" / "synthea-with-dependencies.jar"


def test_setup_script_exists_and_executable() -> None:
    """Script file exists at the expected path with execute permissions."""
    assert SCRIPT.exists(), f"missing: {SCRIPT}"
    assert os.access(SCRIPT, os.X_OK), f"not executable: {SCRIPT}"


def test_setup_script_help_runs() -> None:
    """Script's syntax parses (run with -h or as a no-op preflight check).

    We invoke `bash -n` which is bash's syntax-check mode — parses the
    script but doesn't execute any commands. Catches typos / unmatched
    quotes / bad syntax without actually cloning Synthea.
    """
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"bash -n failed:\n{result.stderr}"


@pytest.mark.skipif(
    not shutil.which("java"), reason="Java not installed; can't smoke-test JAR"
)
@pytest.mark.skipif(
    not JAR.exists(),
    reason="Synthea JAR not built yet; run ./scripts/setup_synthea.sh first",
)
def test_synthea_jar_launches() -> None:
    """If the JAR has been built, verify it launches and responds to -h."""
    result = subprocess.run(
        ["java", "-jar", str(JAR), "-h"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Synthea's -h exits 0 and prints usage to stdout
    assert result.returncode == 0, f"JAR launch failed:\n{result.stderr}"
    assert "Usage" in result.stdout or "synthea" in result.stdout.lower()
