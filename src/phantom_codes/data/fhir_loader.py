"""Stream FHIR resources from ndjson.gz files (local or GCS).

Used by both the MIMIC pipeline (reading from GCS) and tests (reading from local fixtures).
The same iterator interface keeps the rest of the pipeline storage-agnostic.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_fhir_resources(uri: str | Path) -> Iterator[dict[str, Any]]:
    """Yield FHIR resources from a `.ndjson` or `.ndjson.gz` file.

    Accepts either a `gs://...` URI or a local path. GCS reads use the user's ADC
    (run `gcloud auth application-default login` once on the host).
    """
    uri_str = str(uri)
    is_gz = uri_str.endswith(".gz")

    if uri_str.startswith("gs://"):
        import gcsfs  # imported lazily so unit tests don't pull GCP creds

        fs = gcsfs.GCSFileSystem()
        opener = fs.open(uri_str, "rb")
    else:
        opener = open(uri_str, "rb")

    with opener as raw:
        stream = gzip.open(raw, "rt", encoding="utf-8") if is_gz else raw
        try:
            for line in stream:
                line = line.strip() if isinstance(line, str) else line.decode().strip()
                if not line:
                    continue
                yield json.loads(line)
        finally:
            if is_gz:
                stream.close()


def iter_conditions(uri: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate FHIR Condition resources only — skips anything else (e.g., from a Bundle file)."""
    for resource in iter_fhir_resources(uri):
        if resource.get("resourceType") == "Condition":
            yield resource


def count_resources(uri: str | Path) -> int:
    """Convenience: count resources at a URI. Streams; doesn't load everything in memory."""
    return sum(1 for _ in iter_fhir_resources(uri))
