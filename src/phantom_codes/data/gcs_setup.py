"""GCS filesystem helper for reading from / writing to the user's own bucket.

Note: PhysioNet's MIMIC-IV-FHIR v2.1 dataset is **not** mirrored to GCS — it
is hosted via HTTPS download and AWS S3 only. The previous bootstrap step
that did a GCS-to-GCS copy from PhysioNet's bucket to the user's was removed
once we discovered this. The current data-setup workflow is documented in
the README's "Data setup" section: manually `wget` the files from PhysioNet
over HTTPS, then `gcloud storage cp` to the user's own bucket.

This module now contains only the shared filesystem-construction helper used
by `fhir_loader` (when reading from `gs://...` URIs) and the `check-data`
CLI command (when verifying the user's bucket has the expected files).
"""

from __future__ import annotations

import os
from typing import Any


def make_gcs_filesystem() -> Any:
    """Construct a `gcsfs.GCSFileSystem` configured with the billing project.

    Reads `GCP_PROJECT` from the environment so requests bill against the
    user's project (not gcsfs's auto-detected fallback, which can pick the
    wrong account when multiple Google credentials are present on the host).

    The user's destination bucket is not requester-pays, so we don't pass
    `requester_pays=True` here.
    """
    import gcsfs

    project = os.environ.get("GCP_PROJECT")
    return gcsfs.GCSFileSystem(project=project)
