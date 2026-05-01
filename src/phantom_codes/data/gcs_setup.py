"""Bootstrap step: copy MIMIC-IV-FHIR ndjson files from PhysioNet's GCS bucket
to the user's own bucket. Idempotent; safe to re-run.

Requires:
- PhysioNet credentialed access linked to your Google account
- `gcloud auth application-default login` on the host
- Storage Object Viewer on the source bucket (granted via PhysioNet linking)
- Storage Object Admin on the destination bucket (your own)
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)

from phantom_codes.config import DataConfig

console = Console()


@dataclass
class CopyResult:
    resource: str
    src: str
    dst: str
    bytes_copied: int
    skipped: bool


def copy_resources(config: DataConfig) -> list[CopyResult]:
    """Copy each configured resource file from PhysioNet's bucket to the user's bucket.

    Idempotent: skips files that already exist at the destination with matching size.
    """
    import gcsfs

    fs = gcsfs.GCSFileSystem()
    results: list[CopyResult] = []

    for resource in config.resources:
        src = config.physionet_uri(resource)
        dst = config.raw_uri(resource)
        result = _copy_one(fs, resource, src, dst)
        results.append(result)

    return results


def _copy_one(fs, resource: str, src: str, dst: str) -> CopyResult:
    """Copy a single ndjson.gz file, skipping if destination already matches source size."""
    src_size = _stat_size(fs, src)
    if src_size is None:
        raise FileNotFoundError(
            f"Source not found: {src}. Verify your Google account is linked to your "
            "PhysioNet profile and you have access to MIMIC-IV-FHIR v2.1."
        )

    dst_size = _stat_size(fs, dst)
    if dst_size == src_size:
        console.print(f"[dim]skip[/] {resource}: already at {dst} ({src_size:,} bytes)")
        return CopyResult(resource=resource, src=src, dst=dst, bytes_copied=0, skipped=True)

    console.print(f"[bold]copy[/] {src}\n  -> {dst} ({src_size:,} bytes)")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(resource, total=src_size)
        copied = 0
        chunk_size = 8 * 1024 * 1024  # 8 MiB
        with fs.open(src, "rb") as fin, fs.open(dst, "wb") as fout:
            while True:
                chunk = fin.read(chunk_size)
                if not chunk:
                    break
                fout.write(chunk)
                copied += len(chunk)
                progress.update(task, completed=copied)

    return CopyResult(resource=resource, src=src, dst=dst, bytes_copied=copied, skipped=False)


def _stat_size(fs, uri: str) -> int | None:
    try:
        info = fs.info(uri)
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return info.get("size")
