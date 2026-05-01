"""Typer CLI for the phantom-codes benchmark."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from phantom_codes.config import DataConfig, load_data_config
from phantom_codes.data.disease_groups import load as load_scope
from phantom_codes.data.gcs_setup import copy_resources
from phantom_codes.data.icd10cm.validator import load as load_validator
from phantom_codes.data.prepare import prepare as prepare_pipeline
from phantom_codes.eval.runner import (
    load_records,
    run_eval,
    summarize_by_model_and_mode,
)
from phantom_codes.models.base import ConceptNormalizer
from phantom_codes.models.baselines import (
    ExactMatchBaseline,
    FuzzyMatchBaseline,
    TfidfBaseline,
)
from phantom_codes.models.llm import (
    PromptMode,
    make_anthropic_model,
    make_gemini_model,
    make_openai_model,
)

app = typer.Typer(
    name="phantom-codes",
    help="Phantom Codes — a benchmark for hallucination in LLM-based clinical concept normalization.",
    no_args_is_help=True,
)
console = Console()


@app.command("setup-data")
def setup_data(
    config_path: str = typer.Option("configs/data.yaml", "--config", help="Path to data config"),
) -> None:
    """One-time copy of MIMIC-IV-FHIR ndjson files from PhysioNet's bucket to your own bucket.

    Idempotent — files already present at the destination with matching size are skipped.
    """
    config = load_data_config(config_path)
    console.print(f"[bold]Source:[/] {config.physionet_bucket}")
    console.print(f"[bold]Destination:[/] {config.derived_bucket}/mimic/raw/")

    results = copy_resources(config)

    table = Table(title="setup-data results", show_header=True)
    table.add_column("resource")
    table.add_column("status")
    table.add_column("bytes")
    for r in results:
        table.add_row(
            r.resource,
            "skipped (idempotent)" if r.skipped else "copied",
            f"{r.bytes_copied:,}",
        )
    console.print(table)


@app.command()
def prepare(
    config_path: str = typer.Option("configs/data.yaml", "--config", help="Path to data config"),
    source: str = typer.Option(
        "",
        help=(
            "Override source URI for ndjson(.gz) of Conditions. Defaults to the raw URI "
            "in your derived bucket (i.e., the file copied by `setup-data`)."
        ),
    ),
    local_out: str = typer.Option(
        "",
        help="If set, write parquet splits to this local directory instead of GCS.",
    ),
) -> None:
    """Build degraded train/val/test parquet splits from FHIR Conditions."""
    config = load_data_config(config_path)
    src = source or config.raw_uri("MimicCondition")
    out_dir = Path(local_out) if local_out else None

    console.print(f"[bold]Reading[/] {src}")
    written = prepare_pipeline(config, src, local_out=out_dir)

    table = Table(title="prepare results", show_header=True)
    table.add_column("split")
    table.add_column("path")
    for split, path in written.items():
        table.add_row(split, path)
    console.print(table)


@app.command("smoke-test")
def smoke_test(
    fixtures: str = typer.Option(
        "tests/fixtures/conditions.ndjson",
        "--fixtures",
        help="Path to a local ndjson of FHIR Conditions to evaluate against.",
    ),
    include_llms: bool = typer.Option(
        False,
        "--llms/--no-llms",
        help=(
            "Also run LLM models (Claude Haiku zero-shot + constrained; GPT-4o-mini "
            "and Gemini 2.5 Flash zero-shot if their keys are set). Requires "
            "ANTHROPIC_API_KEY env var; OPENAI_API_KEY and GEMINI_API_KEY are optional. "
            "Costs a few cents per run."
        ),
    ),
    out: str = typer.Option(
        "",
        help="If set, write the per-prediction long-format CSV to this path.",
    ),
) -> None:
    """End-to-end smoke test on local fixtures.

    Runs the full pipeline (degrade → eval → outcome classification → summary)
    against the fixture conditions. Verifies all wiring without needing MIMIC.
    """
    scope = load_scope()
    validator = load_validator()
    console.print(f"[bold]ICD-10-CM lookup:[/] {validator.n_codes:,} codes")

    # Prepare records via the same pipeline `prepare` uses (degrade + scope filter).
    cfg = DataConfig.model_validate({
        "derived_bucket": "gs://throwaway/phantom-codes",
        "resources": ["MimicCondition"],
        "top_n_codes": 50,
        "seed": 42,
        "splits": {"train": 1.0, "val": 0.0, "test": 0.0},
    })
    with tempfile.TemporaryDirectory() as tmp:
        written = prepare_pipeline(cfg, fixtures, local_out=Path(tmp))
        records = load_records(written["train"])

    # Build the candidate list from the union of CMS explicit codes + observed codes
    # in the prepared records. Mirrors the production setup (option 1c).
    observed = list({(r.gt_code, r.gt_code) for r in records})  # display fallback to code
    # If we have ground-truth display, prefer it. Pull from one record per code.
    seen_codes: dict[str, str] = {}
    for r in records:
        if r.gt_code not in seen_codes and r.input_fhir:
            for c in (r.input_fhir.get("code") or {}).get("coding") or []:
                if c.get("display"):
                    seen_codes[r.gt_code] = c["display"]
                    break
    observed = [(code, seen_codes.get(code, code)) for code, _ in observed]
    candidates = scope.candidates_for_codes(observed)
    console.print(
        f"[bold]Candidates:[/] {len(candidates)} "
        f"(CMS explicit + {len(set(c[0] for c in observed))} observed)"
    )
    console.print(f"[bold]Records:[/] {len(records)} (in-scope conditions × 4 modes)")

    # Models.
    models: list[ConceptNormalizer] = [
        ExactMatchBaseline(candidates),
        FuzzyMatchBaseline(candidates),
        TfidfBaseline(candidates),
    ]
    if include_llms:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            console.print("[red]ANTHROPIC_API_KEY not set — skipping LLM models[/]")
        else:
            models.extend([
                make_anthropic_model(
                    name="claude-haiku-4-5:zeroshot",
                    model_id="claude-haiku-4-5",
                    mode=PromptMode.ZEROSHOT,
                ),
                make_anthropic_model(
                    name="claude-haiku-4-5:constrained",
                    model_id="claude-haiku-4-5",
                    mode=PromptMode.CONSTRAINED,
                    candidates=candidates,
                ),
            ])
        if os.environ.get("OPENAI_API_KEY"):
            models.append(
                make_openai_model(
                    name="gpt-4o-mini:zeroshot",
                    model_id="gpt-4o-mini",
                    mode=PromptMode.ZEROSHOT,
                )
            )
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            models.append(
                make_gemini_model(
                    name="gemini-2.5-flash:zeroshot",
                    model_id="gemini-2.5-flash",
                    mode=PromptMode.ZEROSHOT,
                )
            )

    console.print(f"[bold]Models:[/] {', '.join(m.name for m in models)}")

    df = run_eval(models, records, validator, top_k=5)
    summary = summarize_by_model_and_mode(df)

    table = Table(title="Smoke test summary (top-1 outcome rates by model × degradation mode)")
    for col in summary.columns:
        table.add_column(col)
    for row in summary.itertuples(index=False):
        formatted = []
        for col, val in zip(summary.columns, row, strict=True):
            if col in {"model_name", "mode", "n"}:
                formatted.append(str(val))
            else:
                formatted.append(f"{float(val):.2f}")
        table.add_row(*formatted)
    console.print(table)

    if out:
        df.to_csv(out, index=False)
        console.print(f"[green]Wrote per-prediction rows to {out}[/]")


@app.command()
def train(
    config_path: str = typer.Option("configs/models.yaml", "--config"),
    models: str = typer.Option("retrieval,classifier", help="Comma-separated model names"),
) -> None:
    """[stub] Fine-tune trained models on the train split."""
    console.print(f"[yellow]stub[/] train config={config_path} models={models}")


@app.command()
def evaluate(
    config_path: str = typer.Option("configs/eval.yaml", "--config"),
    models: str = typer.Option(..., help="Comma-separated model names"),
) -> None:
    """[stub] Run all (model, degradation) combinations and persist per-prediction results."""
    console.print(f"[yellow]stub[/] evaluate config={config_path} models={models}")


@app.command()
def report(
    out: str = typer.Option("results/report.md", help="Output report path"),
) -> None:
    """[stub] Aggregate results into CSVs and a markdown report."""
    console.print(f"[yellow]stub[/] report out={out}")


if __name__ == "__main__":
    app()
