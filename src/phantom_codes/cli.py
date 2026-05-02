"""Typer CLI for the phantom-codes benchmark."""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from phantom_codes.config import DataConfig, load_data_config
from phantom_codes.data.disease_groups import load as load_scope
from phantom_codes.data.gcs_setup import make_gcs_filesystem
from phantom_codes.data.icd10cm.validator import load as load_validator
from phantom_codes.data.prepare import prepare as prepare_pipeline
from phantom_codes.eval.cost import load_pricing
from phantom_codes.eval.infra import InfraAssertions, infra_assertions
from phantom_codes.eval.manifest import (
    build_manifest,
    manifest_path_for,
    write_manifest,
)
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
    AnthropicClient,
    GoogleClient,
    OpenAIClient,
    PromptMode,
    build_system_prompt,
    build_user_message,
    make_anthropic_model,
    make_gemini_model,
    make_openai_model,
    parse_predictions,
)
from phantom_codes.models.rag_llm import make_rag_anthropic_model
from phantom_codes.models.retrieval import RetrievalModel

# Load .env (gitignored) so ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY
# are visible via os.environ before any provider client is constructed.
# No-op if .env is absent; existing environment variables take precedence.
load_dotenv()

app = typer.Typer(
    name="phantom-codes",
    help="Phantom Codes — a benchmark for hallucination in LLM-based clinical concept normalization.",
    no_args_is_help=True,
)
console = Console()


@app.command("verify-keys")
def verify_keys() -> None:
    """Quick API-key sanity check — one structured call per provider.

    For each provider whose key is set in `.env`, sends a single ICD-10
    prediction request to the cheapest model in that family and confirms
    (a) the auth works, (b) the structured-output schema returns valid
    JSON, and (c) the response parses into our Prediction format.

    Cost: pennies (~$0.001 per provider). Run this **before** the full
    smoke test to catch auth / quota / billing issues early.
    """
    import time

    from phantom_codes.eval.cost import (
        compute_call_cost,
        load_pricing,
        resolve_pricing_for_model,
    )

    system = build_system_prompt(PromptMode.ZEROSHOT)
    user = build_user_message(
        input_fhir=None,
        input_text="Type 2 diabetes mellitus without complications",
    )

    # Best-effort pricing load so we can show $ per call inline.
    pricing_path = Path("configs/pricing.yaml")
    pricing_table = load_pricing(pricing_path) if pricing_path.exists() else None

    # One cheapest model per provider — auth is per-provider, not per-model,
    # so verifying one model proves the key works for all models in that family.
    providers: list[tuple[str, str, list[str], type]] = [
        ("Anthropic", "claude-haiku-4-5", ["ANTHROPIC_API_KEY"], AnthropicClient),
        ("OpenAI", "gpt-4o-mini", ["OPENAI_API_KEY"], OpenAIClient),
        ("Google", "gemini-2.5-flash", ["GEMINI_API_KEY", "GOOGLE_API_KEY"], GoogleClient),
    ]

    table = Table(title="API key verification — one structured call per provider")
    table.add_column("provider")
    table.add_column("model_id")
    table.add_column("status")
    table.add_column("top prediction")
    table.add_column("tokens in/out", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("latency", justify="right")

    import json

    n_ok = 0
    n_skipped = 0
    n_failed = 0
    total_cost = 0.0
    n_priced = 0
    failures: list[tuple[str, str, Exception]] = []  # (provider, model_id, exception)
    suspicious_empty: list[tuple[str, str, dict]] = []  # connected but parsed 0 predictions
    for provider_name, model_id, env_keys, client_cls in providers:
        has_key = any(os.environ.get(k) for k in env_keys)
        if not has_key:
            table.add_row(
                provider_name, model_id, "[dim]skipped (no key)[/]",
                "—", "—", "—", "—",
            )
            n_skipped += 1
            continue

        client = client_cls(model_id=model_id)
        started = time.perf_counter()
        try:
            response = client.predict_structured(system, user)
            elapsed_ms = (time.perf_counter() - started) * 1000
            preds = parse_predictions(response.tool_input)
            if preds:
                top = f"{preds[0].code} ({preds[0].score:.2f})"
                status = "[green]✓ ok[/]"
            else:
                # API call succeeded but the response shape didn't match our
                # schema — worth flagging since the smoke test will silently
                # produce hallucination rows for every record otherwise.
                top = "[yellow](empty — schema mismatch?)[/]"
                status = "[yellow]⚠ ok (no preds)[/]"
                suspicious_empty.append((provider_name, model_id, response.tool_input))

            # Compute per-call cost from token counts × pricing.
            cost_str = "—"
            if pricing_table is not None:
                model_pricing = resolve_pricing_for_model(model_id, pricing_table)
                if model_pricing is not None:
                    call_cost = compute_call_cost(
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        cache_read_tokens=response.cache_read_tokens,
                        cache_creation_tokens=response.cache_creation_tokens,
                        pricing=model_pricing,
                    )
                    total_cost += call_cost
                    n_priced += 1
                    cost_str = f"${call_cost:.6f}"

            table.add_row(
                provider_name,
                model_id,
                status,
                top,
                f"{response.input_tokens} / {response.output_tokens}",
                cost_str,
                f"{elapsed_ms:.0f} ms",
            )
            n_ok += 1
        except Exception as e:
            elapsed_ms = (time.perf_counter() - started) * 1000
            err_short = f"{type(e).__name__}: {e}"
            if len(err_short) > 60:
                err_short = err_short[:57] + "..."
            table.add_row(
                provider_name, model_id, "[red]✗ failed[/]",
                err_short, "—", "—", f"{elapsed_ms:.0f} ms",
            )
            n_failed += 1
            failures.append((provider_name, model_id, e))

    console.print(table)
    summary_line = f"[bold]Summary:[/] {n_ok} ok, {n_skipped} skipped, {n_failed} failed"
    if n_priced > 0:
        summary_line += f"  [dim]·  total cost: ${total_cost:.6f} ({n_priced} priced)[/]"
    console.print(summary_line)
    if pricing_table is None:
        console.print(
            "[dim]Note: configs/pricing.yaml not found; per-call cost not computed.[/]"
        )

    if suspicious_empty:
        console.print()
        console.print(
            "[yellow bold]Suspicious empty-prediction responses "
            "(API call succeeded but parsed 0 predictions):[/]"
        )
        for provider_name, model_id, raw in suspicious_empty:
            console.print(f"\n[bold]{provider_name} ({model_id}) raw response:[/]")
            dump = json.dumps(raw, indent=2)
            # Cap the dump to keep terminal output readable.
            if len(dump) > 1500:
                dump = dump[:1500] + "\n  ... (truncated)"
            for line in dump.splitlines():
                console.print(f"  {line}")
        console.print(
            "\n[yellow]This usually means the response shape didn't match our "
            "predictions schema. Common cause: the model wrapped the response "
            "under a different key.[/]"
        )

    if failures:
        console.print()
        console.print("[red bold]Full error details:[/]")
        for provider_name, model_id, e in failures:
            console.print(
                f"\n[bold]{provider_name} ({model_id}):[/] {type(e).__name__}"
            )
            # Indent each line for readability.
            for line in str(e).splitlines():
                console.print(f"  {line}")
        console.print(
            "\n[red]Common causes: invalid key, no billing credit, "
            "model_id not enabled for the account, or rate limit.[/]"
        )
        raise typer.Exit(code=1)
    if suspicious_empty:
        # Connected but useless — exit non-zero so CI / scripts catch it.
        raise typer.Exit(code=1)
    if n_ok == 0:
        console.print(
            "[yellow]No keys were set. Add at least ANTHROPIC_API_KEY to your .env file.[/]"
        )
        raise typer.Exit(code=1)


@app.command("check-data")
def check_data(
    config_path: str = typer.Option("configs/data.yaml", "--config", help="Path to data config"),
) -> None:
    """Verify the user's GCS bucket contains the expected MIMIC-IV-FHIR files.

    PhysioNet does not host MIMIC-IV-FHIR on GCS — users must download the
    files manually over HTTPS (wget) and upload them to their own bucket
    with `gcloud storage cp`. See the README's "Data setup" section for the
    full walkthrough.

    This command lists the configured resources, checks each one's presence
    and size in `gs://{derived_bucket}/mimic/raw/`, and prints download +
    upload instructions if anything is missing.
    """
    config = load_data_config(config_path)
    console.print(f"[bold]Bucket:[/] {config.derived_bucket}/mimic/raw/")

    fs = make_gcs_filesystem()

    table = Table(title="check-data: expected files in derived bucket", show_header=True)
    table.add_column("resource")
    table.add_column("status")
    table.add_column("size", justify="right")

    missing: list[str] = []
    for resource in config.resources:
        uri = config.raw_uri(resource)
        try:
            info = fs.info(uri)
            size = int(info.get("size") or 0)
            table.add_row(resource, "[green]✓ present[/]", f"{size:,} B")
        except FileNotFoundError:
            table.add_row(resource, "[red]✗ missing[/]", "—")
            missing.append(resource)
        except Exception as e:  # noqa: BLE001 — surface arbitrary GCS errors
            table.add_row(resource, f"[red]✗ {type(e).__name__}[/]", "—")
            missing.append(resource)

    console.print(table)

    if not missing:
        console.print("[green]All configured resources are present.[/]")
        return

    # Helpful next-steps walkthrough for any missing resources.
    console.print(
        f"\n[yellow]Missing {len(missing)} resource(s).[/] "
        "PhysioNet doesn't mirror MIMIC-IV-FHIR to GCS — download manually and upload:\n"
    )
    console.print("[bold]1. Download from PhysioNet (HTTPS, requires credentialed access):[/]")
    console.print("   mkdir -p /tmp/mimic-fhir && cd /tmp/mimic-fhir")
    for resource in missing:
        console.print(
            f"   wget --user YOUR_PHYSIONET_USERNAME --ask-password "
            f"https://physionet.org/files/mimic-iv-fhir/2.1/fhir/{resource}.ndjson.gz"
        )
    console.print("\n[bold]2. Upload to your GCS bucket:[/]")
    console.print(
        f"   gcloud storage cp *.ndjson.gz {config.derived_bucket}/mimic/raw/"
    )
    console.print("\n[bold]3. Re-run check-data to verify.[/]")
    console.print(
        "\n[dim]Full walkthrough (including AWS and local-only paths) "
        "is in the README's 'Data setup' section.[/]"
    )
    raise typer.Exit(code=1)


@app.command()
def prepare(
    config_path: str = typer.Option("configs/data.yaml", "--config", help="Path to data config"),
    source: str = typer.Option(
        "",
        help=(
            "Override source URI for ndjson(.gz) of Conditions. Defaults to the raw URI "
            "in your derived bucket — the file you uploaded per the README's "
            "'Data setup' walkthrough. Use a local path here if you want to skip "
            "the cloud round-trip during development."
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
            "Also run LLM models. Haiku 4.5 runs all three prompting modes "
            "(zero-shot + constrained + RAG); premium models (Opus 4.7, Sonnet 4.6, "
            "GPT-5.5, Gemini 2.5 Pro) and economy comparators (GPT-4o-mini, Gemini "
            "2.5 Flash) run zero-shot only for wiring validation. Requires "
            "ANTHROPIC_API_KEY; OPENAI_API_KEY and GEMINI_API_KEY are optional. "
            "Cost: ~$3-5 per run with all keys set."
        ),
    ),
    include_retrieval: bool = typer.Option(
        True,
        "--retrieval/--no-retrieval",
        help=(
            "Include the bi-encoder retrieval baseline (frozen sentence-transformer "
            "+ cosine similarity). First run downloads ~80MB of weights."
        ),
    ),
    out: str = typer.Option(
        "",
        help=(
            "If set, write the per-prediction long-format CSV to this path. "
            "When --infra-only is set and --out is empty, defaults to "
            "results/raw/smoke_test_{utc_timestamp}.csv (gitignored)."
        ),
    ),
    infra_only: bool = typer.Option(
        False,
        "--infra-only",
        help=(
            "Blinded scaffolding mode: print structural wiring assertions only "
            "(call counts, token totals, cache hits, latency, classifier-bucket "
            "coverage). Suppresses the per-model performance summary. Use this "
            "during scaffolding to avoid bias from peeking at how each model "
            "performs on fixtures. Per-prediction CSV is still written so the "
            "data is available for later debugging."
        ),
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

    # Bi-encoder retrieval baseline (also reused as the retriever component of
    # RAG-LLM if LLMs are enabled). Built once; sentence-transformer download
    # happens here on first run.
    retrieval_model: RetrievalModel | None = None
    if include_retrieval:
        console.print("[dim]Building retrieval encoder (first run downloads weights)...[/]")
        retrieval_model = RetrievalModel(candidates)
        models.append(retrieval_model)

    if include_llms:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            console.print("[red]ANTHROPIC_API_KEY not set — skipping Anthropic models[/]")
        else:
            # Haiku gets all three prompting modes — full wiring validation on the
            # cheapest model. Premium models are zero-shot only.
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
                make_anthropic_model(
                    name="claude-sonnet-4-6:zeroshot",
                    model_id="claude-sonnet-4-6",
                    mode=PromptMode.ZEROSHOT,
                ),
                make_anthropic_model(
                    name="claude-opus-4-7:zeroshot",
                    model_id="claude-opus-4-7",
                    mode=PromptMode.ZEROSHOT,
                ),
            ])
            # RAG-LLM only makes sense when we have a retriever.
            if retrieval_model is not None:
                models.append(
                    make_rag_anthropic_model(
                        name="claude-haiku-4-5:rag",
                        model_id="claude-haiku-4-5",
                        retriever=retrieval_model,
                        candidates=candidates,
                        retrieve_k=20,
                    )
                )
        if os.environ.get("OPENAI_API_KEY"):
            models.extend([
                make_openai_model(
                    name="gpt-4o-mini:zeroshot",
                    model_id="gpt-4o-mini",
                    mode=PromptMode.ZEROSHOT,
                ),
                make_openai_model(
                    name="gpt-5.5:zeroshot",
                    model_id="gpt-5.5",
                    mode=PromptMode.ZEROSHOT,
                ),
            ])
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            models.extend([
                make_gemini_model(
                    name="gemini-2.5-flash:zeroshot",
                    model_id="gemini-2.5-flash",
                    mode=PromptMode.ZEROSHOT,
                ),
                make_gemini_model(
                    name="gemini-2.5-pro:zeroshot",
                    model_id="gemini-2.5-pro",
                    mode=PromptMode.ZEROSHOT,
                ),
            ])

    console.print(f"[bold]Models:[/] {', '.join(m.name for m in models)}")

    # Best-effort load of pricing so cost_usd gets populated per row at
    # runner time. If pricing.yaml is missing, cost columns will be None.
    pricing_path = Path("configs/pricing.yaml")
    pricing_table = load_pricing(pricing_path) if pricing_path.exists() else None

    started_at = datetime.now(UTC)
    df = run_eval(models, records, validator, top_k=5, pricing=pricing_table)
    finished_at = datetime.now(UTC)

    # Resolve the CSV output path. In infra-only mode, auto-default to a
    # timestamped path under results/raw/ (gitignored) so the data is
    # preserved without requiring an explicit --out flag.
    out_path = Path(out) if out else None
    if infra_only and out_path is None:
        ts = started_at.strftime("%Y%m%dT%H%M%SZ")
        out_path = Path("results/raw") / f"smoke_test_{ts}.csv"

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        console.print(f"[green]Wrote per-prediction rows to {out_path}[/]")

        manifest = build_manifest(
            run_id=started_at.strftime("%Y%m%dT%H%M%SZ"),
            command_name="smoke-test",
            started_at=started_at,
            finished_at=finished_at,
            seed=cfg.seed,
            fixtures_path=fixtures,
            n_records=len(records),
            n_candidates=len(candidates),
            models=models,
            df=df,
            pricing_table=pricing_table,
            csv_path=out_path,
            infra_only=infra_only,
        )
        manifest_path = manifest_path_for(out_path)
        write_manifest(manifest, manifest_path)
        console.print(f"[green]Wrote run manifest to {manifest_path}[/]")

    if infra_only:
        _print_infra_assertions(infra_assertions(df))
        console.print(
            "[dim]Performance summary suppressed (--infra-only). "
            "Run without the flag, or analyze the CSV directly, to view results.[/]"
        )
        return

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


def _print_infra_assertions(assertions: InfraAssertions) -> None:
    """Render structural wiring assertions to the console.

    Deliberately omits per-model accuracy / hallucination / bucket-distribution
    information — those are performance signals that bias scaffolding work.
    """
    table = Table(title="Smoke test wiring assertions (infra-only — no performance reveal)")
    table.add_column("model")
    table.add_column("calls", justify="right")
    table.add_column("errors", justify="right")
    table.add_column("tokens_in", justify="right")
    table.add_column("tokens_out", justify="right")
    table.add_column("cache_read", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")

    total_cost = 0.0
    has_any_cost = False
    total_errors = 0
    failing_models: list[tuple[str, int, str | None]] = []
    for m in assertions.per_model:
        if m.cost_usd is not None:
            cost_str = f"${m.cost_usd:.6f}"
            total_cost += m.cost_usd
            has_any_cost = True
        else:
            cost_str = "—"
        # Highlight error counts so they stand out at a glance.
        if m.n_errors > 0:
            err_str = f"[red]{m.n_errors}[/]"
            total_errors += m.n_errors
            failing_models.append((m.model_name, m.n_errors, m.dominant_error_type))
        else:
            err_str = "0"
        table.add_row(
            m.model_name,
            f"{m.n_calls:,}",
            err_str,
            f"{m.tokens_in:,}",
            f"{m.tokens_out:,}",
            f"{m.cache_read_tokens:,}",
            cost_str,
            f"{m.latency_p50_ms:.0f}" if m.latency_p50_ms is not None else "—",
            f"{m.latency_p95_ms:.0f}" if m.latency_p95_ms is not None else "—",
        )
    console.print(table)

    if has_any_cost:
        console.print(f"[bold]Total run cost:[/] ${total_cost:.6f}")

    if failing_models:
        console.print()
        console.print(
            f"[yellow bold]⚠ {total_errors} call(s) failed across "
            f"{len(failing_models)} model(s) — run completed via fault tolerance:[/]"
        )
        for name, n, dom in failing_models:
            dom_str = f" (dominant: {dom})" if dom else ""
            console.print(f"  • [yellow]{name}[/]: {n} errors{dom_str}")
        console.print(
            "[dim]Failed-call detail (error_type, error_msg) is in the CSV's rank-0 rows. "
            "Filter with `pandas.read_csv(...).query('error_type.notna()')`.[/]"
        )

    if assertions.all_buckets_reached:
        console.print(
            "[green]✓ Outcome buckets — all 5 reachable "
            "(exact_match, category_match, chapter_match, out_of_domain, hallucination)[/]"
        )
    else:
        console.print(
            f"[yellow]⚠ Outcome buckets not reached: {assertions.missing_buckets}[/]"
        )
        console.print(
            "[dim]Expected on small fixture sets if no fixture exercises those classifier "
            "branches. Confirm by inspecting the CSV directly if concerned about coverage.[/]"
        )


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
