"""Aggregate per-prediction CSVs into paper-ready result tables.

Consumes the long-format prediction CSV produced by `evaluate`'s
streaming runner and produces five aggregate views ready to drop
into §3 Results of the manuscript:

  - `headline_table`        : per-(model, mode) outcome distribution
  - `hallucination_table`   : per-mode hallucination rate by model + 95% CI
  - `cost_per_correct_table`: per-model $ per exact-match prediction
  - `top_k_lift_table`      : top-1 vs top-5 exact-match comparison
  - `per_bucket_cost_table` : cost decomposition by outcome bucket

All tables emit both a DataFrame and a markdown string. Markdown is
the default output format because §3 Results is currently markdown
and pandoc-driven; CSV / LaTeX outputs are also supported via the
`format=` argument on the public `write_report()` function.

Compliance posture: this module reads only aggregate per-prediction
output already on disk (Synthea-derived CSV — safe). Never touches
MIMIC content. Doesn't surface per-record predictions in any
aggregate output — only counts, rates, and totals.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from phantom_codes.eval.cost import PricingTable, compute_call_cost, resolve_pricing_for_model
from phantom_codes.eval.metrics import Outcome


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _top1_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rank-0 rows (one per (model, record) pair).

    The runner emits top_k rows per evaluation; only rank-0 has the
    cost / latency / token metadata. For aggregation we want one row
    per record, not per slot.
    """
    return df[df["pred_rank"] == 0].copy()


def _wilson_95ci(k: int, n: int) -> tuple[float, float]:
    """95% Wilson confidence interval for a proportion.

    Wilson is preferred over normal-approximation for small n or
    rates near 0/1 (typical for hallucination rates which can be 0%
    or close to it for trained baselines).
    """
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _format_pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _format_pct_with_ci(rate: float, lo: float, hi: float) -> str:
    return f"{rate * 100:.1f}% ({lo * 100:.1f}-{hi * 100:.1f})"


def _format_usd(amount: float | None) -> str:
    if amount is None or pd.isna(amount):
        return "—"
    if amount == 0:
        return "$0.00"
    if amount < 0.01:
        return f"${amount:.4f}"
    return f"${amount:.2f}"


# ─────────────────────────────────────────────────────────────────────────
# Public table generators
# ─────────────────────────────────────────────────────────────────────────


def headline_table(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Per-(model, mode) outcome distribution — the §3 Results headline.

    Rows: (model_name, mode) pairs.
    Columns: count of predictions in each Outcome bucket.

    The 5-way taxonomy gets one column per bucket plus an `n` column.
    Order: exact_match, category_match, chapter_match, out_of_domain,
    hallucination — matches the paper's §1 description.
    """
    top1 = _top1_only(df)
    rows: list[dict[str, Any]] = []
    for (model, mode), grp in top1.groupby(["model_name", "mode"]):
        n = len(grp)
        row = {"model": model, "mode": mode, "n": n}
        for outcome in Outcome:
            count = int((grp["best_top1"] == outcome.value).sum())
            row[outcome.value] = count
            row[f"{outcome.value}_pct"] = (count / n) if n else 0.0
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["model", "mode"]).reset_index(drop=True)

    # Markdown rendering — counts + percentages per bucket
    lines = ["| Model | Mode | n | Exact | Category | Chapter | OOD | Hallucination |"]
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['mode']} | {r['n']} | "
            f"{r[Outcome.EXACT_MATCH.value]} ({_format_pct(r[Outcome.EXACT_MATCH.value + '_pct'])}) | "
            f"{r[Outcome.CATEGORY_MATCH.value]} ({_format_pct(r[Outcome.CATEGORY_MATCH.value + '_pct'])}) | "
            f"{r[Outcome.CHAPTER_MATCH.value]} ({_format_pct(r[Outcome.CHAPTER_MATCH.value + '_pct'])}) | "
            f"{r[Outcome.OUT_OF_DOMAIN.value]} ({_format_pct(r[Outcome.OUT_OF_DOMAIN.value + '_pct'])}) | "
            f"{r[Outcome.HALLUCINATION.value]} ({_format_pct(r[Outcome.HALLUCINATION.value + '_pct'])}) |"
        )
    return out, "\n".join(lines)


def hallucination_table(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Per-mode hallucination rate by model with 95% CI.

    Wilson CIs handle small-n cells and rates at the boundary
    correctly (some configurations may have 0% hallucination, which
    breaks normal-approximation CIs).

    The headline metric of the paper. Highlight: per-mode breakdown
    surfaces whether D4_abbreviated stress-tests reveal patterns
    obscured by aggregate top-1 numbers.
    """
    top1 = _top1_only(df)
    rows: list[dict[str, Any]] = []
    for (model, mode), grp in top1.groupby(["model_name", "mode"]):
        n = len(grp)
        k = int((grp["best_top1"] == Outcome.HALLUCINATION.value).sum())
        rate = (k / n) if n else 0.0
        lo, hi = _wilson_95ci(k, n)
        rows.append({
            "model": model,
            "mode": mode,
            "n": n,
            "n_hallucinations": k,
            "rate": rate,
            "ci_lower": lo,
            "ci_upper": hi,
        })

    # Sort: model A-Z, then mode in canonical order (D1, D2, D3, D4)
    mode_order = ["D1_full", "D2_no_code", "D3_text_only", "D4_abbreviated"]
    rows.sort(key=lambda r: (r["model"], _safe_index(r["mode"], mode_order)))

    out = pd.DataFrame(rows)

    lines = ["| Model | Mode | n | Hallucinations | Rate (95% CI) |"]
    lines.append("|---|---|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['mode']} | {r['n']} | {r['n_hallucinations']} | "
            f"{_format_pct_with_ci(r['rate'], r['ci_lower'], r['ci_upper'])} |"
        )
    return out, "\n".join(lines)


def cost_per_correct_table(
    df: pd.DataFrame,
    pricing: PricingTable | None = None,
) -> tuple[pd.DataFrame, str]:
    """Per-model cost-per-exact-match for the LLM models.

    Cost source priority:
      1. Per-row `cost_usd` column already in the CSV (set by runner
         when pricing was passed at evaluate time)
      2. Recomputed from token counts if `pricing` is provided here
      3. None (skipped) otherwise

    Models with zero LLM cost (baselines, retrieval, classifier) get
    rate `$0.00` per correct prediction, which is informative for the
    deployment-economics framing.
    """
    top1 = _top1_only(df)
    rows: list[dict[str, Any]] = []

    for model, grp in top1.groupby("model_name"):
        n = len(grp)
        n_exact = int((grp["best_top1"] == Outcome.EXACT_MATCH.value).sum())

        total_cost = _resolve_total_cost(grp, model, pricing)

        if total_cost is None:
            cost_per_correct = None
            cost_per_call = None
        else:
            cost_per_call = total_cost / n if n else 0.0
            cost_per_correct = total_cost / n_exact if n_exact else float("inf")

        rows.append({
            "model": model,
            "n": n,
            "n_exact": n_exact,
            "total_cost_usd": total_cost,
            "cost_per_call_usd": cost_per_call,
            "cost_per_correct_usd": cost_per_correct,
        })

    rows.sort(key=lambda r: r["model"])
    out = pd.DataFrame(rows)

    lines = ["| Model | n | Exact match | Total cost | $/call | $/correct |"]
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        cpc = r["cost_per_correct_usd"]
        cpc_str = "—" if cpc is None else ("∞" if cpc == float("inf") else _format_usd(cpc))
        lines.append(
            f"| {r['model']} | {r['n']} | {r['n_exact']} | "
            f"{_format_usd(r['total_cost_usd'])} | "
            f"{_format_usd(r['cost_per_call_usd'])} | "
            f"{cpc_str} |"
        )
    return out, "\n".join(lines)


def top_k_lift_table(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Top-1 vs top-5 exact-match accuracy by model.

    Surfaces whether a model's first prediction is reliably right
    (top-1) or whether it just generally has the right answer in the
    top-K (top-5). For deployment, top-1 matters most; top-5 is the
    upper bound a well-tuned ranker / human-in-the-loop reviewer
    could exploit.
    """
    top1 = _top1_only(df)
    rows: list[dict[str, Any]] = []
    for model, grp in top1.groupby("model_name"):
        n = len(grp)
        n_top1 = int((grp["best_top1"] == Outcome.EXACT_MATCH.value).sum())
        n_top5 = int((grp["best_top5"] == Outcome.EXACT_MATCH.value).sum())
        rate1 = n_top1 / n if n else 0.0
        rate5 = n_top5 / n if n else 0.0
        lift = rate5 - rate1
        rows.append({
            "model": model,
            "n": n,
            "top1_exact": n_top1,
            "top5_exact": n_top5,
            "top1_rate": rate1,
            "top5_rate": rate5,
            "lift_pp": lift,  # absolute percentage-point lift
        })
    rows.sort(key=lambda r: r["model"])
    out = pd.DataFrame(rows)

    lines = ["| Model | n | Top-1 | Top-5 | Lift (pp) |"]
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['n']} | "
            f"{_format_pct(r['top1_rate'])} | {_format_pct(r['top5_rate'])} | "
            f"{r['lift_pp'] * 100:+.1f}pp |"
        )
    return out, "\n".join(lines)


def per_bucket_cost_table(
    df: pd.DataFrame,
    pricing: PricingTable | None = None,
) -> tuple[pd.DataFrame, str]:
    """Cost decomposition by outcome bucket — addresses BACKLOG line 292.

    For each (model, outcome) cell: count × $/call → $ spent on that
    bucket. Lets the paper distinguish "$X spent on hallucinations"
    from "$Y spent on correct answers" — useful for the cost-economics
    framing where hallucinations have negative deployment value.
    """
    top1 = _top1_only(df)
    rows: list[dict[str, Any]] = []
    for (model, outcome), grp in top1.groupby(["model_name", "best_top1"]):
        n = len(grp)
        cost = _resolve_total_cost(grp, model, pricing)
        rows.append({
            "model": model,
            "outcome": outcome,
            "n": n,
            "total_cost_usd": cost,
            "cost_per_call_usd": (cost / n) if (cost is not None and n) else None,
        })

    # Stable ordering: model A-Z, then outcome in canonical taxonomy order.
    outcome_order = [o.value for o in Outcome]
    rows.sort(key=lambda r: (r["model"], _safe_index(r["outcome"], outcome_order)))
    out = pd.DataFrame(rows)

    lines = ["| Model | Outcome | n | Total cost | $/call |"]
    lines.append("|---|---|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['outcome']} | {r['n']} | "
            f"{_format_usd(r['total_cost_usd'])} | "
            f"{_format_usd(r['cost_per_call_usd'])} |"
        )
    return out, "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Top-level write_report orchestration
# ─────────────────────────────────────────────────────────────────────────


def write_report(
    df: pd.DataFrame,
    out_dir: str | Path,
    pricing: PricingTable | None = None,
    fmt: str = "markdown",
) -> dict[str, Path]:
    """Generate all five tables and persist to `out_dir`.

    Args:
        df: long-format prediction DataFrame from runner / loaded from CSV.
        out_dir: directory to write to (created if absent).
        pricing: optional PricingTable for cost recomputation if rows
            don't already have `cost_usd`.
        fmt: "markdown" (default), "csv", or "latex". Markdown writes
            a single combined `headline.md` plus per-table CSVs as
            companions (so the data is always available even when the
            primary output is markdown).

    Returns:
        Mapping {table_name: path} for everything written.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    tables = {
        "headline": headline_table(df),
        "hallucination": hallucination_table(df),
        "cost_per_correct": cost_per_correct_table(df, pricing),
        "top_k_lift": top_k_lift_table(df),
        "per_bucket_cost": per_bucket_cost_table(df, pricing),
    }

    # Per-table CSVs always written (cheap; useful for downstream analyses).
    for name, (table_df, _md) in tables.items():
        csv_path = out_path / f"{name}.csv"
        table_df.to_csv(csv_path, index=False)
        written[f"{name}_csv"] = csv_path

    # Combined markdown report — primary deliverable for paper §3.
    if fmt in ("markdown", "csv", "latex"):
        md_lines = [
            "# Phantom Codes — evaluation report",
            "",
            "Auto-generated by `phantom-codes report`. Drop these tables into",
            "`paper/sections/03_results.md` to replace the [TBD] placeholders.",
            "",
            "## Headline outcome distribution (per model × mode)",
            "",
            tables["headline"][1],
            "",
            "## Hallucination rate (with 95% Wilson CI)",
            "",
            tables["hallucination"][1],
            "",
            "## Top-1 vs Top-5 exact-match lift",
            "",
            tables["top_k_lift"][1],
            "",
            "## Cost per correct prediction (LLM economics)",
            "",
            tables["cost_per_correct"][1],
            "",
            "## Cost decomposition by outcome bucket",
            "",
            tables["per_bucket_cost"][1],
            "",
        ]
        md_path = out_path / "headline.md"
        md_path.write_text("\n".join(md_lines))
        written["headline_md"] = md_path

    return written


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────


def _resolve_total_cost(
    grp: pd.DataFrame,
    model: str,
    pricing: PricingTable | None,
) -> float | None:
    """Compute total $ for a per-model group of rank-0 rows.

    Priority: per-row cost_usd column (already computed at evaluate
    time) > recompute from tokens × pricing > None (skip).
    """
    if "cost_usd" in grp.columns:
        col_total = grp["cost_usd"].fillna(0).sum()
        if col_total > 0:
            return float(col_total)

    # Recompute from tokens if pricing is provided
    if pricing is None:
        return None
    model_pricing = resolve_pricing_for_model(model, pricing)
    if model_pricing is None:
        return None

    total = 0.0
    for _, row in grp.iterrows():
        cost = compute_call_cost(
            input_tokens=int(row.get("input_tokens", 0) or 0),
            output_tokens=int(row.get("output_tokens", 0) or 0),
            cache_read_tokens=int(row.get("cache_read_tokens", 0) or 0),
            cache_creation_tokens=int(row.get("cache_creation_tokens", 0) or 0),
            pricing=model_pricing,
        )
        total += cost
    return total


def _safe_index(value: str, order: list[str]) -> int:
    """Return the index of value in order, or len(order) if missing
    (so unknown values sort last in a stable position)."""
    try:
        return order.index(value)
    except ValueError:
        return len(order)
