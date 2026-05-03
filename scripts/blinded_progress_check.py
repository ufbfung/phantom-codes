#!/usr/bin/env python3
"""Blinded structural spot-check for an in-flight headline evaluation run.

Reads ONLY structural/wiring fields from the per-prediction CSV — never
touches `outcome`, `pred_code`, `pred_display`, `pred_score`, `gt_code`,
`gt_display`, `gt_group`, `best_top1`, or `best_top5`. The intent is to
catch structural issues (the kind that took out gpt-5.5 in the aborted
2026-05-03 run) without surfacing the actual results that will populate
the paper.

Usage:
    uv run python scripts/blinded_progress_check.py \\
        --csv results/raw/headline_<timestamp>.csv \\
        --models-config configs/models.yaml \\
        --models-set headline_set \\
        --max-records 500 \\
        --max-cost-usd 500

Recommended cadence on a 500-record / ~12-hour run:

    EARLY  (~10% in, ~50 rec, ~1 hr)  — catastrophic-failure detector.
                                        Aggressive abort posture: <$10
                                        spent so far, aborting saves 90%
                                        of the budget.

    MID    (~50% in, ~250 rec, ~6 hr) — trend / cost-trajectory check.
                                        Abort if a new error type
                                        appears or projected cost > 80%
                                        of cap. Saves 50% of budget.

    LATE   (~75% in, ~375 rec, ~9 hr) — validation, not abort. By here
                                        finishing is cheaper than
                                        re-running; just confirm shape
                                        matches the mid checkpoint.

The same checks run every invocation; only the interpretation changes
across checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

# Strict allowlist of CSV columns this script may read. Anything that
# would surface model performance is excluded by construction.
SAFE_COLUMNS = frozenset({
    "model_name",
    "resource_id",
    "mode",
    "pred_rank",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "latency_ms",
    "cost_usd",
    "error_type",
    "error_msg",
})

UNSAFE_COLUMNS = frozenset({
    "outcome",
    "pred_code",
    "pred_display",
    "pred_score",
    "pred_system",
    "gt_code",
    "gt_display",
    "gt_group",
    "gt_system",
    "best_top1",
    "best_top5",
})

EXPECTED_MODES = ["D1_full", "D2_no_code", "D3_text_only", "D4_abbreviated"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=Path, help="Path to the in-flight headline CSV")
    p.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    p.add_argument("--models-set", default="headline_set")
    p.add_argument("--max-records", type=int, default=500, help="Cap from the running evaluate command")
    p.add_argument("--max-cost-usd", type=float, default=500.0, help="Hard cap from the running evaluate command")
    return p.parse_args()


def load_expected_models(config_path: Path, set_name: str) -> list[str]:
    """Names of every model the registered eval is expected to produce."""
    cfg = yaml.safe_load(config_path.read_text())
    return [entry["name"] for entry in cfg[set_name]]


def safe_float(s: str | None, default: float = 0.0) -> float:
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def safe_int(s: str | None, default: int = 0) -> int:
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        return default


def read_csv_blinded(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read CSV, returning only SAFE_COLUMNS values per row.

    Hard guard: if any UNSAFE column is referenced anywhere in this
    function's output, that's a bug — the script exits non-zero rather
    than risk surfacing a result.
    """
    rows: list[dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        unknown = set(header) - SAFE_COLUMNS - UNSAFE_COLUMNS
        if unknown:
            print(f"⚠ unexpected columns in CSV (not on safe/unsafe list): {sorted(unknown)}", file=sys.stderr)
        for row in reader:
            rows.append({k: row.get(k, "") for k in SAFE_COLUMNS})
    return rows, header


def fmt_status(ok: bool, warn_only: bool = False) -> str:
    if ok:
        return "[ok]"
    return "[WARN]" if warn_only else "[FAIL]"


def main() -> int:
    args = parse_args()
    csv_path: Path = args.csv

    if not csv_path.exists():
        print(f"[FAIL] CSV not found: {csv_path}", file=sys.stderr)
        return 2

    print("=" * 78)
    print("Phantom Codes — blinded headline-run progress check")
    print("=" * 78)
    print(f"  Run CSV:       {csv_path}")
    print(f"  Models set:    {args.models_set} (from {args.models_config})")
    print(f"  Cap:           {args.max_records} records, ${args.max_cost_usd:.0f} budget")
    print()

    expected_models = load_expected_models(args.models_config, args.models_set)
    rows, header = read_csv_blinded(csv_path)

    # Wall-clock + rate
    csv_mtime = os.path.getmtime(csv_path)
    csv_birth = csv_path.stat().st_birthtime
    elapsed_hr = max((csv_mtime - csv_birth) / 3600, 1e-6)
    last_write_age_s = datetime.now().timestamp() - csv_mtime

    # Aggregations
    records: set[str] = set()
    per_model_rows: Counter = Counter()
    per_model_mode_records: dict[tuple[str, str], set[str]] = defaultdict(set)
    per_model_errors: dict[str, Counter] = defaultdict(Counter)
    per_model_error_msgs: dict[str, list[str]] = defaultdict(list)
    per_model_latencies: dict[str, list[float]] = defaultdict(list)
    per_model_cost: Counter = Counter()
    per_model_tokens: dict[str, dict[str, int]] = defaultdict(
        lambda: {"in": 0, "out": 0, "cache_r": 0, "cache_w": 0}
    )
    total_cost = 0.0

    for row in rows:
        m = row["model_name"]
        mode = row["mode"]
        rid = row["resource_id"]
        per_model_rows[m] += 1
        records.add(rid)
        per_model_mode_records[(m, mode)].add(rid)

        err = row["error_type"]
        if err:
            per_model_errors[m][err] += 1
            if len(per_model_error_msgs[m]) < 1:
                msg = row["error_msg"] or ""
                per_model_error_msgs[m].append(msg[:240])

        # Tokens / latency / cost only on rank-0 (per cost.py convention)
        if row["pred_rank"] == "0":
            lat = safe_float(row["latency_ms"])
            if lat > 0:
                per_model_latencies[m].append(lat)
            for fld, key in [
                ("input_tokens", "in"),
                ("output_tokens", "out"),
                ("cache_read_tokens", "cache_r"),
                ("cache_creation_tokens", "cache_w"),
            ]:
                per_model_tokens[m][key] += safe_int(row[fld])
            c = safe_float(row["cost_usd"])
            total_cost += c
            per_model_cost[m] += c

    n_records = len(records)
    pct_complete = 100.0 * n_records / args.max_records
    rate = n_records / elapsed_hr
    proj_total_hr = args.max_records / max(rate, 1e-6)
    proj_remaining_hr = max(proj_total_hr - elapsed_hr, 0.0)
    proj_total_cost = total_cost * args.max_records / max(n_records, 1)

    # ────────────────────────────────────────────────────────────────────
    # Run summary
    # ────────────────────────────────────────────────────────────────────
    print(f"  Records done:       {n_records}/{args.max_records} ({pct_complete:.1f}%)")
    print(f"  Wall-clock:         {elapsed_hr:.2f} hr elapsed; rate {rate:.1f} rec/hr")
    print(f"  Projected total:    {proj_total_hr:.1f} hr  ({proj_remaining_hr:.1f} hr remaining)")
    print(f"  Cost so far:        ${total_cost:.2f} / ${args.max_cost_usd:.0f} cap")
    print(f"  Projected total:    ${proj_total_cost:.2f}  ({100*proj_total_cost/args.max_cost_usd:.0f}% of cap)")
    print(f"  Last CSV write:     {last_write_age_s:.0f}s ago "
          f"({'still running' if last_write_age_s < 120 else 'STALE — process may have exited'})")
    print()

    issues: list[str] = []

    # ────────────────────────────────────────────────────────────────────
    # Check 1: configured-vs-actual model presence
    # ────────────────────────────────────────────────────────────────────
    actual_models = set(per_model_rows.keys())
    expected_set = set(expected_models)
    missing = expected_set - actual_models
    extra = actual_models - expected_set
    ok = not missing and not extra
    print(f"{fmt_status(ok)} (1) Model presence — {len(actual_models)}/{len(expected_set)} configured models firing")
    if missing:
        print(f"       MISSING: {sorted(missing)}")
        issues.append(f"missing models: {sorted(missing)}")
    if extra:
        print(f"       EXTRA (not in {args.models_set}): {sorted(extra)}")
    print()

    # ────────────────────────────────────────────────────────────────────
    # Check 2: coverage matrix (records per model × mode)
    # ────────────────────────────────────────────────────────────────────
    coverage_counts = [
        len(per_model_mode_records.get((m, mode), set()))
        for m in expected_models for mode in EXPECTED_MODES
        if m in actual_models
    ]
    if coverage_counts:
        cov_min, cov_max = min(coverage_counts), max(coverage_counts)
        cov_skew = cov_max - cov_min
        # In a healthy record-first iteration, skew should be <= 1.
        ok = cov_skew <= 2
        print(f"{fmt_status(ok, warn_only=True)} (2) Coverage matrix — min={cov_min} max={cov_max} per (model,mode); skew={cov_skew}")
        if cov_skew > 2:
            issues.append(f"coverage skew {cov_skew} (some models far ahead of others)")
    else:
        print("[WARN] (2) Coverage matrix — no data yet")
    print()

    # ────────────────────────────────────────────────────────────────────
    # Check 3: errors per model
    # ────────────────────────────────────────────────────────────────────
    print("       (3) Errors per model:")
    any_error = False
    catastrophic = []
    elevated = []
    for m in sorted(actual_models):
        n_rows = per_model_rows[m]
        n_err = sum(per_model_errors[m].values())
        if n_err == 0:
            continue
        any_error = True
        rate_pct = 100.0 * n_err / n_rows
        types = ", ".join(f"{k}:{v}" for k, v in per_model_errors[m].most_common())
        sample_msg = per_model_error_msgs[m][0] if per_model_error_msgs[m] else ""
        marker = "🔴" if rate_pct >= 50 else ("🟡" if rate_pct >= 15 else "  ")
        print(f"          {marker} {m:<42} {n_err:>5}/{n_rows:<5} ({rate_pct:5.1f}%) — {types}")
        if sample_msg:
            print(f"             sample: {sample_msg[:160]}")
        if rate_pct >= 50:
            catastrophic.append((m, rate_pct))
        elif rate_pct >= 15:
            elevated.append((m, rate_pct))
    if not any_error:
        print("          [ok] no errors recorded across any model")
    if catastrophic:
        issues.append(f"catastrophic error rate (>=50%): {catastrophic}")
    if elevated:
        issues.append(f"elevated error rate (>=15%): {elevated}")
    print()

    # ────────────────────────────────────────────────────────────────────
    # Check 4: cost trajectory
    # ────────────────────────────────────────────────────────────────────
    cost_pct_of_cap = 100 * proj_total_cost / args.max_cost_usd
    if cost_pct_of_cap >= 100:
        ok, marker = False, "[FAIL]"
        issues.append(f"projected cost ${proj_total_cost:.0f} EXCEEDS cap ${args.max_cost_usd:.0f}")
    elif cost_pct_of_cap >= 80:
        ok, marker = False, "[WARN]"
        issues.append(f"projected cost ${proj_total_cost:.0f} is {cost_pct_of_cap:.0f}% of cap")
    else:
        ok, marker = True, "[ok]"
    print(f"{marker} (4) Cost trajectory — projecting ${proj_total_cost:.2f} of ${args.max_cost_usd:.0f} cap ({cost_pct_of_cap:.0f}%)")
    print("          per-model spend so far:")
    for m in sorted(per_model_cost.keys(), key=lambda k: -per_model_cost[k]):
        if per_model_cost[m] > 0:
            print(f"            {m:<42} ${per_model_cost[m]:>7.4f}")
    print()

    # ────────────────────────────────────────────────────────────────────
    # Check 5: latency p50/p95 (anomaly hint — slow throttling)
    # ────────────────────────────────────────────────────────────────────
    print("       (5) Latency p50 / p95 per model (rank-0 only, ms):")
    for m in sorted(actual_models):
        lat = sorted(per_model_latencies.get(m, []))
        if not lat:
            continue
        p50 = lat[len(lat) // 2]
        p95 = lat[int(len(lat) * 0.95)]
        # Heuristic anomaly: p95 > 30s suggests heavy throttling on an LLM call
        marker = "🟡" if p95 > 30000 and "baseline" not in m and "classifier" not in m and "retrieval" not in m else "  "
        print(f"          {marker} {m:<42} p50={p50:>9.1f}  p95={p95:>9.1f}  n={len(lat)}")
        if p95 > 30000 and "baseline" not in m and "classifier" not in m and "retrieval" not in m:
            issues.append(f"high latency on {m}: p95={p95:.0f}ms (possible throttling)")
    print()

    # ────────────────────────────────────────────────────────────────────
    # Check 6: cache behavior (Anthropic prompt caching)
    # ────────────────────────────────────────────────────────────────────
    print("       (6) Anthropic prompt-caching status:")
    for m in sorted(actual_models):
        if "claude" not in m:
            continue
        t = per_model_tokens[m]
        if t["cache_r"] > 0 or t["cache_w"] > 0:
            ratio = 100.0 * t["cache_r"] / max(1, t["in"] + t["cache_r"])
            print(f"          [ok] {m:<42} cache_read={t['cache_r']:>8}  cache_write={t['cache_w']:>6}  ({ratio:.1f}% of input)")
        else:
            # Not an issue; this is the documented sub-threshold case.
            print(f"          [no cache] {m:<42} sub-threshold prompt (documented; not an issue)")
    print()

    # ────────────────────────────────────────────────────────────────────
    # Verdict
    # ────────────────────────────────────────────────────────────────────
    print("=" * 78)
    if not issues:
        print("Verdict: [ok] CONTINUE — no structural issues detected")
        ret = 0
    else:
        print(f"Verdict: [WARN] REVIEW — {len(issues)} issue(s) flagged:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        ret = 1
    print()
    print("Decision criteria reminder:")
    print("  EARLY checkpoint (~10% / ~1 hr): abort if any model >50% errors or coverage broken")
    print("  MID   checkpoint (~50% / ~6 hr): abort if NEW error type or projected cost >80% of cap")
    print("  LATE  checkpoint (~75% / ~9 hr): validation only — finishing cheaper than re-running")
    print("=" * 78)
    return ret


if __name__ == "__main__":
    sys.exit(main())
