#!/usr/bin/env python3
"""Live progress monitor for an in-flight headline run.

Strictly blinded: never reads `outcome`, `pred_code`, `pred_display`,
or any ground-truth column — same allowlist principle as
blinded_progress_check.py. Designed for the `watch` command:

    watch -n 10 'uv run python scripts/headline_run_livestat.py \\
        results/raw/headline_<utc>.csv 500 500'

Or one-shot:

    uv run python scripts/headline_run_livestat.py \\
        results/raw/headline_<utc>.csv 500 500

Args (positional):
    1: CSV path
    2: max-records (used as denominator for % complete)  [default 500]
    3: max-cost-usd (used for cap %)                     [default 500]

Reads CSV via the csv module (not awk) so embedded commas in fields
don't corrupt the cost calculation.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SAFE_COLUMNS = {
    "model_name", "resource_id", "mode", "pred_rank",
    "input_tokens", "output_tokens", "cache_read_tokens",
    "cache_creation_tokens", "latency_ms", "cost_usd",
    "error_type", "error_msg",
}


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv-path> [max-records] [max-cost]")
        return 1
    csv_path = Path(sys.argv[1])
    max_records = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    max_cost = float(sys.argv[3]) if len(sys.argv) > 3 else 500.0

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        print("(eval may still be loading models — first batch writes after record 1 completes ~2-3 min in)")
        return 0

    records = set()
    cost = 0.0
    last_model_modes: list[tuple[str, str]] = []
    n_rows = 0
    n_errors = 0

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            records.add(row.get("resource_id", ""))
            cost_str = row.get("cost_usd") or ""
            if cost_str:
                try:
                    cost += float(cost_str)
                except ValueError:
                    pass
            if row.get("error_type"):
                n_errors += 1
            last_model_modes.append((row.get("model_name", ""), row.get("mode", "")))

    # Process check
    try:
        out = subprocess.run(
            ["pgrep", "-f", "phantom-codes evaluate"],
            capture_output=True, text=True, timeout=2
        )
        proc_state = "alive" if out.stdout.strip() else "GONE — process exited"
    except Exception:
        proc_state = "?"

    # Last write age
    age_s = int(datetime.now().timestamp() - os.path.getmtime(csv_path))
    size_kb = os.path.getsize(csv_path) // 1024

    n = len(records)
    pct_records = 100.0 * n / max_records if n else 0.0
    proj_cost = cost * max_records / n if n else 0.0
    pct_cost = 100.0 * proj_cost / max_cost if max_cost else 0.0

    print("=" * 64)
    print(f"Phantom Codes — headline-run live monitor   {datetime.now():%H:%M:%S}")
    print("=" * 64)
    print(f"  CSV:              {csv_path}")
    print(f"  Process:          {proc_state}")
    print(f"  Records:          {n} / {max_records}  ({pct_records:.1f}%)")
    print(f"  Total rows:       {n_rows}  ({size_kb} KB)")
    print(f"  Errors so far:    {n_errors}")
    print(f"  Cost so far:      ${cost:.2f}")
    print(f"  Projected total:  ${proj_cost:.2f}  ({pct_cost:.0f}% of ${max_cost:.0f} cap)")
    print(f"  Last write:       {age_s}s ago")
    print()
    print("  Last 5 (model:mode) tuples written:")
    for m, mode in last_model_modes[-5:]:
        print(f"    {m:<46} {mode}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
