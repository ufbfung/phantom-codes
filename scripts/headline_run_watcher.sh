#!/usr/bin/env bash
# Background watcher for an in-flight headline evaluation run.
#
# Polls the streaming CSV every 2 min. When unique-record count crosses
# each of EARLY (50), MID (250), and LATE (375), runs the blinded
# spot-check tool and tee's its output to a `<csv-stem>.spotchecks.log`
# sidecar. Exits after the LATE checkpoint completes.
#
# Strictly blinded: defers all CSV reads to scripts/blinded_progress_check.py,
# which has the SAFE_COLUMNS allowlist guard. This wrapper only counts
# unique resource_ids (column 2) for poll-trigger purposes.
#
# Usage:
#     nohup bash scripts/headline_run_watcher.sh \
#         results/raw/headline_<utc>.csv > /dev/null 2>&1 &
#     jobs -l                                   # see the watcher PID
#     tail -f results/raw/headline_<utc>.spotchecks.log   # watch live
#
# Kill early:
#     kill <pid-from-jobs-l>

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <path-to-headline-csv>" >&2
    exit 1
fi

CSV="$1"
LOG="${CSV%.csv}.spotchecks.log"
THRESHOLDS=(50 250 375)
LABELS=(EARLY MID LATE)
POLL_SECS=120
# Eval may take ~30-60s to load 32 models before opening the CSV for
# the first streaming write. Wait up to 5 minutes for the file to
# appear before giving up.
CSV_WAIT_MAX_SECS=300

waited=0
while [ ! -f "$CSV" ]; do
    if [ "$waited" -ge "$CSV_WAIT_MAX_SECS" ]; then
        echo "CSV did not appear within ${CSV_WAIT_MAX_SECS}s: $CSV" >&2
        echo "Check that the evaluate process is still running." >&2
        exit 2
    fi
    sleep 5
    waited=$((waited + 5))
done

{
    echo "================================================================"
    echo "Spot-check watcher started at $(date)"
    echo "  Watching:   $CSV"
    echo "  Log:        $LOG"
    echo "  Thresholds: EARLY=50, MID=250, LATE=375 records"
    echo "  Poll:       every ${POLL_SECS}s"
    echo "  Watcher PID: $$"
    echo "================================================================"
    echo ""
} | tee -a "$LOG"

for i in 0 1 2; do
    threshold="${THRESHOLDS[$i]}"
    label="${LABELS[$i]}"

    while true; do
        if [ -f "$CSV" ]; then
            n=$(awk -F, 'NR>1 {print $2}' "$CSV" 2>/dev/null | sort -u | wc -l | tr -d ' ')
        else
            n=0
        fi
        if [ "$n" -ge "$threshold" ]; then
            break
        fi
        sleep "$POLL_SECS"
    done

    {
        echo "================================================================"
        echo "=== ${label} CHECKPOINT — ${n} unique records done at $(date) ==="
        echo "================================================================"
        uv run python scripts/blinded_progress_check.py \
            --csv "$CSV" \
            --models-config configs/models.yaml \
            --models-set headline_set \
            --max-records 500 --max-cost-usd 500 || true
        echo ""
    } | tee -a "$LOG"
done

{
    echo "================================================================"
    echo "All 3 checkpoints completed at $(date) — watcher exiting"
    echo "================================================================"
} | tee -a "$LOG"
