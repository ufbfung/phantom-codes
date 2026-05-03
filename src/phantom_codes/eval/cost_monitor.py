"""Running-cost tracker with soft-warn thresholds and a hard abort.

Wraps `evaluate_one()` in `runner.py` so the eval matrix can be paused
or aborted before running away on API spend. Conservative-by-default:
- Tracks running cost across all per-call writes
- Logs warnings at 50% / 75% / 90% of the configured budget
- Raises `BudgetExceededError` once the running cost exceeds the cap

Combined with `runner.py`'s incremental CSV writes (one batch per
record), an aborted run leaves all completed records' predictions
durable on disk — no work lost when the cap fires.

Compliance note: this module reads only aggregate per-call USD from
`runner.py`'s LLM responses. Never touches per-record predictions
directly. Safe to operate in any compliance regime.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


class BudgetExceededError(Exception):
    """Raised when running cost exceeds the configured hard cap.

    The exception message includes the spend numbers so the catching
    code can include them in its abort log + manifest sidecar.
    """


@dataclass
class CostMonitor:
    """Track running cost across `add()` calls; abort if cap exceeded.

    Args:
        budget_usd: hard ceiling. None disables the cap entirely (still
            tracks cost for reporting).
        soft_warn_pcts: thresholds (as fractions of `budget_usd`) at
            which to emit a warning. Default:
            [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]. Early thresholds
            (5/10/25%) give the user a chance to react if a run is
            spending faster than expected; later thresholds (50/75/90%)
            are the standard "approaching the cap" alerts.
        warn: callable receiving a warning string. Default writes to
            stderr with a `[cost-monitor]` prefix.

    Usage:
        monitor = CostMonitor(budget_usd=500.0)
        for record in records:
            for model in models:
                row_cost = run_one_call(...)
                monitor.add(row_cost)  # raises if budget hit

    The class is intentionally simple — single counter plus threshold
    fires. Heavier accounting (per-model breakdown, per-bucket cost)
    happens downstream in `report.py` after the run finishes.
    """

    budget_usd: float | None = None
    soft_warn_pcts: list[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
    )
    warn: Callable[[str], None] | None = None

    # Internal state
    _running_cost: float = 0.0
    _warned_at: set[float] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.warn is None:
            self.warn = _default_warn
        # Sort warn thresholds ascending so the warning order matches
        # spend progression.
        self.soft_warn_pcts = sorted(set(self.soft_warn_pcts))

    @property
    def running_cost(self) -> float:
        return self._running_cost

    @property
    def fraction_used(self) -> float | None:
        """Spend as a fraction of the budget, or None if no budget set."""
        if self.budget_usd is None or self.budget_usd <= 0:
            return None
        return self._running_cost / self.budget_usd

    def add(self, call_cost: float | None) -> None:
        """Add a single call's cost. None / 0 / negative are treated as
        free (baselines, retrieval, classifier — anything not API-billed).
        Raises BudgetExceededError if the running total now exceeds
        `budget_usd`."""
        if call_cost is None or call_cost <= 0:
            return
        self._running_cost += float(call_cost)

        if self.budget_usd is None:
            return

        # Soft warnings: emit each threshold once, in ascending order.
        if self.warn is not None:
            for pct in self.soft_warn_pcts:
                if pct in self._warned_at:
                    continue
                if self._running_cost >= pct * self.budget_usd:
                    self._warned_at.add(pct)
                    self.warn(
                        f"running cost ${self._running_cost:.2f} "
                        f"crossed {int(pct * 100)}% of budget "
                        f"${self.budget_usd:.2f}"
                    )

        # Hard cap.
        if self._running_cost > self.budget_usd:
            raise BudgetExceededError(
                f"running cost ${self._running_cost:.2f} exceeded "
                f"hard cap ${self.budget_usd:.2f}; aborting run "
                "(partial results preserved on disk)"
            )

    def status(self) -> str:
        """Human-readable summary for logging / manifest output."""
        if self.budget_usd is None:
            return f"running cost ${self._running_cost:.2f} (no budget cap)"
        pct = (self._running_cost / self.budget_usd) * 100 if self.budget_usd > 0 else 0
        return (
            f"running cost ${self._running_cost:.2f} / "
            f"cap ${self.budget_usd:.2f} ({pct:.1f}%)"
        )


def _default_warn(message: str) -> None:
    import sys

    print(f"[cost-monitor] {message}", file=sys.stderr)
