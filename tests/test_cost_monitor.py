"""Tests for the running-cost monitor + budget cap.

Pure-Python unit tests; no API calls, no I/O, instant.
"""

from __future__ import annotations

import pytest

from phantom_codes.eval.cost_monitor import BudgetExceededError, CostMonitor


def test_no_budget_never_aborts() -> None:
    """budget_usd=None → tracks cost but never raises."""
    monitor = CostMonitor(budget_usd=None)
    for _ in range(1000):
        monitor.add(1.0)
    assert monitor.running_cost == 1000.0


def test_under_budget_does_not_abort() -> None:
    monitor = CostMonitor(budget_usd=100.0)
    for _ in range(50):
        monitor.add(1.0)
    assert monitor.running_cost == 50.0


def test_exceeding_cap_raises() -> None:
    monitor = CostMonitor(budget_usd=10.0)
    monitor.add(5.0)
    monitor.add(4.0)
    with pytest.raises(BudgetExceededError, match="exceeded"):
        monitor.add(2.0)  # total 11.0 > 10.0


def test_exact_cap_does_not_raise() -> None:
    """At the cap is OK; only strictly-over fires the abort."""
    monitor = CostMonitor(budget_usd=10.0)
    monitor.add(10.0)  # exactly at cap
    assert monitor.running_cost == 10.0


def test_zero_and_none_costs_ignored() -> None:
    monitor = CostMonitor(budget_usd=100.0)
    monitor.add(None)  # free (baseline)
    monitor.add(0.0)  # free
    monitor.add(-1.0)  # weird; treated as free
    assert monitor.running_cost == 0.0


def test_soft_warnings_fire_in_order() -> None:
    warns: list[str] = []
    monitor = CostMonitor(budget_usd=100.0, warn=warns.append)

    monitor.add(40.0)  # 40% — no warn
    assert warns == []

    monitor.add(15.0)  # 55% — fires 50% warn
    assert len(warns) == 1
    assert "50%" in warns[0]

    monitor.add(25.0)  # 80% — fires 75% warn
    assert len(warns) == 2
    assert "75%" in warns[1]

    monitor.add(15.0)  # 95% — fires 90% warn
    assert len(warns) == 3
    assert "90%" in warns[2]


def test_soft_warnings_fire_only_once_per_threshold() -> None:
    warns: list[str] = []
    monitor = CostMonitor(budget_usd=100.0, warn=warns.append)

    # Cross all thresholds in one call
    monitor.add(95.0)
    assert len(warns) == 3  # 50/75/90 all fire once

    # Subsequent adds shouldn't re-fire
    monitor.add(2.0)
    assert len(warns) == 3


def test_status_string_with_budget() -> None:
    monitor = CostMonitor(budget_usd=100.0)
    monitor.add(50.0)
    s = monitor.status()
    assert "$50.00" in s
    assert "$100.00" in s
    assert "50.0%" in s


def test_status_string_no_budget() -> None:
    monitor = CostMonitor(budget_usd=None)
    monitor.add(42.0)
    s = monitor.status()
    assert "$42.00" in s
    assert "no budget" in s


def test_fraction_used() -> None:
    monitor = CostMonitor(budget_usd=200.0)
    assert monitor.fraction_used == 0.0
    monitor.add(50.0)
    assert monitor.fraction_used == 0.25
    monitor.add(150.0)
    assert monitor.fraction_used == 1.0


def test_fraction_used_no_budget() -> None:
    monitor = CostMonitor(budget_usd=None)
    monitor.add(100.0)
    assert monitor.fraction_used is None


def test_custom_warn_thresholds() -> None:
    warns: list[str] = []
    monitor = CostMonitor(
        budget_usd=100.0,
        soft_warn_pcts=[0.25, 0.5],
        warn=warns.append,
    )
    monitor.add(30.0)
    assert len(warns) == 1
    assert "25%" in warns[0]
    monitor.add(25.0)
    assert len(warns) == 2
    assert "50%" in warns[1]


def test_running_cost_preserved_at_abort() -> None:
    """When budget abort fires, running_cost reflects the spend that
    triggered it (so the catching code can log accurate totals)."""
    monitor = CostMonitor(budget_usd=10.0)
    monitor.add(8.0)
    try:
        monitor.add(5.0)
    except BudgetExceededError:
        pass
    assert monitor.running_cost == 13.0
