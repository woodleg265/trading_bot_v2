"""Immutable performance summaries derived from backtest audit events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.simulation.backtest_events import BacktestEvent


@dataclass(frozen=True, slots=True)
class BacktestPerformanceReport:
    """Non-financial deterministic metrics derived from a backtest ledger."""

    total_events: int
    submitted_decisions: int
    executed_decisions: int
    rejected_decisions: int
    duplicate_decisions: int

    enter_submissions: int
    vault_submissions: int
    exit_submissions: int

    executed_enters: int
    executed_vaults: int
    executed_exits: int

    rejections_by_reason: tuple[tuple[str, int], ...]


def build_backtest_performance_report(
    events: Sequence[BacktestEvent],
) -> BacktestPerformanceReport:
    """Build a pure immutable report from submitted backtest events.

    This report intentionally does not estimate fills, fees, slippage, P&L,
    win rate, profit factor, or drawdown. Those metrics require explicit
    execution and valuation assumptions.
    """

    total_events = 0
    submitted_decisions = 0
    executed_decisions = 0
    rejected_decisions = 0
    duplicate_decisions = 0

    enter_submissions = 0
    vault_submissions = 0
    exit_submissions = 0

    executed_enters = 0
    executed_vaults = 0
    executed_exits = 0

    rejection_counts: dict[str, int] = {}

    for event in events:
        if not isinstance(event, BacktestEvent):
            raise TypeError("events must contain only BacktestEvent instances")

        total_events += 1
        submitted_decisions += 1

        if event.action == "ENTER":
            enter_submissions += 1
        elif event.action == "VAULT":
            vault_submissions += 1
        elif event.action == "EXIT":
            exit_submissions += 1

        if event.outcome == "EXECUTED":
            executed_decisions += 1

            if event.action == "ENTER":
                executed_enters += 1
            elif event.action == "VAULT":
                executed_vaults += 1
            elif event.action == "EXIT":
                executed_exits += 1

        elif event.outcome == "REJECTED":
            rejected_decisions += 1

            if event.risk_code is not None:
                rejection_counts[event.risk_code] = (
                    rejection_counts.get(event.risk_code, 0) + 1
                )

        elif event.outcome == "DUPLICATE":
            duplicate_decisions += 1

    return BacktestPerformanceReport(
        total_events=total_events,
        submitted_decisions=submitted_decisions,
        executed_decisions=executed_decisions,
        rejected_decisions=rejected_decisions,
        duplicate_decisions=duplicate_decisions,
        enter_submissions=enter_submissions,
        vault_submissions=vault_submissions,
        exit_submissions=exit_submissions,
        executed_enters=executed_enters,
        executed_vaults=executed_vaults,
        executed_exits=executed_exits,
        rejections_by_reason=tuple(sorted(rejection_counts.items())),
    )
