"""Tests for immutable backtest performance reporting."""

from __future__ import annotations

from decimal import Decimal

from src.simulation.backtest_events import BacktestEvent
from src.simulation.backtest_performance import (
    BacktestPerformanceReport,
    build_backtest_performance_report,
)


def make_event(
    *,
    sequence: int,
    action: str = "ENTER",
    outcome: str = "EXECUTED",
    risk_code: str | None = None,
) -> BacktestEvent:
    """Create a minimal immutable event for report projection tests."""

    return BacktestEvent(
        sequence=sequence,
        candle_timestamp="2025-01-01T00:00:00+00:00",
        action=action,
        reason="test decision",
        outcome=outcome,
        risk_approved=outcome != "REJECTED",
        risk_code=risk_code,
        cash_before=Decimal("1000"),
        cash_after=Decimal("1000"),
        active_quantity_before=Decimal("0"),
        active_quantity_after=Decimal("0"),
        vaulted_quantity_before=Decimal("0"),
        vaulted_quantity_after=Decimal("0"),
        campaign_before="IDLE",
        campaign_after="IDLE",
    )


def test_empty_ledger_produces_zeroed_performance_report() -> None:
    report = build_backtest_performance_report(())

    assert report == BacktestPerformanceReport(
        total_events=0,
        submitted_decisions=0,
        executed_decisions=0,
        rejected_decisions=0,
        duplicate_decisions=0,
        enter_submissions=0,
        vault_submissions=0,
        exit_submissions=0,
        executed_enters=0,
        executed_vaults=0,
        executed_exits=0,
        rejections_by_reason=(),
    )


def test_report_counts_submitted_actions_and_executed_actions() -> None:
    events = (
        make_event(sequence=1, action="ENTER", outcome="EXECUTED"),
        make_event(sequence=2, action="VAULT", outcome="EXECUTED"),
        make_event(sequence=3, action="EXIT", outcome="EXECUTED"),
        make_event(sequence=4, action="ENTER", outcome="REJECTED"),
    )

    report = build_backtest_performance_report(events)

    assert report.total_events == 4
    assert report.submitted_decisions == 4
    assert report.enter_submissions == 2
    assert report.vault_submissions == 1
    assert report.exit_submissions == 1

    assert report.executed_decisions == 3
    assert report.executed_enters == 1
    assert report.executed_vaults == 1
    assert report.executed_exits == 1


def test_report_separates_executed_rejected_and_duplicate_outcomes() -> None:
    events = (
        make_event(sequence=1, action="ENTER", outcome="EXECUTED"),
        make_event(
            sequence=2,
            action="ENTER",
            outcome="REJECTED",
            risk_code="MAX_ORDER_NOTIONAL_EXCEEDED",
        ),
        make_event(sequence=3, action="ENTER", outcome="DUPLICATE"),
        make_event(sequence=4, action="ENTER", outcome="NO_ACTION"),
    )

    report = build_backtest_performance_report(events)

    assert report.submitted_decisions == 4
    assert report.executed_decisions == 1
    assert report.rejected_decisions == 1
    assert report.duplicate_decisions == 1
    assert report.rejections_by_reason == (
        ("MAX_ORDER_NOTIONAL_EXCEEDED", 1),
    )


def test_report_groups_rejections_by_reason_in_sorted_order() -> None:
    events = (
        make_event(
            sequence=1,
            outcome="REJECTED",
            risk_code="MAX_ORDER_NOTIONAL_EXCEEDED",
        ),
        make_event(
            sequence=2,
            outcome="REJECTED",
            risk_code="INSUFFICIENT_VIRTUAL_CASH",
        ),
        make_event(
            sequence=3,
            outcome="REJECTED",
            risk_code="MAX_ORDER_NOTIONAL_EXCEEDED",
        ),
        make_event(sequence=4, outcome="REJECTED"),
    )

    report = build_backtest_performance_report(events)

    assert report.rejected_decisions == 4
    assert report.rejections_by_reason == (
        ("INSUFFICIENT_VIRTUAL_CASH", 1),
        ("MAX_ORDER_NOTIONAL_EXCEEDED", 2),
    )


def test_equivalent_event_ledgers_produce_equal_reports() -> None:
    first_events = (
        make_event(sequence=1, action="ENTER", outcome="EXECUTED"),
        make_event(
            sequence=2,
            action="VAULT",
            outcome="REJECTED",
            risk_code="INSUFFICIENT_SIMULATED_POSITION",
        ),
    )
    second_events = tuple(first_events)

    assert (
        build_backtest_performance_report(first_events)
        == build_backtest_performance_report(second_events)
    )


def test_builder_does_not_mutate_supplied_events() -> None:
    events = [
        make_event(sequence=1, action="ENTER", outcome="EXECUTED"),
        make_event(
            sequence=2,
            action="VAULT",
            outcome="REJECTED",
            risk_code="INSUFFICIENT_SIMULATED_POSITION",
        ),
    ]
    original_events = tuple(events)

    build_backtest_performance_report(events)

    assert tuple(events) == original_events
