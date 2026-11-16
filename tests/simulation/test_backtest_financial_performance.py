"""Tests for immutable financial backtest performance reporting."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
from typing import Sequence

import pytest

from src.simulation.backtest_events import BacktestEvent
from src.simulation.backtest_financial_performance import (
    BacktestEquityPoint,
    BacktestFinancialPerformanceReport,
    build_backtest_financial_performance_report,
)
from src.simulation.execution_model import DEFAULT_EXECUTION_MODEL


def make_event(
    *,
    sequence: int,
    reference_price: Decimal = Decimal("100"),
    cash_before: Decimal = Decimal("1000"),
    cash_after: Decimal = Decimal("1000"),
    active_quantity_before: Decimal = Decimal("0"),
    active_quantity_after: Decimal = Decimal("0"),
    vaulted_quantity_before: Decimal = Decimal("0"),
    vaulted_quantity_after: Decimal = Decimal("0"),
    action: str = "HOLD",
    outcome: str = "NO_ACTION",
) -> BacktestEvent:
    """Create a minimal immutable event for financial projection tests."""

    return BacktestEvent(
        sequence=sequence,
        candle_timestamp=f"2025-01-0{sequence}T00:00:00+00:00",
        reference_price=reference_price,
        action=action,
        reason="test decision",
        outcome=outcome,
        risk_approved=None,
        risk_code=None,
        cash_before=cash_before,
        cash_after=cash_after,
        active_quantity_before=active_quantity_before,
        active_quantity_after=active_quantity_after,
        vaulted_quantity_before=vaulted_quantity_before,
        vaulted_quantity_after=vaulted_quantity_after,
        campaign_before="IDLE",
        campaign_after="IDLE",
    )


def build_report(
    events: tuple[BacktestEvent, ...] | list[BacktestEvent],
    *,
    starting_equity: Decimal = Decimal("1000"),
) -> BacktestFinancialPerformanceReport:
    """Build a report with the standard deterministic execution assumptions."""

    return build_backtest_financial_performance_report(
        events,
        execution_model=DEFAULT_EXECUTION_MODEL,
        starting_equity=starting_equity,
    )


def test_empty_ledger_preserves_starting_equity() -> None:
    report = build_report((), starting_equity=Decimal("1000"))

    assert report == BacktestFinancialPerformanceReport(
        starting_equity=Decimal("1000"),
        ending_equity=Decimal("1000"),
        net_pnl=Decimal("0"),
        return_rate=Decimal("0"),
        maximum_drawdown=Decimal("0"),
        equity_curve=(),
    )


def test_report_marks_active_inventory_at_each_event_reference_price() -> None:
    events = (
        make_event(
            sequence=1,
            reference_price=Decimal("100"),
            cash_after=Decimal("800"),
            active_quantity_after=Decimal("2"),
            action="ENTER",
            outcome="EXECUTED",
        ),
        make_event(
            sequence=2,
            reference_price=Decimal("125"),
            cash_before=Decimal("800"),
            cash_after=Decimal("800"),
            active_quantity_before=Decimal("2"),
            active_quantity_after=Decimal("2"),
        ),
    )

    report = build_report(events)

    assert report.ending_equity == Decimal("1050")
    assert report.net_pnl == Decimal("50")
    assert report.return_rate == Decimal("0.05")
    assert report.maximum_drawdown == Decimal("0")
    assert report.equity_curve == (
        BacktestEquityPoint(
            sequence=1,
            equity=Decimal("1000"),
            peak_equity=Decimal("1000"),
            drawdown=Decimal("0"),
        ),
        BacktestEquityPoint(
            sequence=2,
            equity=Decimal("1050"),
            peak_equity=Decimal("1050"),
            drawdown=Decimal("0"),
        ),
    )


def test_report_includes_vaulted_inventory_in_equity_marking() -> None:
    events = (
        make_event(
            sequence=1,
            reference_price=Decimal("100"),
            cash_after=Decimal("700"),
            vaulted_quantity_after=Decimal("3"),
            action="VAULT",
            outcome="EXECUTED",
        ),
        make_event(
            sequence=2,
            reference_price=Decimal("120"),
            cash_before=Decimal("700"),
            cash_after=Decimal("700"),
            vaulted_quantity_before=Decimal("3"),
            vaulted_quantity_after=Decimal("3"),
        ),
    )

    report = build_report(events)

    assert report.ending_equity == Decimal("1060")
    assert report.net_pnl == Decimal("60")
    assert report.return_rate == Decimal("0.06")
    assert report.equity_curve[-1] == BacktestEquityPoint(
        sequence=2,
        equity=Decimal("1060"),
        peak_equity=Decimal("1060"),
        drawdown=Decimal("0"),
    )


def test_report_marks_active_and_vaulted_inventory_together() -> None:
    events = (
        make_event(
            sequence=1,
            reference_price=Decimal("50"),
            cash_after=Decimal("500"),
            active_quantity_after=Decimal("4"),
            vaulted_quantity_after=Decimal("6"),
        ),
    )

    report = build_report(events)

    # 500 cash + (4 active + 6 vaulted) * 50 reference price.
    assert report.ending_equity == Decimal("1000")
    assert report.net_pnl == Decimal("0")
    assert report.return_rate == Decimal("0")


def test_report_tracks_peak_equity_and_maximum_drawdown() -> None:
    events = (
        make_event(
            sequence=1,
            reference_price=Decimal("100"),
            cash_after=Decimal("800"),
            active_quantity_after=Decimal("2"),
        ),
        make_event(
            sequence=2,
            reference_price=Decimal("150"),
            cash_before=Decimal("800"),
            cash_after=Decimal("800"),
            active_quantity_before=Decimal("2"),
            active_quantity_after=Decimal("2"),
        ),
        make_event(
            sequence=3,
            reference_price=Decimal("110"),
            cash_before=Decimal("800"),
            cash_after=Decimal("800"),
            active_quantity_before=Decimal("2"),
            active_quantity_after=Decimal("2"),
        ),
        make_event(
            sequence=4,
            reference_price=Decimal("130"),
            cash_before=Decimal("800"),
            cash_after=Decimal("800"),
            active_quantity_before=Decimal("2"),
            active_quantity_after=Decimal("2"),
        ),
    )

    report = build_report(events)

    assert report.equity_curve == (
        BacktestEquityPoint(
            sequence=1,
            equity=Decimal("1000"),
            peak_equity=Decimal("1000"),
            drawdown=Decimal("0"),
        ),
        BacktestEquityPoint(
            sequence=2,
            equity=Decimal("1100"),
            peak_equity=Decimal("1100"),
            drawdown=Decimal("0"),
        ),
        BacktestEquityPoint(
            sequence=3,
            equity=Decimal("1020"),
            peak_equity=Decimal("1100"),
            drawdown=Decimal("80"),
        ),
        BacktestEquityPoint(
            sequence=4,
            equity=Decimal("1060"),
            peak_equity=Decimal("1100"),
            drawdown=Decimal("40"),
        ),
    )
    assert report.maximum_drawdown == Decimal("80")


def test_report_uses_final_event_equity_for_pnl_and_return() -> None:
    events = (
        make_event(
            sequence=1,
            reference_price=Decimal("50"),
            cash_after=Decimal("400"),
            active_quantity_after=Decimal("10"),
        ),
        make_event(
            sequence=2,
            reference_price=Decimal("60"),
            cash_before=Decimal("400"),
            cash_after=Decimal("400"),
            active_quantity_before=Decimal("10"),
            active_quantity_after=Decimal("10"),
        ),
    )

    report = build_report(events, starting_equity=Decimal("800"))

    assert report.starting_equity == Decimal("800")
    assert report.ending_equity == Decimal("1000")
    assert report.net_pnl == Decimal("200")
    assert report.return_rate == Decimal("0.25")


def test_zero_starting_equity_has_zero_return_rate() -> None:
    events = (
        make_event(
            sequence=1,
            reference_price=Decimal("100"),
            cash_after=Decimal("0"),
            active_quantity_after=Decimal("1"),
        ),
    )

    report = build_report(events, starting_equity=Decimal("0"))

    assert report.ending_equity == Decimal("100")
    assert report.net_pnl == Decimal("100")
    assert report.return_rate == Decimal("0")


def test_equivalent_ledgers_produce_equal_financial_reports() -> None:
    first_events = (
        make_event(
            sequence=1,
            reference_price=Decimal("100"),
            cash_after=Decimal("800"),
            active_quantity_after=Decimal("2"),
        ),
        make_event(
            sequence=2,
            reference_price=Decimal("125"),
            cash_before=Decimal("800"),
            cash_after=Decimal("800"),
            active_quantity_before=Decimal("2"),
            active_quantity_after=Decimal("2"),
        ),
    )
    second_events = tuple(first_events)

    assert build_report(first_events) == build_report(second_events)


def test_report_and_equity_curve_are_immutable() -> None:
    report = build_report(
        (
            make_event(
                sequence=1,
                reference_price=Decimal("100"),
                cash_after=Decimal("900"),
                active_quantity_after=Decimal("1"),
            ),
        )
    )

    assert isinstance(report.equity_curve, tuple)

    with pytest.raises(FrozenInstanceError):
        report.ending_equity = Decimal("0")  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        report.equity_curve[0].equity = Decimal("0")  # type: ignore[misc]


def test_builder_does_not_mutate_the_supplied_event_ledger() -> None:
    events = [
        make_event(
            sequence=1,
            reference_price=Decimal("100"),
            cash_after=Decimal("800"),
            active_quantity_after=Decimal("2"),
        ),
        make_event(
            sequence=2,
            reference_price=Decimal("125"),
            cash_before=Decimal("800"),
            cash_after=Decimal("800"),
            active_quantity_before=Decimal("2"),
            active_quantity_after=Decimal("2"),
        ),
    ]
    original_events = tuple(events)

    build_report(events)

    assert tuple(events) == original_events
