from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from src.simulation.backtest_events import BacktestEvent
from src.simulation.execution_model import ExecutionModel


@dataclass(frozen=True)
class BacktestEquityPoint:
    sequence: int
    equity: Decimal
    peak_equity: Decimal
    drawdown: Decimal


@dataclass(frozen=True)
class BacktestFinancialPerformanceReport:
    starting_equity: Decimal
    ending_equity: Decimal
    net_pnl: Decimal
    return_rate: Decimal
    maximum_drawdown: Decimal
    equity_curve: tuple[BacktestEquityPoint, ...]


def build_backtest_financial_performance_report(
    events: Sequence[BacktestEvent],
    *,
    execution_model: ExecutionModel,
    starting_equity: Decimal,
) -> BacktestFinancialPerformanceReport:
    """Build an immutable candle-close financial report from ledger events."""

    del execution_model  # Financial marking uses the recorded event state and price.

    equity_points: list[BacktestEquityPoint] = []
    peak_equity = starting_equity
    maximum_drawdown = Decimal("0")

    for event in events:
        inventory_quantity = (
            event.active_quantity_after + event.vaulted_quantity_after
        )
        equity = event.cash_after + (inventory_quantity * event.reference_price)

        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity
        maximum_drawdown = max(maximum_drawdown, drawdown)

        equity_points.append(
            BacktestEquityPoint(
                sequence=event.sequence,
                equity=equity,
                peak_equity=peak_equity,
                drawdown=drawdown,
            )
        )

    ending_equity = (
        equity_points[-1].equity
        if equity_points
        else starting_equity
    )
    net_pnl = ending_equity - starting_equity
    return_rate = (
        Decimal("0")
        if starting_equity == Decimal("0")
        else net_pnl / starting_equity
    )

    return BacktestFinancialPerformanceReport(
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        net_pnl=net_pnl,
        return_rate=return_rate,
        maximum_drawdown=maximum_drawdown,
        equity_curve=tuple(equity_points),
    )
