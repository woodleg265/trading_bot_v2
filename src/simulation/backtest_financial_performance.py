from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from src.simulation.backtest_events import BacktestEvent
from src.simulation.execution_model import ExecutionModel
from src.simulation.models import VaultedLot


@dataclass(frozen=True)
class BacktestEquityPoint:
    sequence: int
    equity: Decimal
    peak_equity: Decimal
    drawdown: Decimal


@dataclass(frozen=True)
class BacktestFinancialPerformanceReport:
    """Active-strategy-only candle-close financial performance."""

    starting_equity: Decimal
    ending_equity: Decimal
    net_pnl: Decimal
    return_rate: Decimal
    maximum_drawdown: Decimal
    equity_curve: tuple[BacktestEquityPoint, ...]


@dataclass(frozen=True)
class VaultedInventoryMark:
    """Final candle-close valuation of one vaulted inventory lot."""

    symbol: str
    quantity: Decimal
    reference_price: Decimal
    market_value: Decimal


@dataclass(frozen=True)
class ConsolidatedBacktestFinancialPerformanceReport:
    """Final economic value across active strategy capital and vaulted lots."""

    active_performance: BacktestFinancialPerformanceReport
    vaulted_inventory_marks: tuple[VaultedInventoryMark, ...]
    vaulted_inventory_value: Decimal
    starting_equity: Decimal
    ending_equity: Decimal
    net_pnl: Decimal
    return_rate: Decimal


def build_backtest_financial_performance_report(
    events: Sequence[BacktestEvent],
    *,
    execution_model: ExecutionModel,
    starting_equity: Decimal,
) -> BacktestFinancialPerformanceReport:
    """Build an immutable active-only candle-close financial report."""

    del execution_model

    equity_points: list[BacktestEquityPoint] = []
    peak_equity = starting_equity
    maximum_drawdown = Decimal("0")

    for event in events:
        equity = event.cash_after + (
            event.active_quantity_after * event.reference_price
        )

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


def build_consolidated_backtest_financial_performance_report(
    events: Sequence[BacktestEvent],
    *,
    execution_model: ExecutionModel,
    starting_equity: Decimal,
    vaulted_lots: Sequence[VaultedLot],
    strategy_symbol: str,
) -> ConsolidatedBacktestFinancialPerformanceReport:
    """Build final active-plus-vaulted economic performance.

    Vaulted lots are valued at the final available replay candle-close price.
    A vault lot for another symbol is rejected because this backtest event
    stream has no independently auditable price series for that symbol.
    """

    active_performance = build_backtest_financial_performance_report(
        events,
        execution_model=execution_model,
        starting_equity=starting_equity,
    )

    if vaulted_lots and not events:
        raise ValueError(
            "Cannot value vaulted inventory without a final reference price."
        )

    final_reference_price = (
        events[-1].reference_price
        if events
        else Decimal("0")
    )

    vaulted_marks: list[VaultedInventoryMark] = []

    for vaulted_lot in vaulted_lots:
        if vaulted_lot.symbol != strategy_symbol:
            raise ValueError(
                "Cannot value vaulted inventory for symbol "
                f"{vaulted_lot.symbol!r}; expected {strategy_symbol!r}."
            )

        market_value = vaulted_lot.quantity * final_reference_price

        vaulted_marks.append(
            VaultedInventoryMark(
                symbol=vaulted_lot.symbol,
                quantity=vaulted_lot.quantity,
                reference_price=final_reference_price,
                market_value=market_value,
            )
        )

    immutable_vaulted_marks = tuple(vaulted_marks)
    vaulted_inventory_value = sum(
        (mark.market_value for mark in immutable_vaulted_marks),
        start=Decimal("0"),
    )
    ending_equity = (
        active_performance.ending_equity + vaulted_inventory_value
    )
    net_pnl = ending_equity - starting_equity
    return_rate = (
        Decimal("0")
        if starting_equity == Decimal("0")
        else net_pnl / starting_equity
    )

    return ConsolidatedBacktestFinancialPerformanceReport(
        active_performance=active_performance,
        vaulted_inventory_marks=immutable_vaulted_marks,
        vaulted_inventory_value=vaulted_inventory_value,
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        net_pnl=net_pnl,
        return_rate=return_rate,
    )
