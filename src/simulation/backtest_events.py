"""Immutable audit events emitted during Brawler historical replay."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.simulation.session import SimulationSession
from src.simulation.strategy_adapter import SimulationStrategyAdapterResult
from src.strategies.brawler_strategy import BrawlerEvaluation


ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class BacktestEvent:
    """Read-only record of one submitted Brawler evaluation and its outcome."""

    sequence: int
    candle_timestamp: str
    action: str
    reason: str
    outcome: str
    risk_approved: bool | None
    risk_code: str | None
    cash_before: Decimal
    cash_after: Decimal
    active_quantity_before: Decimal
    active_quantity_after: Decimal
    vaulted_quantity_before: Decimal
    vaulted_quantity_after: Decimal
    campaign_before: str
    campaign_after: str


@dataclass(frozen=True, slots=True)
class BacktestEventState:
    """Durable session values relevant to an audit event."""

    cash_balance: Decimal
    active_quantity: Decimal
    vaulted_quantity: Decimal
    campaign_state: str


def capture_backtest_event_state(
    session: SimulationSession,
) -> BacktestEventState:
    """Read reportable session state without mutating the session."""

    if not isinstance(session, SimulationSession):
        raise TypeError("session must be a SimulationSession")

    symbol = session.config.strategy_adapter.symbol
    base_asset = symbol.split("/", 1)[0]
    position_quantities = session.ledger_state.position_quantities

    # Existing replay reporting projects active quantity using the base asset.
    # Also support symbol-keyed ledger quantities used by execution internals.
    active_quantity = position_quantities.get(
        base_asset,
        position_quantities.get(symbol, ZERO),
    )

    vaulted_quantity = sum(
        (lot.quantity for lot in session.vaulted_lots),
        start=ZERO,
    )

    campaign = session.brawler_campaign
    campaign_state = (
        "ACTIVE"
        if campaign.is_active
        else "COOLDOWN"
        if campaign.cooldown_until is not None
        else "IDLE"
    )

    return BacktestEventState(
        cash_balance=session.ledger_state.cash_balance,
        active_quantity=active_quantity,
        vaulted_quantity=vaulted_quantity,
        campaign_state=campaign_state,
    )


def build_backtest_event(
    *,
    sequence: int,
    evaluation: BrawlerEvaluation,
    result: SimulationStrategyAdapterResult,
    state_before: BacktestEventState,
    state_after: BacktestEventState,
) -> BacktestEvent:
    """Build an immutable event from a completed session submission."""

    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise TypeError("sequence must be an integer")
    if sequence < 1:
        raise ValueError("sequence must be greater than zero")
    if not isinstance(evaluation, BrawlerEvaluation):
        raise TypeError("evaluation must be a BrawlerEvaluation")
    if not isinstance(result, SimulationStrategyAdapterResult):
        raise TypeError("result must be a SimulationStrategyAdapterResult")
    if not isinstance(state_before, BacktestEventState):
        raise TypeError("state_before must be a BacktestEventState")
    if not isinstance(state_after, BacktestEventState):
        raise TypeError("state_after must be a BacktestEventState")

    decision = evaluation.decision

    if decision.candle_timestamp is None:
        raise ValueError(
            "Backtest events require a submitted evaluation with a candle timestamp"
        )

    risk_approved: bool | None = None
    risk_code: str | None = None

    if result.run_results:
        risk_decision = result.run_results[0].risk_decision
        risk_approved = risk_decision.approved
        risk_code = risk_decision.code

    return BacktestEvent(
        sequence=sequence,
        candle_timestamp=decision.candle_timestamp,
        action=decision.action.value,
        reason=decision.reason,
        outcome=result.outcome.value,
        risk_approved=risk_approved,
        risk_code=risk_code,
        cash_before=state_before.cash_balance,
        cash_after=state_after.cash_balance,
        active_quantity_before=state_before.active_quantity,
        active_quantity_after=state_after.active_quantity,
        vaulted_quantity_before=state_before.vaulted_quantity,
        vaulted_quantity_after=state_after.vaulted_quantity,
        campaign_before=state_before.campaign_state,
        campaign_after=state_after.campaign_state,
    )
