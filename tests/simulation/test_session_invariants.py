from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from src.simulation.session import SimulationSession, SimulationSessionConfig
from src.simulation.session_snapshot import SimulationSessionSnapshot
from src.simulation.fill_model import SimulationFillConfig
from src.simulation.risk_engine import SimulationRiskLimits
from src.simulation.strategy_adapter import (
    SimulationAdapterOutcome,
    SimulationStrategyAdapterConfig,
)
from src.strategies.ema_regime_strategy import DecisionAction, StrategyDecision


@pytest.fixture
def config() -> SimulationSessionConfig:
    return SimulationSessionConfig(
        run_id="session-invariants-001",
        strategy_adapter=SimulationStrategyAdapterConfig(
            symbol="BTC/USD",
            entry_quantity=Decimal("2"),
            max_exit_quantity=Decimal("1"),
        ),
        risk_limits=SimulationRiskLimits(
            allowed_symbols=frozenset({"BTC/USD"}),
            max_order_notional=Decimal("1000"),
            max_position_notional=Decimal("1000"),
            max_total_exposure=Decimal("1000"),
            max_market_data_age=timedelta(minutes=1),
            estimated_fee_rate=Decimal("0"),
        ),
        fill_config=SimulationFillConfig(
            fee_rate=Decimal("0"),
            slippage_rate=Decimal("0"),
        ),
        initial_cash_balance=Decimal("1000"),
        cash_currency="USD",
    )


def get_decision(
    action: DecisionAction,
    timestamp: str,
    *,
    close: float = 100.0,
) -> StrategyDecision:
    return StrategyDecision(
        action=action,
        reason="test",
        candle_timestamp=timestamp,
        close=close,
        rsi=None,
        ema_fast=None,
        ema_slow=None,
        ema_200=None,
        adx=None,
        plus_di=None,
        minus_di=None,
    )


def assert_inventory_invariants(session: SimulationSession) -> None:
    symbol = session.config.strategy_adapter.symbol
    base_asset = symbol.split("/")[0]
    quantity = session.ledger_state.position_quantities.get(
        base_asset,
        Decimal("0"),
    )

    assert quantity >= 0, "Position quantity cannot be negative"

    if quantity > 0:
        assert (
            session.active_entry_price is not None
        ), "Open position missing active_entry_price"
        assert (
            session.active_entry_timestamp is not None
        ), "Open position missing active_entry_timestamp"
    else:
        assert (
            session.active_entry_price is None
        ), "Closed position has stale active_entry_price"
        assert (
            session.active_entry_timestamp is None
        ), "Closed position has stale active_entry_timestamp"


def test_lifecycle_invariants_sequence(config) -> None:
    session = SimulationSession(config)

    decisions = [
        get_decision(DecisionAction.ENTER, "2026-10-16T14:00:00+00:00"),
        get_decision(DecisionAction.HOLD, "2026-10-16T14:01:00+00:00"),
        get_decision(DecisionAction.VAULT, "2026-10-16T14:02:00+00:00"),
        get_decision(DecisionAction.ENTER, "2026-10-16T14:03:00+00:00"),
        get_decision(DecisionAction.EXIT, "2026-10-16T14:04:00+00:00"),
    ]

    for decision in decisions:
        session.process_decision(decision)
        assert_inventory_invariants(session)

    assert len(session.vaulted_lots) == 1
    assert session.ledger_state.position_quantities == {}


def test_vault_isolation(config) -> None:
    session = SimulationSession(config)

    session.process_decision(
        get_decision(DecisionAction.ENTER, "2026-10-16T14:00:00+00:00")
    )

    initial_cash = session.ledger_state.cash_balance
    initial_intent_ids = set(session.risk_context.processed_intent_ids)

    assert session.ledger_state.position_quantities != {}

    result = session.process_decision(
        get_decision(DecisionAction.VAULT, "2026-10-16T14:01:00+00:00")
    )

    assert result.outcome == SimulationAdapterOutcome.EXECUTED
    assert session.ledger_state.cash_balance == initial_cash
    assert session.risk_context.processed_intent_ids == initial_intent_ids
    assert len(session.vaulted_lots) == 1
    assert session.ledger_state.position_quantities == {}
    assert_inventory_invariants(session)


def test_duplicate_decision_is_idempotent(config) -> None:
    session = SimulationSession(config)

    decision = get_decision(
        DecisionAction.ENTER,
        "2026-10-16T14:00:00+00:00",
    )

    first_result = session.process_decision(decision)

    ledger_after_first = session.ledger_state
    risk_after_first = session.risk_context
    vaulted_lots_after_first = session.vaulted_lots
    entry_price_after_first = session.active_entry_price
    entry_timestamp_after_first = session.active_entry_timestamp

    second_result = session.process_decision(decision)

    assert first_result.outcome == SimulationAdapterOutcome.EXECUTED
    assert second_result.outcome != SimulationAdapterOutcome.EXECUTED

    assert session.ledger_state == ledger_after_first
    assert session.risk_context == risk_after_first
    assert session.vaulted_lots == vaulted_lots_after_first
    assert session.active_entry_price == entry_price_after_first
    assert session.active_entry_timestamp == entry_timestamp_after_first
    assert_inventory_invariants(session)


def test_risk_rejected_enter_is_atomic_except_for_intent_idempotency(config) -> None:
    # Force rejection through configuration rather than mutating the frozen
    # risk context. An ENTER at $100 for quantity 2 has $200 notional.
    risk_rejecting_config = replace(
        config,
        risk_limits=replace(
            config.risk_limits,
            max_order_notional=Decimal("1"),
        ),
    )
    session = SimulationSession(risk_rejecting_config)

    # The rejected intent should still be recorded for idempotency.

    ledger_before = session.ledger_state
    risk_before = session.risk_context
    vaulted_lots_before = session.vaulted_lots
    entry_price_before = session.active_entry_price
    entry_timestamp_before = session.active_entry_timestamp

    result = session.process_decision(
        get_decision(DecisionAction.ENTER, "2026-10-16T14:00:00+00:00")
    )

    assert result.outcome != SimulationAdapterOutcome.EXECUTED

    # A rejected entry must not mutate execution-side state.
    assert session.ledger_state == ledger_before
    assert session.vaulted_lots == vaulted_lots_before
    assert session.active_entry_price == entry_price_before
    assert session.active_entry_timestamp == entry_timestamp_before

    # The financial and position portions of the risk context remain unchanged.
    assert session.risk_context.available_cash == risk_before.available_cash
    assert (
        session.risk_context.position_quantities
        == risk_before.position_quantities
    )
    assert (
        session.risk_context.current_total_exposure
        == risk_before.current_total_exposure
    )

    # The rejected runner intent is intentionally retained for idempotency.
    assert (
        len(session.risk_context.processed_intent_ids)
        == len(risk_before.processed_intent_ids) + 1
    )

    assert_inventory_invariants(session)


def test_snapshot_fixed_point_determinism(config) -> None:
    session = SimulationSession(config)

    session.process_decision(
        get_decision(DecisionAction.ENTER, "2026-10-16T14:00:00+00:00")
    )
    session.process_decision(
        get_decision(DecisionAction.HOLD, "2026-10-16T14:01:00+00:00")
    )
    session.process_decision(
        get_decision(DecisionAction.VAULT, "2026-10-16T14:02:00+00:00")
    )
    session.process_decision(
        get_decision(DecisionAction.ENTER, "2026-10-16T14:03:00+00:00")
    )

    first_json = SimulationSessionSnapshot.capture(session).to_json()

    restored = SimulationSessionSnapshot.from_json(first_json).restore()

    second_json = SimulationSessionSnapshot.capture(restored).to_json()

    assert second_json == first_json
    assert_inventory_invariants(restored)
