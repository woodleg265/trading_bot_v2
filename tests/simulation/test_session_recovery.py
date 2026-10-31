from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from src.simulation.fill_model import SimulationFillConfig
from src.simulation.risk_engine import SimulationRiskLimits
from src.simulation.session import SimulationSession, SimulationSessionConfig
from src.simulation.session_snapshot import SimulationSessionSnapshot
from src.simulation.strategy_adapter import (
    SimulationAdapterOutcome,
    SimulationStrategyAdapterConfig,
)
from src.strategies.ema_regime_strategy import DecisionAction, StrategyDecision


def decision(
    action: DecisionAction,
    *,
    timestamp: str,
    close: float,
) -> StrategyDecision:
    return StrategyDecision(
        action=action,
        reason="session recovery test",
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


@pytest.fixture
def config() -> SimulationSessionConfig:
    return SimulationSessionConfig(
        run_id="session-recovery-001",
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


def restore(session: SimulationSession) -> SimulationSession:
    snapshot = SimulationSessionSnapshot.capture(session)
    return SimulationSessionSnapshot.from_json(snapshot.to_json()).restore()


def assert_resumable_state_matches(
    uninterrupted: SimulationSession,
    restored: SimulationSession,
) -> None:
    uninterrupted_snapshot = SimulationSessionSnapshot.capture(uninterrupted)
    restored_snapshot = SimulationSessionSnapshot.capture(restored)

    assert restored.config == uninterrupted.config
    assert restored.ledger_state == uninterrupted.ledger_state
    assert restored.risk_context == uninterrupted.risk_context
    assert restored.processed_decision_keys == uninterrupted.processed_decision_keys
    assert restored.vaulted_lots == uninterrupted.vaulted_lots
    assert restored.active_entry_price == uninterrupted.active_entry_price
    assert restored.active_entry_timestamp == uninterrupted.active_entry_timestamp
    assert restored.brawler_campaign == uninterrupted.brawler_campaign
    assert restored_snapshot.to_dict() == uninterrupted_snapshot.to_dict()
    assert restored_snapshot.to_json() == uninterrupted_snapshot.to_json()


def test_enter_snapshot_restore_then_final_exit_matches_uninterrupted_session(
    config: SimulationSessionConfig,
) -> None:
    enter_decision = decision(
        DecisionAction.ENTER,
        timestamp="2026-10-16T14:00:00+00:00",
        close=100.0,
    )
    final_exit_decision = decision(
        DecisionAction.EXIT,
        timestamp="2026-10-16T14:05:00+00:00",
        close=110.0,
    )

    uninterrupted = SimulationSession(config)

    enter_result = uninterrupted.process_decision(enter_decision)
    assert enter_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert uninterrupted.ledger_state.position_quantities == {
        "BTC": Decimal("2"),
    }
    assert uninterrupted.active_entry_price == Decimal("100")
    assert uninterrupted.active_entry_timestamp is not None

    restored = restore(uninterrupted)

    assert restored.ledger_state == uninterrupted.ledger_state
    assert restored.risk_context == uninterrupted.risk_context
    assert restored.active_entry_price == uninterrupted.active_entry_price
    assert restored.active_entry_timestamp == uninterrupted.active_entry_timestamp

    uninterrupted_result = uninterrupted.process_decision(final_exit_decision)
    restored_result = restored.process_decision(final_exit_decision)

    assert uninterrupted_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert restored_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert restored_result == uninterrupted_result

    assert restored.ledger_state.position_quantities == {}
    assert restored.risk_context.position_quantities == {}
    assert restored.active_entry_price is None
    assert restored.active_entry_timestamp is None

    assert_resumable_state_matches(uninterrupted, restored)
