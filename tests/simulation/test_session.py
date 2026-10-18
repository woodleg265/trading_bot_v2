from datetime import timedelta
from decimal import Decimal

import pytest

from src.simulation.fill_model import SimulationFillConfig
from src.simulation.ledger import SimulationLedgerState
from src.simulation.risk_engine import (
    SimulationRiskContext,
    SimulationRiskLimits,
)
from src.simulation.session import SimulationSession, SimulationSessionConfig
from src.simulation.strategy_adapter import (
    SimulationAdapterOutcome,
    SimulationStrategyAdapterConfig,
)
from src.strategies.ema_regime_strategy import DecisionAction, StrategyDecision


def decision(
    action: DecisionAction,
    *,
    timestamp: str = "2026-10-16T14:00:00+00:00",
    close: float | None = 100.0,
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


@pytest.fixture
def config() -> SimulationSessionConfig:
    return SimulationSessionConfig(
        run_id="simulation-session-001",
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


def test_session_initializes_isolated_virtual_state(
    config: SimulationSessionConfig,
) -> None:
    session = SimulationSession(config)

    assert session.run_id == "simulation-session-001"
    assert session.ledger_state.cash_balance == Decimal("1000")
    assert session.ledger_state.cash_currency == "USD"
    assert dict(session.ledger_state.position_quantities) == {}
    assert session.risk_context.available_cash == Decimal("1000")
    assert dict(session.risk_context.position_quantities) == {}
    assert session.processed_decision_keys == frozenset()
    assert session.events == ()


def test_session_processes_enter_through_strategy_adapter(
    config: SimulationSessionConfig,
) -> None:
    session = SimulationSession(config)

    result = session.process_decision(decision(DecisionAction.ENTER))

    assert result.outcome is SimulationAdapterOutcome.EXECUTED
    assert len(result.run_results) == 1
    assert session.ledger_state.cash_balance == Decimal("800")
    assert session.ledger_state.position_quantities == {"BTC": Decimal("2")}
    assert session.risk_context.position_quantities == {"BTC/USD": Decimal("2")}


def test_sessions_do_not_share_state_or_idempotency(
    config: SimulationSessionConfig,
) -> None:
    first = SimulationSession(config)
    second = SimulationSession(
        SimulationSessionConfig(
            run_id="simulation-session-002",
            strategy_adapter=config.strategy_adapter,
            risk_limits=config.risk_limits,
            fill_config=config.fill_config,
            initial_cash_balance=config.initial_cash_balance,
            cash_currency=config.cash_currency,
        )
    )

    first_result = first.process_decision(decision(DecisionAction.ENTER))
    second_result = second.process_decision(decision(DecisionAction.ENTER))

    assert first_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert second_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert first.ledger_state.cash_balance == Decimal("800")
    assert second.ledger_state.cash_balance == Decimal("800")
    assert len(first.processed_decision_keys) == 1
    assert len(second.processed_decision_keys) == 1
    assert first.processed_decision_keys != second.processed_decision_keys
    assert first.events != second.events


def test_hold_preserves_session_state(
    config: SimulationSessionConfig,
) -> None:
    session = SimulationSession(config)
    before_state = session.ledger_state
    before_context = session.risk_context

    result = session.process_decision(decision(DecisionAction.HOLD))

    assert result.outcome is SimulationAdapterOutcome.NO_ACTION
    assert session.ledger_state == before_state
    assert session.risk_context == before_context


def test_restoring_open_ledger_without_matching_risk_context_is_rejected(
    config: SimulationSessionConfig,
) -> None:
    restored_state = SimulationLedgerState(
        cash_balance=Decimal("800"),
        cash_currency="USD",
        position_quantities={"BTC": Decimal("2")},
    )

    with pytest.raises(ValueError, match="initial_risk_context is required"):
        SimulationSession(config, initial_ledger_state=restored_state)


def test_session_accepts_explicit_restored_ledger_and_risk_context(
    config: SimulationSessionConfig,
) -> None:
    restored_state = SimulationLedgerState(
        cash_balance=Decimal("800"),
        cash_currency="USD",
        position_quantities={"BTC": Decimal("2")},
    )
    restored_risk_context = SimulationRiskContext(
        available_cash=Decimal("800"),
        position_quantities={"BTC/USD": Decimal("2")},
        current_total_exposure=Decimal("200"),
        processed_intent_ids=frozenset({"previous-intent"}),
    )

    session = SimulationSession(
        config,
        initial_ledger_state=restored_state,
        initial_risk_context=restored_risk_context,
    )

    assert session.ledger_state == restored_state
    assert session.risk_context == restored_risk_context
