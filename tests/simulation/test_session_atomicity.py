"""Failure-injection tests for SimulationSession transition atomicity."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from src.simulation.fill_model import SimulationFillConfig
from src.simulation.risk_engine import SimulationRiskLimits
from src.simulation.session import SimulationSession, SimulationSessionConfig
from src.simulation.session_snapshot import SimulationSessionSnapshot
from src.simulation.strategy_adapter import (
    SimulationStrategyAdapterConfig,
)
from src.strategies.ema_regime_strategy import DecisionAction, StrategyDecision


@pytest.fixture
def config() -> SimulationSessionConfig:
    return SimulationSessionConfig(
        run_id="session-atomicity-001",
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


def decision(
    action: DecisionAction,
    *,
    timestamp: str,
    close: float = 100.0,
) -> StrategyDecision:
    return StrategyDecision(
        action=action,
        reason="failure injection test",
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


def capture_durable_state(session: SimulationSession) -> dict[str, object]:
    """Capture every observable execution and resumability state value."""

    return {
        "ledger_state": session.ledger_state,
        "risk_context": session.risk_context,
        "processed_decision_keys": session.processed_decision_keys,
        "vaulted_lots": session.vaulted_lots,
        "active_entry_price": session.active_entry_price,
        "active_entry_timestamp": session.active_entry_timestamp,
        "brawler_campaign": session.brawler_campaign,
        "events": session.events,
        "snapshot_json": SimulationSessionSnapshot.capture(session).to_json(),
    }


def assert_durable_state_unchanged(
    session: SimulationSession,
    before: dict[str, object],
) -> None:
    """Assert that an unexpected execution failure committed nothing."""

    assert session.ledger_state == before["ledger_state"]
    assert session.risk_context == before["risk_context"]
    assert session.processed_decision_keys == before["processed_decision_keys"]
    assert session.vaulted_lots == before["vaulted_lots"]
    assert session.active_entry_price == before["active_entry_price"]
    assert session.active_entry_timestamp == before["active_entry_timestamp"]
    assert session.brawler_campaign == before["brawler_campaign"]
    assert session.events == before["events"]
    assert (
        SimulationSessionSnapshot.capture(session).to_json()
        == before["snapshot_json"]
    )


def test_enter_failure_after_runner_result_is_fully_atomic(
    config: SimulationSessionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-run adapter failure must not debit cash or open inventory."""

    session = SimulationSession(config)
    before = capture_durable_state(session)

    original_process_intent = session._adapter._process_intent

    def fail_after_runner_result(**kwargs):
        original_process_intent(**kwargs)
        raise RuntimeError("injected failure after runner result")

    monkeypatch.setattr(
        session._adapter,
        "_process_intent",
        fail_after_runner_result,
    )

    with pytest.raises(RuntimeError, match="injected failure after runner result"):
        session.process_decision(
            decision(
                DecisionAction.ENTER,
                timestamp="2026-10-16T14:00:00+00:00",
            )
        )

    assert_durable_state_unchanged(session, before)


def test_exit_failure_during_second_fill_is_fully_atomic(
    config: SimulationSessionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-fill EXIT failure must not leave the first sell committed."""

    session = SimulationSession(config)

    session.process_decision(
        decision(
            DecisionAction.ENTER,
            timestamp="2026-10-16T14:00:00+00:00",
        )
    )
    before = capture_durable_state(session)

    original_process_intent = session._adapter._process_intent
    calls = 0

    def fail_on_second_intent(**kwargs):
        nonlocal calls
        calls += 1

        if calls == 2:
            raise RuntimeError("injected failure during second exit fill")

        return original_process_intent(**kwargs)

    monkeypatch.setattr(
        session._adapter,
        "_process_intent",
        fail_on_second_intent,
    )

    with pytest.raises(RuntimeError, match="injected failure during second exit fill"):
        session.process_decision(
            decision(
                DecisionAction.EXIT,
                timestamp="2026-10-16T14:01:00+00:00",
            )
        )

    assert calls == 2
    assert_durable_state_unchanged(session, before)
