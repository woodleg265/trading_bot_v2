from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.simulation.audit_writer import SimulationAuditWriter
from src.simulation.fill_model import SimulationFillConfig, SimulationFillModel
from src.simulation.ledger import SimulationLedger, SimulationLedgerState
from src.simulation.risk_engine import (
    SimulationRiskContext,
    SimulationRiskEngine,
    SimulationRiskLimits,
)
from src.simulation.runner import SimulationRunner
from src.simulation.strategy_adapter import (
    SimulationAdapterOutcome,
    SimulationStrategyAdapter,
    SimulationStrategyAdapterConfig,
)
from src.strategies.ema_regime_strategy import DecisionAction, StrategyDecision


NOW = datetime(2026, 10, 16, 14, 0, tzinfo=timezone.utc)


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
def adapter(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    runner = SimulationRunner(
        risk_engine=SimulationRiskEngine(
            SimulationRiskLimits(
                allowed_symbols=frozenset({"BTC/USD"}),
                max_order_notional=Decimal("1000"),
                max_position_notional=Decimal("1000"),
                max_total_exposure=Decimal("1000"),
                max_market_data_age=timedelta(minutes=1),
                estimated_fee_rate=Decimal("0"),
            )
        ),
        fill_model=SimulationFillModel(
            SimulationFillConfig(
                fee_rate=Decimal("0"),
                slippage_rate=Decimal("0"),
            )
        ),
        ledger=SimulationLedger(),
        audit_writer=SimulationAuditWriter(),
    )

    return SimulationStrategyAdapter(
        runner=runner,
        config=SimulationStrategyAdapterConfig(
            symbol="BTC/USD",
            entry_quantity=Decimal("2"),
            max_exit_quantity=Decimal("1"),
        ),
        state=SimulationLedgerState(
            cash_balance=Decimal("1000"),
            cash_currency="USD",
        ),
        risk_context=SimulationRiskContext(
            available_cash=Decimal("1000"),
            position_quantities={},
            current_total_exposure=Decimal("0"),
            processed_intent_ids=frozenset(),
        ),
        run_id="strategy-session-001",
    )


def test_enter_routes_to_runner_and_updates_session(adapter: SimulationStrategyAdapter) -> None:
    result = adapter.apply(decision(DecisionAction.ENTER))

    assert result.outcome is SimulationAdapterOutcome.EXECUTED
    assert len(result.run_results) == 1
    assert result.run_results[0].risk_decision.approved is True
    assert adapter.state.cash_balance == Decimal("800")
    assert adapter.state.position_quantities == {"BTC": Decimal("2")}
    assert adapter.risk_context.position_quantities == {"BTC/USD": Decimal("2")}


def test_exit_closes_position_in_configured_chunks(
    adapter: SimulationStrategyAdapter,
) -> None:
    adapter.apply(decision(DecisionAction.ENTER))

    result = adapter.apply(
        decision(
            DecisionAction.EXIT,
            timestamp="2026-10-16T14:00:30+00:00",
        )
    )

    assert result.outcome is SimulationAdapterOutcome.EXECUTED
    assert len(result.run_results) == 2
    assert adapter.state.cash_balance == Decimal("1000")
    assert dict(adapter.state.position_quantities) == {}
    assert dict(adapter.risk_context.position_quantities) == {}


def test_hold_has_no_order_or_state_transition(adapter: SimulationStrategyAdapter) -> None:
    before_state = adapter.state
    before_context = adapter.risk_context

    result = adapter.apply(decision(DecisionAction.HOLD))

    assert result.outcome is SimulationAdapterOutcome.NO_ACTION
    assert result.run_results == ()
    assert adapter.state == before_state
    assert adapter.risk_context == before_context


def test_duplicate_decision_is_not_sent_to_runner(
    adapter: SimulationStrategyAdapter,
) -> None:
    first = adapter.apply(decision(DecisionAction.ENTER))
    second = adapter.apply(decision(DecisionAction.ENTER))

    assert first.outcome is SimulationAdapterOutcome.EXECUTED
    assert second.outcome is SimulationAdapterOutcome.DUPLICATE
    assert len(adapter.events) == 2
    assert adapter.state.position_quantities == {"BTC": Decimal("2")}


def test_risk_rejection_preserves_ledger_but_records_intent(
    adapter: SimulationStrategyAdapter,
) -> None:
    adapter.config = SimulationStrategyAdapterConfig(
        symbol="BTC/USD",
        entry_quantity=Decimal("20"),
        max_exit_quantity=Decimal("1"),
    )

    result = adapter.apply(decision(DecisionAction.ENTER))

    assert result.outcome is SimulationAdapterOutcome.REJECTED
    assert result.run_results[0].risk_decision.code == "MAX_ORDER_NOTIONAL_EXCEEDED"
    assert adapter.state.cash_balance == Decimal("1000")
    assert len(adapter.risk_context.processed_intent_ids) == 1


def test_naive_candle_timestamp_is_rejected_before_runner(
    adapter: SimulationStrategyAdapter,
) -> None:
    result = adapter.apply(
        decision(
            DecisionAction.ENTER,
            timestamp="2026-10-16T14:00:00",
        )
    )

    assert result.outcome is SimulationAdapterOutcome.REJECTED
    assert result.run_results == ()
    assert adapter.state.cash_balance == Decimal("1000")
    assert adapter.risk_context.processed_intent_ids == frozenset()
