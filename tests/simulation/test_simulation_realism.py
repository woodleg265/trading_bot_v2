from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.simulation.audit_writer import SimulationAuditWriter
from src.simulation.fill_model import SimulationFillConfig, SimulationFillModel
from src.simulation.ledger import SimulationLedger, SimulationLedgerState
from src.simulation.models import OrderIntent, Side
from src.simulation.risk_engine import (
    SimulationRiskContext,
    SimulationRiskEngine,
    SimulationRiskLimits,
)
from src.simulation.runner import SimulationRunner
from src.simulation.staged_audit_writer import StagedSimulationAuditWriter
from src.simulation.strategy_adapter import (
    SimulationAdapterOutcome,
    SimulationStrategyAdapter,
    SimulationStrategyAdapterConfig,
)
from src.strategies.ema_regime_strategy import DecisionAction, StrategyDecision


NOW = datetime(2026, 10, 16, 14, 0, tzinfo=timezone.utc)
ZERO = Decimal("0")


def make_decision(
    action: DecisionAction,
    *,
    timestamp: str,
    close: float = 100.0,
) -> StrategyDecision:
    return StrategyDecision(
        action=action,
        reason="simulation realism test",
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


def make_adapter(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    initial_cash: Decimal = Decimal("1000"),
    entry_quantity: Decimal = Decimal("2"),
    max_exit_quantity: Decimal = Decimal("1"),
    fee_rate: Decimal = ZERO,
    slippage_rate: Decimal = ZERO,
) -> SimulationStrategyAdapter:
    monkeypatch.chdir(tmp_path)

    audit_writer = StagedSimulationAuditWriter(SimulationAuditWriter())

    runner = SimulationRunner(
        risk_engine=SimulationRiskEngine(
            SimulationRiskLimits(
                allowed_symbols=frozenset({"BTC/USD"}),
                max_order_notional=Decimal("10000"),
                max_position_notional=Decimal("10000"),
                max_total_exposure=Decimal("10000"),
                max_market_data_age=timedelta(minutes=1),
                estimated_fee_rate=fee_rate,
            )
        ),
        fill_model=SimulationFillModel(
            SimulationFillConfig(
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
            )
        ),
        ledger=SimulationLedger(),
        audit_writer=audit_writer,
    )

    return SimulationStrategyAdapter(
        runner=runner,
        audit_writer=audit_writer,
        config=SimulationStrategyAdapterConfig(
            symbol="BTC/USD",
            entry_quantity=entry_quantity,
            max_exit_quantity=max_exit_quantity,
        ),
        state=SimulationLedgerState(
            cash_balance=initial_cash,
            cash_currency="USD",
        ),
        risk_context=SimulationRiskContext(
            available_cash=initial_cash,
            position_quantities={},
            current_total_exposure=ZERO,
            processed_intent_ids=frozenset(),
        ),
        run_id="simulation-realism",
    )


def test_frictionless_unchanged_price_round_trip_returns_starting_cash(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = make_adapter(tmp_path, monkeypatch)

    enter = adapter.apply(
        make_decision(
            DecisionAction.ENTER,
            timestamp="2026-10-16T14:00:00+00:00",
        )
    )
    exit_result = adapter.apply(
        make_decision(
            DecisionAction.EXIT,
            timestamp="2026-10-16T14:00:30+00:00",
        )
    )

    assert enter.outcome is SimulationAdapterOutcome.EXECUTED
    assert exit_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert adapter.state.cash_balance == Decimal("1000")
    assert dict(adapter.state.position_quantities) == {}
    assert dict(adapter.risk_context.position_quantities) == {}
    assert adapter.risk_context.current_total_exposure == ZERO


def test_fees_reduce_final_cash_against_frictionless_round_trip(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = make_adapter(
        tmp_path,
        monkeypatch,
        fee_rate=Decimal("0.01"),
    )

    adapter.apply(
        make_decision(
            DecisionAction.ENTER,
            timestamp="2026-10-16T14:00:00+00:00",
        )
    )
    exit_result = adapter.apply(
        make_decision(
            DecisionAction.EXIT,
            timestamp="2026-10-16T14:00:30+00:00",
        )
    )

    assert exit_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert adapter.state.cash_balance == Decimal("996")
    assert adapter.state.cash_balance < Decimal("1000")
    assert dict(adapter.state.position_quantities) == {}


def test_adverse_slippage_reduces_round_trip_cash_at_unchanged_market_price(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = make_adapter(
        tmp_path,
        monkeypatch,
        slippage_rate=Decimal("0.01"),
    )

    adapter.apply(
        make_decision(
            DecisionAction.ENTER,
            timestamp="2026-10-16T14:00:00+00:00",
        )
    )
    exit_result = adapter.apply(
        make_decision(
            DecisionAction.EXIT,
            timestamp="2026-10-16T14:00:30+00:00",
        )
    )

    assert exit_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert adapter.state.cash_balance == Decimal("996")
    assert adapter.state.cash_balance < Decimal("1000")
    assert dict(adapter.state.position_quantities) == {}


def test_successive_execution_results_have_continuous_portfolio_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = make_adapter(tmp_path, monkeypatch)

    enter = adapter.apply(
        make_decision(
            DecisionAction.ENTER,
            timestamp="2026-10-16T14:00:00+00:00",
        )
    )
    exit_result = adapter.apply(
        make_decision(
            DecisionAction.EXIT,
            timestamp="2026-10-16T14:00:30+00:00",
        )
    )

    buy_run = enter.run_results[0]
    first_sell_run, second_sell_run = exit_result.run_results

    assert buy_run.state.cash_balance == Decimal("800")
    assert buy_run.state.position_quantities == {"BTC": Decimal("2")}

    assert first_sell_run.state.cash_balance == Decimal("900")
    assert first_sell_run.state.position_quantities == {"BTC": Decimal("1")}

    assert second_sell_run.state.cash_balance == Decimal("1000")
    assert dict(second_sell_run.state.position_quantities) == {}

    assert first_sell_run.ledger_entry is not None
    assert second_sell_run.ledger_entry is not None
    assert first_sell_run.ledger_entry.sequence < second_sell_run.ledger_entry.sequence


def test_completed_transitions_never_produce_negative_cash_or_inventory(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = make_adapter(
        tmp_path,
        monkeypatch,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.002"),
    )

    results = (
        adapter.apply(
            make_decision(
                DecisionAction.ENTER,
                timestamp="2026-10-16T14:00:00+00:00",
            )
        ),
        adapter.apply(
            make_decision(
                DecisionAction.EXIT,
                timestamp="2026-10-16T14:00:30+00:00",
            )
        ),
    )

    for result in results:
        for run_result in result.run_results:
            assert run_result.state.cash_balance >= ZERO
            assert all(
                quantity >= ZERO
                for quantity in run_result.state.position_quantities.values()
            )

    assert adapter.state.cash_balance >= ZERO
    assert all(
        quantity >= ZERO
        for quantity in adapter.state.position_quantities.values()
    )
    assert all(lot.quantity > ZERO for lot in adapter.vaulted_lots)


def test_vault_preserves_cash_total_inventory_and_marked_equity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = make_adapter(tmp_path, monkeypatch)

    enter = adapter.apply(
        make_decision(
            DecisionAction.ENTER,
            timestamp="2026-10-16T14:00:00+00:00",
        )
    )
    assert enter.outcome is SimulationAdapterOutcome.EXECUTED

    mark_price = Decimal("125")
    cash_before = adapter.state.cash_balance
    active_quantity_before = adapter.state.position_quantities["BTC"]
    marked_equity_before = cash_before + (active_quantity_before * mark_price)

    vault = adapter.apply(
        make_decision(
            DecisionAction.VAULT,
            timestamp="2026-10-16T14:00:30+00:00",
            close=float(mark_price),
        )
    )

    assert vault.outcome is SimulationAdapterOutcome.EXECUTED
    assert vault.run_results == ()
    assert adapter.state.cash_balance == cash_before
    assert dict(adapter.state.position_quantities) == {}

    vaulted_quantity = sum(
        (lot.quantity for lot in adapter.vaulted_lots),
        start=ZERO,
    )
    marked_equity_after = adapter.state.cash_balance + (
        vaulted_quantity * mark_price
    )

    assert vaulted_quantity == active_quantity_before
    assert marked_equity_after == marked_equity_before


def test_unaffordable_buy_is_rejected_without_state_mutation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = make_adapter(
        tmp_path,
        monkeypatch,
        initial_cash=Decimal("100"),
        entry_quantity=Decimal("1"),
        fee_rate=Decimal("0.01"),
    )

    before_state = adapter.state
    before_context = adapter.risk_context

    result = adapter.apply(
        make_decision(
            DecisionAction.ENTER,
            timestamp="2026-10-16T14:00:00+00:00",
        )
    )

    assert result.outcome is SimulationAdapterOutcome.REJECTED
    assert result.run_results[0].risk_decision.code == "INSUFFICIENT_VIRTUAL_CASH"
    assert adapter.state == before_state
    assert adapter.risk_context.available_cash == before_context.available_cash
    assert adapter.state.cash_balance == Decimal("100")
    assert dict(adapter.state.position_quantities) == {}


def test_oversize_sell_is_rejected_without_state_mutation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    audit_writer = SimulationAuditWriter()
    runner = SimulationRunner(
        risk_engine=SimulationRiskEngine(
            SimulationRiskLimits(
                allowed_symbols=frozenset({"BTC/USD"}),
                max_order_notional=Decimal("10000"),
                max_position_notional=Decimal("10000"),
                max_total_exposure=Decimal("10000"),
                max_market_data_age=timedelta(minutes=1),
            )
        ),
        fill_model=SimulationFillModel(
            SimulationFillConfig(
                fee_rate=ZERO,
                slippage_rate=ZERO,
            )
        ),
        ledger=SimulationLedger(),
        audit_writer=audit_writer,
    )

    state = SimulationLedgerState(
        cash_balance=Decimal("500"),
        cash_currency="USD",
        position_quantities={"BTC": Decimal("1")},
    )
    context = SimulationRiskContext(
        available_cash=Decimal("500"),
        position_quantities={"BTC/USD": Decimal("1")},
        current_total_exposure=Decimal("100"),
        processed_intent_ids=frozenset(),
    )
    intent = OrderIntent(
        intent_id="oversize-sell",
        symbol="BTC/USD",
        side=Side.SELL,
        quantity=Decimal("2"),
        reference_price=Decimal("100"),
        market_data_timestamp=NOW,
        created_at=NOW,
    )

    result = runner.process_intent(
        intent,
        state=state,
        risk_context=context,
        run_id="oversize-sell-test",
        sequence=1,
        evaluated_at=NOW,
    )

    assert result.risk_decision.approved is False
    assert result.risk_decision.code == "INSUFFICIENT_SIMULATED_POSITION"
    assert result.fill is None
    assert result.ledger_entry is None
    assert result.state == state
    assert result.state.cash_balance == Decimal("500")
    assert result.state.position_quantities == {"BTC": Decimal("1")}
