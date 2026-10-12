from decimal import Decimal

import pytest

from src.portfolio.allocation_ledger import AllocationLedger, IntentStatus
from src.portfolio.strategy_ledger_adapter import (
    AdapterOutcome,
    StrategyLedgerAdapter,
)
from src.strategies.ema_regime_strategy import DecisionAction, StrategyDecision


def make_ledger(**overrides: Decimal) -> AllocationLedger:
    values = {
        "allocated_cash": Decimal("1000"),
        "fee_rate": Decimal("0.001"),
        "max_order_notional": Decimal("1000"),
    }
    values.update(overrides)
    return AllocationLedger(**values)


def make_decision(
    action: DecisionAction,
    *,
    timestamp: str | None = "2026-10-11T00:00:00Z",
    close: float | None = 100.0,
) -> StrategyDecision:
    return StrategyDecision(
        action=action,
        reason="TEST_DECISION",
        candle_timestamp=timestamp,
        close=close,
        rsi=50.0,
        ema_fast=100.0,
        ema_slow=99.0,
        ema_200=98.0,
        adx=20.0,
        plus_di=25.0,
        minus_di=15.0,
    )


def make_adapter(**ledger_overrides: Decimal) -> StrategyLedgerAdapter:
    return StrategyLedgerAdapter(
        ledger=make_ledger(**ledger_overrides),
        strategy_id="ema-regime-btc-usd",
    )


def test_enter_fills_a_fee_aware_full_allocation() -> None:
    adapter = make_adapter()
    decision = make_decision(DecisionAction.ENTER, close=100.0)

    result = adapter.apply(decision)

    assert result.outcome is AdapterOutcome.EXECUTED
    assert result.intent is not None
    assert result.intent.status is IntentStatus.FILLED
    assert result.intent.quantity == Decimal("9.99000999")
    assert adapter.ledger.position.quantity == Decimal("9.99000999")
    assert adapter.ledger.available_cash == Decimal("0.000000001")
    assert adapter.ledger.trade_count == 1
    assert len(adapter.ledger.intents) == 1


def test_enter_respects_max_order_notional() -> None:
    adapter = make_adapter(max_order_notional=Decimal("250"))
    decision = make_decision(DecisionAction.ENTER, close=100.0)

    result = adapter.apply(decision)

    assert result.outcome is AdapterOutcome.EXECUTED
    assert result.intent is not None
    assert result.intent.quantity == Decimal("2.5")
    assert adapter.ledger.available_cash == Decimal("749.75")
    assert adapter.ledger.position.quantity == Decimal("2.5")


def test_exit_sells_the_entire_open_position() -> None:
    adapter = make_adapter()
    adapter.apply(make_decision(DecisionAction.ENTER, close=100.0))

    result = adapter.apply(
        make_decision(
            DecisionAction.EXIT,
            timestamp="2026-10-11T00:05:00Z",
            close=120.0,
        )
    )

    assert result.outcome is AdapterOutcome.EXECUTED
    assert result.intent is not None
    assert result.intent.status is IntentStatus.FILLED
    assert not adapter.ledger.position.is_open
    # Exit at 120 exceeds the 1000 notional cap, so it is split into two sells.
    assert adapter.ledger.trade_count == 3
    assert adapter.ledger.realized_pnl > Decimal("0")


@pytest.mark.parametrize(
    "action",
    [DecisionAction.HOLD, DecisionAction.BLOCKED],
)
def test_hold_and_blocked_never_create_intents(action: DecisionAction) -> None:
    adapter = make_adapter()

    result = adapter.apply(make_decision(action))

    assert result.outcome is AdapterOutcome.NO_ACTION
    assert adapter.ledger.intents == ()
    assert adapter.ledger.trade_count == 0


def test_duplicate_enter_does_not_create_a_second_order() -> None:
    adapter = make_adapter()
    decision = make_decision(DecisionAction.ENTER)

    first = adapter.apply(decision)
    second = adapter.apply(decision)

    assert first.outcome is AdapterOutcome.EXECUTED
    assert second.outcome is AdapterOutcome.DUPLICATE
    assert len(adapter.ledger.intents) == 1
    assert adapter.ledger.trade_count == 1


def test_enter_is_a_no_action_when_already_in_a_position() -> None:
    adapter = make_adapter()
    adapter.apply(make_decision(DecisionAction.ENTER))

    result = adapter.apply(
        make_decision(
            DecisionAction.ENTER,
            timestamp="2026-10-11T00:05:00Z",
            close=101.0,
        )
    )

    assert result.outcome is AdapterOutcome.NO_ACTION
    assert len(adapter.ledger.intents) == 1
    assert adapter.ledger.trade_count == 1


def test_exit_is_a_no_action_without_an_open_position() -> None:
    adapter = make_adapter()

    result = adapter.apply(make_decision(DecisionAction.EXIT))

    assert result.outcome is AdapterOutcome.NO_ACTION
    assert adapter.ledger.intents == ()
    assert adapter.ledger.trade_count == 0


@pytest.mark.parametrize(
    ("timestamp", "close"),
    [
        (None, 100.0),
        ("", 100.0),
        ("2026-10-11T00:00:00Z", None),
        ("2026-10-11T00:00:00Z", 0.0),
        ("2026-10-11T00:00:00Z", -1.0),
    ],
)
def test_executable_decision_with_invalid_identity_or_price_is_rejected(
    timestamp: str | None,
    close: float | None,
) -> None:
    adapter = make_adapter()

    result = adapter.apply(
        make_decision(
            DecisionAction.ENTER,
            timestamp=timestamp,
            close=close,
        )
    )

    assert result.outcome is AdapterOutcome.REJECTED
    assert adapter.ledger.intents == ()
    assert adapter.ledger.trade_count == 0
    assert adapter.processed_keys == frozenset()


def test_failed_enter_is_not_consumed_and_can_be_retried() -> None:
    adapter = make_adapter(
        allocated_cash=Decimal("100"),
        max_order_notional=Decimal("100"),
    )
    decision = make_decision(DecisionAction.ENTER, close=100.0)

    result = adapter.apply(decision)

    assert result.outcome is AdapterOutcome.EXECUTED
    assert adapter.ledger.position.is_open


def test_adapter_events_are_append_only_and_auditable() -> None:
    adapter = make_adapter()

    adapter.apply(make_decision(DecisionAction.HOLD))
    adapter.apply(
        make_decision(
            DecisionAction.ENTER,
            timestamp="2026-10-11T00:05:00Z",
        )
    )

    assert [event.outcome for event in adapter.events] == [
        AdapterOutcome.NO_ACTION,
        AdapterOutcome.EXECUTED,
    ]
    assert [event.sequence for event in adapter.events] == [1, 2]
    assert adapter.events[1].intent_id is not None


def test_empty_strategy_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="strategy_id"):
        StrategyLedgerAdapter(
            ledger=make_ledger(),
            strategy_id="   ",
        )
