from decimal import Decimal

import pytest

from src.portfolio.allocation_ledger import (
    AllocationLedger,
    IntentStatus,
    LedgerError,
)


def make_ledger(**overrides: Decimal) -> AllocationLedger:
    values = {
        "allocated_cash": Decimal("1000"),
        "fee_rate": Decimal("0.001"),
        "max_order_notional": Decimal("1000"),
    }
    values.update(overrides)
    return AllocationLedger(**values)


def open_position(
    ledger: AllocationLedger,
    *,
    quantity: Decimal = Decimal("1"),
    price: Decimal = Decimal("100"),
) -> str:
    intent = ledger.create_buy_intent(
        quantity=quantity,
        reference_price=price,
    )
    ledger.fill_intent(intent.intent_id, fill_price=price)
    return intent.intent_id


def test_ledger_starts_with_explicit_allocated_cash() -> None:
    ledger = make_ledger()

    assert ledger.allocated_cash == Decimal("1000")
    assert ledger.available_cash == Decimal("1000")
    assert ledger.position.quantity == Decimal("0")
    assert ledger.position.average_entry_price == Decimal("0")
    assert ledger.realized_pnl == Decimal("0")
    assert ledger.total_fees == Decimal("0")
    assert ledger.trade_count == 0
    assert len(ledger.events) == 1
    assert ledger.events[0].event_type == "LEDGER_CREATED"


def test_buy_intent_does_not_mutate_cash_or_position() -> None:
    ledger = make_ledger()

    intent = ledger.create_buy_intent(
        quantity=Decimal("1"),
        reference_price=Decimal("100"),
    )

    assert intent.status is IntentStatus.CREATED
    assert ledger.available_cash == Decimal("1000")
    assert ledger.position.quantity == Decimal("0")
    assert ledger.trade_count == 0
    assert ledger.intents == (intent,)


def test_buy_fill_subtracts_notional_and_fee_and_opens_position() -> None:
    ledger = make_ledger()

    intent = ledger.create_buy_intent(
        quantity=Decimal("2"),
        reference_price=Decimal("100"),
    )
    filled = ledger.fill_intent(intent.intent_id, fill_price=Decimal("100"))

    assert filled.status is IntentStatus.FILLED
    assert ledger.available_cash == Decimal("799.8")
    assert ledger.position.quantity == Decimal("2")
    assert ledger.position.average_entry_price == Decimal("100.1")
    assert ledger.total_fees == Decimal("0.2")
    assert ledger.trade_count == 1


def test_buy_fill_fails_when_cash_cannot_cover_notional_plus_fee() -> None:
    ledger = make_ledger(allocated_cash=Decimal("100"))

    intent = ledger.create_buy_intent(
        quantity=Decimal("1"),
        reference_price=Decimal("100"),
    )

    with pytest.raises(LedgerError, match="Insufficient allocated"):
        ledger.fill_intent(intent.intent_id, fill_price=Decimal("100"))

    assert ledger.available_cash == Decimal("100")
    assert ledger.position.quantity == Decimal("0")
    assert ledger.trade_count == 0
    assert ledger.intents[0].status is IntentStatus.CREATED


def test_second_buy_intent_is_rejected_while_position_is_open() -> None:
    ledger = make_ledger()
    open_position(ledger)

    with pytest.raises(LedgerError, match="position is open"):
        ledger.create_buy_intent(
            quantity=Decimal("1"),
            reference_price=Decimal("100"),
        )


def test_sell_intent_cannot_exceed_open_position_quantity() -> None:
    ledger = make_ledger()
    open_position(ledger, quantity=Decimal("1"))

    with pytest.raises(LedgerError, match="exceeds"):
        ledger.create_sell_intent(
            quantity=Decimal("1.00000001"),
            reference_price=Decimal("100"),
        )


def test_sell_fill_adds_net_proceeds_and_realizes_pnl() -> None:
    ledger = make_ledger()
    open_position(ledger, quantity=Decimal("1"), price=Decimal("100"))

    sell_intent = ledger.create_sell_intent(
        quantity=Decimal("1"),
        reference_price=Decimal("120"),
    )
    ledger.fill_intent(sell_intent.intent_id, fill_price=Decimal("120"))

    assert ledger.position.quantity == Decimal("0")
    assert ledger.position.average_entry_price == Decimal("0")
    assert ledger.available_cash == Decimal("1019.78")
    assert ledger.total_fees == Decimal("0.22")
    assert ledger.realized_pnl == Decimal("19.78")
    assert ledger.trade_count == 2


def test_full_sell_closes_position() -> None:
    ledger = make_ledger()
    open_position(ledger, quantity=Decimal("2"), price=Decimal("100"))

    sell_intent = ledger.create_sell_intent(
        quantity=Decimal("2"),
        reference_price=Decimal("100"),
    )
    ledger.fill_intent(sell_intent.intent_id, fill_price=Decimal("100"))

    assert not ledger.position.is_open
    assert ledger.position.quantity == Decimal("0")
    assert ledger.position.average_entry_price == Decimal("0")


def test_partial_sell_preserves_average_entry_price() -> None:
    ledger = make_ledger()
    open_position(ledger, quantity=Decimal("2"), price=Decimal("100"))

    sell_intent = ledger.create_sell_intent(
        quantity=Decimal("0.5"),
        reference_price=Decimal("120"),
    )
    ledger.fill_intent(sell_intent.intent_id, fill_price=Decimal("120"))

    assert ledger.position.quantity == Decimal("1.5")
    assert ledger.position.average_entry_price == Decimal("100.1")
    assert ledger.realized_pnl == Decimal("9.89")


def test_intent_cannot_be_filled_twice() -> None:
    ledger = make_ledger()

    intent = ledger.create_buy_intent(
        quantity=Decimal("1"),
        reference_price=Decimal("100"),
    )
    ledger.fill_intent(intent.intent_id, fill_price=Decimal("100"))

    with pytest.raises(LedgerError, match="cannot be changed"):
        ledger.fill_intent(intent.intent_id, fill_price=Decimal("100"))

    assert ledger.trade_count == 1


def test_cancelled_intent_cannot_be_filled() -> None:
    ledger = make_ledger()

    intent = ledger.create_buy_intent(
        quantity=Decimal("1"),
        reference_price=Decimal("100"),
    )
    cancelled = ledger.cancel_intent(intent.intent_id)

    assert cancelled.status is IntentStatus.CANCELLED

    with pytest.raises(LedgerError, match="cannot be changed"):
        ledger.fill_intent(intent.intent_id, fill_price=Decimal("100"))

    assert ledger.available_cash == Decimal("1000")
    assert ledger.position.quantity == Decimal("0")


def test_order_notional_cannot_exceed_limit() -> None:
    ledger = make_ledger(max_order_notional=Decimal("99"))

    with pytest.raises(LedgerError, match="max_order_notional"):
        ledger.create_buy_intent(
            quantity=Decimal("1"),
            reference_price=Decimal("100"),
        )

    assert ledger.intents == ()


def test_events_are_append_only_for_create_fill_and_cancel() -> None:
    ledger = make_ledger()

    buy_intent = ledger.create_buy_intent(
        quantity=Decimal("1"),
        reference_price=Decimal("100"),
    )
    ledger.cancel_intent(buy_intent.intent_id)

    second_buy = ledger.create_buy_intent(
        quantity=Decimal("1"),
        reference_price=Decimal("100"),
    )
    ledger.fill_intent(second_buy.intent_id, fill_price=Decimal("100"))

    event_types = [event.event_type for event in ledger.events]

    assert event_types == [
        "LEDGER_CREATED",
        "INTENT_CREATED",
        "INTENT_CANCELLED",
        "INTENT_CREATED",
        "INTENT_FILLED",
    ]
    assert len({event.sequence for event in ledger.events}) == len(ledger.events)


def test_equity_includes_marked_open_position_value() -> None:
    ledger = make_ledger()
    open_position(ledger, quantity=Decimal("2"), price=Decimal("100"))

    assert ledger.equity(Decimal("100")) == Decimal("999.8")
    assert ledger.equity(Decimal("125")) == Decimal("1049.8")


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("allocated_cash", Decimal("0"), "allocated_cash"),
        ("allocated_cash", Decimal("-1"), "allocated_cash"),
        ("fee_rate", Decimal("-0.001"), "fee_rate"),
        ("max_order_notional", Decimal("0"), "max_order_notional"),
        ("max_order_notional", Decimal("-1"), "max_order_notional"),
    ],
)
def test_invalid_ledger_configuration_is_rejected(
    field_name: str,
    value: Decimal,
    message: str,
) -> None:
    values = {
        "allocated_cash": Decimal("1000"),
        "fee_rate": Decimal("0.001"),
        "max_order_notional": Decimal("1000"),
    }
    values[field_name] = value

    with pytest.raises(LedgerError, match=message):
        AllocationLedger(**values)


@pytest.mark.parametrize(
    ("quantity", "price"),
    [
        (Decimal("0"), Decimal("100")),
        (Decimal("-1"), Decimal("100")),
        (Decimal("1"), Decimal("0")),
        (Decimal("1"), Decimal("-100")),
    ],
)
def test_invalid_order_quantity_or_price_is_rejected(
    quantity: Decimal,
    price: Decimal,
) -> None:
    ledger = make_ledger()

    with pytest.raises(LedgerError):
        ledger.create_buy_intent(
            quantity=quantity,
            reference_price=price,
        )


def test_tiny_quantity_that_rounds_to_zero_is_rejected() -> None:
    ledger = make_ledger()

    with pytest.raises(LedgerError, match="too small"):
        ledger.create_buy_intent(
            quantity=Decimal("0.000000001"),
            reference_price=Decimal("100"),
        )


def test_sell_requires_an_open_position() -> None:
    ledger = make_ledger()

    with pytest.raises(LedgerError, match="without an open position"):
        ledger.create_sell_intent(
            quantity=Decimal("1"),
            reference_price=Decimal("100"),
        )


def test_unknown_intent_cannot_be_filled_or_cancelled() -> None:
    ledger = make_ledger()

    with pytest.raises(LedgerError, match="Unknown intent"):
        ledger.fill_intent("does-not-exist", fill_price=Decimal("100"))

    with pytest.raises(LedgerError, match="Unknown intent"):
        ledger.cancel_intent("does-not-exist")
