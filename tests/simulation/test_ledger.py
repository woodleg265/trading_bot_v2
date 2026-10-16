from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.simulation.ledger import SimulationLedger, SimulationLedgerState
from src.simulation.models import Side, SimulatedFill


NOW = datetime(2026, 10, 14, tzinfo=timezone.utc)


def make_fill(
    *,
    fill_id: str = "sim-run-1:fill:1",
    symbol: str = "BTC/USD",
    side: Side = Side.BUY,
    quantity: str = "2",
    fill_price: str = "100",
    fee: str = "1",
) -> SimulatedFill:
    return SimulatedFill(
        fill_id=fill_id,
        intent_id="sim-run-1:intent:1",
        symbol=symbol,
        side=side,
        quantity=Decimal(quantity),
        market_price=Decimal(fill_price),
        fill_price=Decimal(fill_price),
        fee=Decimal(fee),
        filled_at=NOW,
    )


def test_buy_fill_returns_new_state_and_deterministic_entry() -> None:
    ledger = SimulationLedger()
    original = SimulationLedgerState(
        cash_balance=Decimal("1000"),
        cash_currency="USD",
        position_quantities={},
    )

    updated, entry = ledger.apply_fill(
        original,
        make_fill(),
        run_id="sim-run-1",
        sequence=1,
    )

    assert original.cash_balance == Decimal("1000")
    assert dict(original.position_quantities) == {}
    assert updated.cash_balance == Decimal("799")
    assert updated.position_quantities["BTC"] == Decimal("2")
    assert entry.entry_id == "sim-run-1:ledger:1"
    assert entry.executed_notional == Decimal("200")
    assert entry.cash_delta == Decimal("-201")
    assert entry.position_delta == Decimal("2")


def test_sell_fill_increases_cash_and_reduces_position() -> None:
    ledger = SimulationLedger()
    state = SimulationLedgerState(
        cash_balance=Decimal("500"),
        cash_currency="USD",
        position_quantities={"BTC": Decimal("3")},
    )

    updated, entry = ledger.apply_fill(
        state,
        make_fill(side=Side.SELL, quantity="2", fill_price="120", fee="2"),
        run_id="sim-run-1",
        sequence=2,
    )

    assert updated.cash_balance == Decimal("738")
    assert updated.position_quantities["BTC"] == Decimal("1")
    assert entry.executed_notional == Decimal("240")
    assert entry.cash_delta == Decimal("238")
    assert entry.position_delta == Decimal("-2")


def test_multiple_fills_preserve_exact_decimal_values_and_entry_order() -> None:
    ledger = SimulationLedger()
    initial = SimulationLedgerState(Decimal("1000"), "USD", {})

    after_buy, buy_entry = ledger.apply_fill(
        initial,
        make_fill(quantity="3", fill_price="100.10", fee="0.30"),
        run_id="sim-run-1",
        sequence=1,
    )
    after_sell, sell_entry = ledger.apply_fill(
        after_buy,
        make_fill(
            fill_id="sim-run-1:fill:2",
            side=Side.SELL,
            quantity="1.25",
            fill_price="110.20",
            fee="0.20",
        ),
        run_id="sim-run-1",
        sequence=2,
    )

    assert after_sell.cash_balance == Decimal("836.950")
    assert after_sell.position_quantities["BTC"] == Decimal("1.75")
    assert [buy_entry.entry_id, sell_entry.entry_id] == [
        "sim-run-1:ledger:1",
        "sim-run-1:ledger:2",
    ]


def test_buy_rejects_insufficient_cash() -> None:
    state = SimulationLedgerState(Decimal("200"), "USD", {})

    with pytest.raises(ValueError, match="insufficient virtual cash"):
        SimulationLedger().apply_fill(
            state, make_fill(), run_id="sim-run-1", sequence=1
        )


def test_sell_rejects_insufficient_position() -> None:
    state = SimulationLedgerState(
        Decimal("1000"), "USD", {"BTC": Decimal("1")}
    )

    with pytest.raises(ValueError, match="insufficient simulated position"):
        SimulationLedger().apply_fill(
            state,
            make_fill(side=Side.SELL, quantity="2"),
            run_id="sim-run-1",
            sequence=1,
        )


def test_state_position_mapping_cannot_be_mutated() -> None:
    state = SimulationLedgerState(
        Decimal("1000"), "USD", {"BTC": Decimal("1")}
    )

    with pytest.raises(TypeError):
        state.position_quantities["BTC"] = Decimal("2")  # type: ignore[index]


@pytest.mark.parametrize("symbol", ["BTCUSD", "BTC/", "/USD", "BTC/USD/USDT"])
def test_invalid_symbol_is_rejected(symbol: str) -> None:
    state = SimulationLedgerState(Decimal("1000"), "USD", {})

    with pytest.raises(ValueError, match="BASE/QUOTE"):
        SimulationLedger().apply_fill(
            state,
            make_fill(symbol=symbol),
            run_id="sim-run-1",
            sequence=1,
        )


def test_quote_currency_must_match_ledger_cash_currency() -> None:
    state = SimulationLedgerState(Decimal("1000"), "USD", {})

    with pytest.raises(ValueError, match="does not match"):
        SimulationLedger().apply_fill(
            state,
            make_fill(symbol="BTC/EUR"),
            run_id="sim-run-1",
            sequence=1,
        )


def test_same_inputs_produce_equal_entries() -> None:
    ledger = SimulationLedger()
    state = SimulationLedgerState(Decimal("1000"), "USD", {})
    fill = make_fill()

    _, first = ledger.apply_fill(state, fill, run_id="sim-run-1", sequence=1)
    _, second = ledger.apply_fill(state, fill, run_id="sim-run-1", sequence=1)

    assert first == second
