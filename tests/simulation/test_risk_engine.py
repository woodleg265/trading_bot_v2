from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.simulation.models import OrderIntent, Side
from src.simulation.risk_engine import (
    SimulationRiskContext,
    SimulationRiskEngine,
    SimulationRiskLimits,
)


NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def limits() -> SimulationRiskLimits:
    return SimulationRiskLimits(
        allowed_symbols=frozenset({"BTC/USD", "ETH/USD"}),
        max_order_notional=Decimal("1000.00"),
        max_position_notional=Decimal("1500.00"),
        max_total_exposure=Decimal("2000.00"),
        max_market_data_age=timedelta(minutes=5),
        estimated_fee_rate=Decimal("0.001"),
    )


@pytest.fixture
def context() -> SimulationRiskContext:
    return SimulationRiskContext(
        available_cash=Decimal("1000.00"),
        position_quantities={"BTC/USD": Decimal("0.01")},
        current_total_exposure=Decimal("500.00"),
        processed_intent_ids=frozenset(),
    )


def make_intent(
    *,
    intent_id: str = "sim-run-001:intent:000001",
    symbol: str = "BTC/USD",
    side: Side = Side.BUY,
    quantity: Decimal = Decimal("0.01"),
    reference_price: Decimal = Decimal("50000.00"),
    market_data_timestamp: datetime = NOW,
) -> OrderIntent:
    return OrderIntent(
        intent_id=intent_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        reference_price=reference_price,
        market_data_timestamp=market_data_timestamp,
        created_at=NOW,
    )


def test_approves_valid_buy(
    limits: SimulationRiskLimits,
    context: SimulationRiskContext,
) -> None:
    decision = SimulationRiskEngine(limits).evaluate(
        make_intent(),
        context=context,
        evaluated_at=NOW,
    )

    assert decision.approved is True
    assert decision.code == "APPROVED"
    assert decision.evaluated_notional == Decimal("500.00")


@pytest.mark.parametrize(
    ("intent", "expected_code"),
    [
        (
            make_intent(symbol="DOGE/USD"),
            "SYMBOL_NOT_ALLOWED",
        ),
        (
            make_intent(
                market_data_timestamp=NOW - timedelta(minutes=6),
            ),
            "STALE_MARKET_DATA",
        ),
        (
            make_intent(quantity=Decimal("0.03")),
            "MAX_ORDER_NOTIONAL_EXCEEDED",
        ),
    ],
)
def test_rejects_symbol_staleness_and_order_notional(
    limits: SimulationRiskLimits,
    context: SimulationRiskContext,
    intent: OrderIntent,
    expected_code: str,
) -> None:
    decision = SimulationRiskEngine(limits).evaluate(
        intent,
        context=context,
        evaluated_at=NOW,
    )

    assert decision.approved is False
    assert decision.code == expected_code


def test_rejects_duplicate_intent(
    limits: SimulationRiskLimits,
    context: SimulationRiskContext,
) -> None:
    intent = make_intent()
    duplicate_context = SimulationRiskContext(
        available_cash=context.available_cash,
        position_quantities=context.position_quantities,
        current_total_exposure=context.current_total_exposure,
        processed_intent_ids=frozenset({intent.intent_id}),
    )

    decision = SimulationRiskEngine(limits).evaluate(
        intent,
        context=duplicate_context,
        evaluated_at=NOW,
    )

    assert decision.approved is False
    assert decision.code == "DUPLICATE_INTENT"


def test_rejects_buy_when_cash_cannot_cover_fee(
    limits: SimulationRiskLimits,
    context: SimulationRiskContext,
) -> None:
    decision = SimulationRiskEngine(limits).evaluate(
        make_intent(quantity=Decimal("0.02")),
        context=context,
        evaluated_at=NOW,
    )

    assert decision.approved is False
    assert decision.code == "INSUFFICIENT_VIRTUAL_CASH"


def test_rejects_sell_without_virtual_position(
    limits: SimulationRiskLimits,
    context: SimulationRiskContext,
) -> None:
    decision = SimulationRiskEngine(limits).evaluate(
        make_intent(
            side=Side.SELL,
            quantity=Decimal("0.02"),
        ),
        context=context,
        evaluated_at=NOW,
    )

    assert decision.approved is False
    assert decision.code == "INSUFFICIENT_SIMULATED_POSITION"


def test_rejects_position_limit(
    limits: SimulationRiskLimits,
    context: SimulationRiskContext,
) -> None:
    constrained_limits = SimulationRiskLimits(
        allowed_symbols=limits.allowed_symbols,
        max_order_notional=limits.max_order_notional,
        max_position_notional=Decimal("600.00"),
        max_total_exposure=limits.max_total_exposure,
        max_market_data_age=limits.max_market_data_age,
        estimated_fee_rate=limits.estimated_fee_rate,
    )

    decision = SimulationRiskEngine(constrained_limits).evaluate(
        make_intent(quantity=Decimal("0.01")),
        context=context,
        evaluated_at=NOW,
    )

    assert decision.approved is False
    assert decision.code == "MAX_POSITION_NOTIONAL_EXCEEDED"


def test_rejects_total_exposure_limit(
    limits: SimulationRiskLimits,
    context: SimulationRiskContext,
) -> None:
    constrained_context = SimulationRiskContext(
        available_cash=context.available_cash,
        position_quantities=context.position_quantities,
        current_total_exposure=Decimal("1700.00"),
        processed_intent_ids=frozenset(),
    )

    decision = SimulationRiskEngine(limits).evaluate(
        make_intent(),
        context=constrained_context,
        evaluated_at=NOW,
    )

    assert decision.approved is False
    assert decision.code == "MAX_TOTAL_EXPOSURE_EXCEEDED"
