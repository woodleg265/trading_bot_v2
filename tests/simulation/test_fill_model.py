from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.simulation.fill_model import SimulationFillConfig, SimulationFillModel
from src.simulation.models import OrderIntent, RiskDecision, Side


NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)


def make_intent(
    *,
    intent_id: str = "sim-run-001:intent:000001",
    side: Side = Side.BUY,
    quantity: Decimal = Decimal("0.01"),
    reference_price: Decimal = Decimal("50000.00"),
) -> OrderIntent:
    return OrderIntent(
        intent_id=intent_id,
        symbol="BTC/USD",
        side=side,
        quantity=quantity,
        reference_price=reference_price,
        market_data_timestamp=NOW,
        created_at=NOW,
    )


def approved_decision() -> RiskDecision:
    return RiskDecision(
        approved=True,
        code="APPROVED",
        reason="Intent passed all virtual risk checks.",
        evaluated_notional=Decimal("500.00"),
    )


def rejected_decision() -> RiskDecision:
    return RiskDecision(
        approved=False,
        code="INSUFFICIENT_VIRTUAL_CASH",
        reason="Virtual cash is insufficient.",
        evaluated_notional=Decimal("500.00"),
    )


@pytest.fixture
def fill_model() -> SimulationFillModel:
    return SimulationFillModel(
        SimulationFillConfig(
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.002"),
        )
    )


def test_creates_deterministic_buy_fill_with_adverse_slippage(
    fill_model: SimulationFillModel,
) -> None:
    fill = fill_model.create_fill(
        make_intent(side=Side.BUY),
        risk_decision=approved_decision(),
        run_id="sim-run-001",
        sequence=1,
    )

    assert fill.fill_id == "sim-run-001:fill:000001"
    assert fill.intent_id == "sim-run-001:intent:000001"
    assert fill.symbol == "BTC/USD"
    assert fill.side is Side.BUY
    assert fill.quantity == Decimal("0.01")
    assert fill.market_price == Decimal("50000.00")
    assert fill.fill_price == Decimal("50100.00000")
    assert fill.fee == Decimal("0.50100000000")
    assert fill.filled_at == NOW


def test_creates_deterministic_sell_fill_with_adverse_slippage(
    fill_model: SimulationFillModel,
) -> None:
    fill = fill_model.create_fill(
        make_intent(side=Side.SELL),
        risk_decision=approved_decision(),
        run_id="sim-run-001",
        sequence=2,
    )

    assert fill.fill_id == "sim-run-001:fill:000002"
    assert fill.side is Side.SELL
    assert fill.market_price == Decimal("50000.00")
    assert fill.fill_price == Decimal("49900.00000")
    assert fill.fee == Decimal("0.49900000000")


def test_uses_zero_fee_and_slippage_when_configured() -> None:
    model = SimulationFillModel(
        SimulationFillConfig(
            fee_rate=Decimal("0"),
            slippage_rate=Decimal("0"),
        )
    )

    fill = model.create_fill(
        make_intent(),
        risk_decision=approved_decision(),
        run_id="sim-run-001",
        sequence=1,
    )

    assert fill.fill_price == Decimal("50000.00")
    assert fill.fee == Decimal("0.0000")


def test_rejects_fill_for_rejected_risk_decision(
    fill_model: SimulationFillModel,
) -> None:
    with pytest.raises(
        ValueError,
        match="Cannot create a simulated fill for a rejected risk decision",
    ):
        fill_model.create_fill(
            make_intent(),
            risk_decision=rejected_decision(),
            run_id="sim-run-001",
            sequence=1,
        )


@pytest.mark.parametrize(
    ("run_id", "sequence", "expected_message"),
    [
        ("", 1, "run_id must not be empty"),
        ("sim-run-001", 0, "sequence must be greater than zero"),
        ("sim-run-001", -1, "sequence must be greater than zero"),
    ],
)
def test_rejects_invalid_deterministic_id_inputs(
    fill_model: SimulationFillModel,
    run_id: str,
    sequence: int,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        fill_model.create_fill(
            make_intent(),
            risk_decision=approved_decision(),
            run_id=run_id,
            sequence=sequence,
        )


def test_rejects_slippage_that_reduces_sell_price_to_zero() -> None:
    model = SimulationFillModel(
        SimulationFillConfig(
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("1"),
        )
    )

    with pytest.raises(
        ValueError,
        match="slippage produced a zero or negative fill price",
    ):
        model.create_fill(
            make_intent(side=Side.SELL),
            risk_decision=approved_decision(),
            run_id="sim-run-001",
            sequence=1,
        )


@pytest.mark.parametrize(
    ("fee_rate", "slippage_rate"),
    [
        (Decimal("-0.001"), Decimal("0")),
        (Decimal("0"), Decimal("-0.001")),
    ],
)
def test_rejects_negative_fill_config_values(
    fee_rate: Decimal,
    slippage_rate: Decimal,
) -> None:
    with pytest.raises(ValueError):
        SimulationFillConfig(
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
