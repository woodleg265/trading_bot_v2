from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.simulation.models import OrderIntent, Side


def test_order_intent_is_immutable() -> None:
    intent = OrderIntent(
        intent_id="sim-run-001:intent:000001",
        symbol="BTC/USD",
        side=Side.BUY,
        quantity=Decimal("0.01"),
        reference_price=Decimal("50000.00"),
        market_data_timestamp=datetime(2026, 10, 13, tzinfo=timezone.utc),
        created_at=datetime(2026, 10, 13, tzinfo=timezone.utc),
    )

    with pytest.raises(FrozenInstanceError):
        intent.symbol = "ETH/USD"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("quantity", "reference_price"),
    [
        (Decimal("0"), Decimal("1")),
        (Decimal("-0.01"), Decimal("1")),
        (Decimal("1"), Decimal("0")),
        (Decimal("1"), Decimal("-1")),
    ],
)
def test_order_intent_rejects_non_positive_quantity_or_price(
    quantity: Decimal,
    reference_price: Decimal,
) -> None:
    with pytest.raises(ValueError):
        OrderIntent(
            intent_id="sim-run-001:intent:000001",
            symbol="BTC/USD",
            side=Side.BUY,
            quantity=quantity,
            reference_price=reference_price,
            market_data_timestamp=datetime(2026, 10, 13, tzinfo=timezone.utc),
            created_at=datetime(2026, 10, 13, tzinfo=timezone.utc),
        )
