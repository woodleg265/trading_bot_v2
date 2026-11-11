from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from src.simulation.execution_model import (
    DEFAULT_EXECUTION_MODEL,
    ExecutionModel,
    FillPriceBasis,
    FillTiming,
)


def test_default_execution_model_is_explicit_and_deterministic() -> None:
    assert DEFAULT_EXECUTION_MODEL == ExecutionModel(
        model_id="simulation-candle-close-v1",
        fill_timing=FillTiming.EVALUATION_CANDLE_CLOSE,
        fill_price_basis=FillPriceBasis.CANDLE_CLOSE,
        slippage_rate=Decimal("0"),
        fee_rate=Decimal("0"),
    )


def test_execution_model_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_EXECUTION_MODEL.model_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("model_id", "slippage_rate", "fee_rate", "error_message"),
    [
        ("", Decimal("0"), Decimal("0"), "model_id must not be empty"),
        ("   ", Decimal("0"), Decimal("0"), "model_id must not be empty"),
        (
            "valid-model",
            Decimal("-0.0001"),
            Decimal("0"),
            "slippage_rate must be greater than or equal to zero",
        ),
        (
            "valid-model",
            Decimal("0"),
            Decimal("-0.0001"),
            "fee_rate must be greater than or equal to zero",
        ),
    ],
)
def test_execution_model_rejects_invalid_values(
    model_id: str,
    slippage_rate: Decimal,
    fee_rate: Decimal,
    error_message: str,
) -> None:
    with pytest.raises(ValueError, match=error_message):
        ExecutionModel(
            model_id=model_id,
            fill_timing=FillTiming.EVALUATION_CANDLE_CLOSE,
            fill_price_basis=FillPriceBasis.CANDLE_CLOSE,
            slippage_rate=slippage_rate,
            fee_rate=fee_rate,
        )
