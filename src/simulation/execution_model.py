from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class FillTiming(StrEnum):
    EVALUATION_CANDLE_CLOSE = "EVALUATION_CANDLE_CLOSE"


class FillPriceBasis(StrEnum):
    CANDLE_CLOSE = "CANDLE_CLOSE"


@dataclass(frozen=True, slots=True)
class ExecutionModel:
    """Immutable assumptions used to interpret simulated order execution."""

    model_id: str
    fill_timing: FillTiming
    fill_price_basis: FillPriceBasis
    slippage_rate: Decimal
    fee_rate: Decimal

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")

        if self.slippage_rate < Decimal("0"):
            raise ValueError("slippage_rate must be greater than or equal to zero")

        if self.fee_rate < Decimal("0"):
            raise ValueError("fee_rate must be greater than or equal to zero")


DEFAULT_EXECUTION_MODEL = ExecutionModel(
    model_id="simulation-candle-close-v1",
    fill_timing=FillTiming.EVALUATION_CANDLE_CLOSE,
    fill_price_basis=FillPriceBasis.CANDLE_CLOSE,
    slippage_rate=Decimal("0"),
    fee_rate=Decimal("0"),
)
