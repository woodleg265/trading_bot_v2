from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Mapping


def _require_utc_timestamp(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.astimezone(timezone.utc).utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{field_name} must use UTC")


def _require_non_negative_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value < Decimal("0"):
        raise ValueError(
            f"{field_name} must be a finite Decimal greater than or equal to zero"
        )


def _require_positive_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= Decimal("0"):
        raise ValueError(f"{field_name} must be a finite Decimal greater than zero")


class RiskDecisionStatus(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class LiveRiskLimits:
    allowed_symbols: frozenset[str]
    max_order_notional: Decimal
    max_symbol_exposure: Decimal
    max_total_exposure: Decimal
    max_open_orders: int
    max_market_data_age_seconds: int
    max_account_snapshot_age_seconds: int

    def __post_init__(self) -> None:
        if not self.allowed_symbols or any(
            not isinstance(symbol, str) or not symbol.strip()
            for symbol in self.allowed_symbols
        ):
            raise ValueError("allowed_symbols must contain non-empty symbols")

        _require_positive_decimal(self.max_order_notional, "max_order_notional")
        _require_positive_decimal(self.max_symbol_exposure, "max_symbol_exposure")
        _require_positive_decimal(self.max_total_exposure, "max_total_exposure")

        if not isinstance(self.max_open_orders, int) or self.max_open_orders < 0:
            raise ValueError("max_open_orders must be a non-negative integer")
        if (
            not isinstance(self.max_market_data_age_seconds, int)
            or self.max_market_data_age_seconds < 0
        ):
            raise ValueError(
                "max_market_data_age_seconds must be a non-negative integer"
            )
        if (
            not isinstance(self.max_account_snapshot_age_seconds, int)
            or self.max_account_snapshot_age_seconds < 0
        ):
            raise ValueError(
                "max_account_snapshot_age_seconds must be a non-negative integer"
            )


@dataclass(frozen=True, slots=True)
class LiveTradingControlState:
    """Durable operational controls; every false or halted state blocks orders."""

    live_trading_enabled: bool = False
    kill_switch_engaged: bool = True
    reconciliation_required: bool = True
    operationally_healthy: bool = False


@dataclass(frozen=True, slots=True)
class LiveAccountSnapshot:
    captured_at: datetime
    symbol_quantities: Mapping[str, Decimal]
    symbol_exposures: Mapping[str, Decimal]
    total_exposure: Decimal
    open_order_count: int

    def __post_init__(self) -> None:
        _require_utc_timestamp(self.captured_at, "captured_at")
        _require_non_negative_decimal(self.total_exposure, "total_exposure")

        if not isinstance(self.open_order_count, int) or self.open_order_count < 0:
            raise ValueError("open_order_count must be a non-negative integer")

        for symbol, quantity in self.symbol_quantities.items():
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("symbol_quantities keys must be non-empty strings")
            _require_non_negative_decimal(quantity, "symbol_quantities values")

        for symbol, exposure in self.symbol_exposures.items():
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("symbol_exposures keys must be non-empty strings")
            _require_non_negative_decimal(exposure, "symbol_exposures values")


@dataclass(frozen=True, slots=True)
class MarketPriceSnapshot:
    symbol: str
    price: Decimal
    captured_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        _require_positive_decimal(self.price, "price")
        _require_utc_timestamp(self.captured_at, "captured_at")


@dataclass(frozen=True, slots=True)
class RiskDecision:
    intent_id: str
    status: RiskDecisionStatus
    reason: str
    decided_at: datetime
    order_notional: Decimal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.intent_id, str) or not self.intent_id.strip():
            raise ValueError("intent_id must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        _require_utc_timestamp(self.decided_at, "decided_at")
        if self.order_notional is not None:
            _require_non_negative_decimal(self.order_notional, "order_notional")
