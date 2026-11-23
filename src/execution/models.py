from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Final


class ExecutionSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class IdempotencyStatus(str, Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    UNKNOWN = "UNKNOWN"
    REJECTED = "REJECTED"


class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


_TERMINAL_EXECUTION_STATUSES: Final[frozenset[ExecutionStatus]] = frozenset(
    {
        ExecutionStatus.FILLED,
        ExecutionStatus.CANCELLED,
        ExecutionStatus.REJECTED,
    }
)


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_utc_timestamp(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.astimezone(timezone.utc).utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{field_name} must use UTC")


def _require_positive_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= Decimal("0"):
        raise ValueError(f"{field_name} must be a finite Decimal greater than zero")


def _require_non_negative_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value < Decimal("0"):
        raise ValueError(f"{field_name} must be a finite Decimal greater than or equal to zero")


@dataclass(frozen=True, slots=True)
class StrategyIntentEvent:
    """An immutable upstream request, not proof that an order was submitted."""

    intent_id: str
    strategy_name: str
    symbol: str
    side: ExecutionSide
    quantity: Decimal
    created_at: datetime

    def __post_init__(self) -> None:
        _require_non_empty(self.intent_id, "intent_id")
        _require_non_empty(self.strategy_name, "strategy_name")
        _require_non_empty(self.symbol, "symbol")
        _require_positive_decimal(self.quantity, "quantity")
        _require_utc_timestamp(self.created_at, "created_at")


@dataclass(frozen=True, slots=True)
class OrderSubmissionRequest:
    """The gateway-safe request derived from a strategy intent."""

    idempotency_key: str
    symbol: str
    side: ExecutionSide
    quantity: Decimal

    def __post_init__(self) -> None:
        _require_non_empty(self.idempotency_key, "idempotency_key")
        _require_non_empty(self.symbol, "symbol")
        _require_positive_decimal(self.quantity, "quantity")


@dataclass(frozen=True, slots=True)
class IdempotencyClaim:
    idempotency_key: str
    intent_id: str
    status: IdempotencyStatus
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_non_empty(self.idempotency_key, "idempotency_key")
        _require_non_empty(self.intent_id, "intent_id")
        _require_utc_timestamp(self.created_at, "created_at")
        _require_utc_timestamp(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at")


@dataclass(frozen=True, slots=True)
class ExchangeOrderRecord:
    idempotency_key: str
    intent_id: str
    exchange_order_id: str
    accepted_at: datetime

    def __post_init__(self) -> None:
        _require_non_empty(self.idempotency_key, "idempotency_key")
        _require_non_empty(self.intent_id, "intent_id")
        _require_non_empty(self.exchange_order_id, "exchange_order_id")
        _require_utc_timestamp(self.accepted_at, "accepted_at")


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    idempotency_key: str
    intent_id: str
    status: ExecutionStatus
    reported_at: datetime
    exchange_order_id: str | None = None
    filled_quantity: Decimal | None = None
    average_fill_price: Decimal | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.idempotency_key, "idempotency_key")
        _require_non_empty(self.intent_id, "intent_id")
        _require_utc_timestamp(self.reported_at, "reported_at")

        if self.exchange_order_id is not None:
            _require_non_empty(self.exchange_order_id, "exchange_order_id")
        if self.filled_quantity is not None:
            _require_non_negative_decimal(self.filled_quantity, "filled_quantity")
        if self.average_fill_price is not None:
            _require_positive_decimal(self.average_fill_price, "average_fill_price")
        if self.average_fill_price is not None and self.filled_quantity is None:
            raise ValueError("average_fill_price requires filled_quantity")
        if self.status is ExecutionStatus.FILLED and self.filled_quantity is None:
            raise ValueError("FILLED reports require filled_quantity")

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_EXECUTION_STATUSES
