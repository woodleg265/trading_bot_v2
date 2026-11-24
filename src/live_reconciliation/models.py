from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from src.execution.models import ExecutionStatus


def _require_utc_timestamp(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{field_name} must use UTC")


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_non_negative_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value < Decimal("0"):
        raise ValueError(
            f"{field_name} must be a finite Decimal greater than or equal to zero"
        )


class ReconciliationStatus(str, Enum):
    RUNNING = "RUNNING"
    CLEAN = "CLEAN"
    FINDINGS_PRESENT = "FINDINGS_PRESENT"
    FAILED = "FAILED"


class ReconciliationFindingType(str, Enum):
    UNKNOWN_SUBMISSION = "UNKNOWN_SUBMISSION"
    MISSING_EXCHANGE_ORDER = "MISSING_EXCHANGE_ORDER"
    UNEXPECTED_EXCHANGE_ORDER = "UNEXPECTED_EXCHANGE_ORDER"
    EXECUTION_STATUS_MISMATCH = "EXECUTION_STATUS_MISMATCH"
    FILL_QUANTITY_MISMATCH = "FILL_QUANTITY_MISMATCH"
    DUPLICATE_IDEMPOTENCY_KEY = "DUPLICATE_IDEMPOTENCY_KEY"
    SOURCE_DATA_STALE = "SOURCE_DATA_STALE"
    SOURCE_DATA_UNAVAILABLE = "SOURCE_DATA_UNAVAILABLE"


class ReconciliationResolution(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    CONFIRMED_MATCH = "CONFIRMED_MATCH"
    MANUALLY_RESOLVED = "MANUALLY_RESOLVED"


@dataclass(frozen=True, slots=True)
class ExchangeOrderSnapshot:
    """Read-only exchange order state returned by an external adapter."""

    exchange_order_id: str
    idempotency_key: str | None
    intent_id: str | None
    status: ExecutionStatus
    observed_at: datetime
    filled_quantity: Decimal | None = None
    average_fill_price: Decimal | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.exchange_order_id, "exchange_order_id")
        _require_utc_timestamp(self.observed_at, "observed_at")
        if self.idempotency_key is not None:
            _require_non_empty(self.idempotency_key, "idempotency_key")
        if self.intent_id is not None:
            _require_non_empty(self.intent_id, "intent_id")
        if self.filled_quantity is not None:
            _require_non_negative_decimal(self.filled_quantity, "filled_quantity")
        if self.average_fill_price is not None:
            _require_non_negative_decimal(self.average_fill_price, "average_fill_price")


@dataclass(frozen=True, slots=True)
class ExchangeOrderStateSnapshot:
    captured_at: datetime
    orders: tuple[ExchangeOrderSnapshot, ...]

    def __post_init__(self) -> None:
        _require_utc_timestamp(self.captured_at, "captured_at")


@dataclass(frozen=True, slots=True)
class ReconciliationRun:
    run_id: str
    started_at: datetime
    completed_at: datetime | None
    status: ReconciliationStatus
    exchange_snapshot_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, "run_id")
        _require_utc_timestamp(self.started_at, "started_at")
        if self.completed_at is not None:
            _require_utc_timestamp(self.completed_at, "completed_at")
            if self.completed_at < self.started_at:
                raise ValueError("completed_at cannot be earlier than started_at")
        if self.exchange_snapshot_at is not None:
            _require_utc_timestamp(self.exchange_snapshot_at, "exchange_snapshot_at")


@dataclass(frozen=True, slots=True)
class ReconciliationFinding:
    finding_id: str
    run_id: str
    finding_type: ReconciliationFindingType
    message: str
    created_at: datetime
    resolution: ReconciliationResolution = ReconciliationResolution.UNRESOLVED
    idempotency_key: str | None = None
    exchange_order_id: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.finding_id, "finding_id")
        _require_non_empty(self.run_id, "run_id")
        _require_non_empty(self.message, "message")
        _require_utc_timestamp(self.created_at, "created_at")
        if self.idempotency_key is not None:
            _require_non_empty(self.idempotency_key, "idempotency_key")
        if self.exchange_order_id is not None:
            _require_non_empty(self.exchange_order_id, "exchange_order_id")


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    run: ReconciliationRun
    findings: tuple[ReconciliationFinding, ...]


@dataclass(frozen=True, slots=True)
class OperationalStatus:
    live_trading_enabled: bool
    kill_switch_engaged: bool
    reconciliation_required: bool
    latest_reconciliation_status: ReconciliationStatus | None
    last_reconciliation_completed_at: datetime | None
    unresolved_finding_count: int
    unknown_execution_count: int
    last_account_snapshot_at: datetime | None
    last_market_snapshot_at: datetime | None
    operationally_ready: bool
