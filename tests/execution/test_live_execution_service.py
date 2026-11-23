from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.execution.idempotency import derive_idempotency_key
from src.execution.models import (
    ExchangeOrderRecord,
    ExecutionSide,
    ExecutionStatus,
    IdempotencyClaim,
    IdempotencyStatus,
    StrategyIntentEvent,
)
from src.execution.service import LiveExecutionService


NOW = datetime(2026, 11, 23, 16, 0, tzinfo=timezone.utc)


class InMemoryExecutionStore:
    def __init__(self) -> None:
        self.claims: dict[str, IdempotencyClaim] = {}
        self.orders: dict[str, ExchangeOrderRecord] = {}
        self.reports = []

    def create_idempotency_claim(
        self,
        claim: IdempotencyClaim,
    ) -> tuple[IdempotencyClaim, bool]:
        existing = self.claims.get(claim.idempotency_key)
        if existing is not None:
            return existing, False
        self.claims[claim.idempotency_key] = claim
        return claim, True

    def get_exchange_order(self, idempotency_key: str) -> ExchangeOrderRecord | None:
        return self.orders.get(idempotency_key)

    def save_exchange_order(self, record: ExchangeOrderRecord) -> None:
        self.orders[record.idempotency_key] = record

    def save_execution_report(self, report) -> None:
        self.reports.append(report)

    def update_idempotency_claim(self, claim: IdempotencyClaim) -> None:
        self.claims[claim.idempotency_key] = claim

    def now(self) -> datetime:
        return NOW


class RecordingGateway:
    def __init__(self) -> None:
        self.requests = []
        self.raise_on_submit: Exception | None = None

    def submit_order(self, request):
        self.requests.append(request)
        if self.raise_on_submit is not None:
            raise self.raise_on_submit
        return ExchangeOrderRecord(
            idempotency_key=request.idempotency_key,
            intent_id="intent-001",
            exchange_order_id="exchange-order-001",
            accepted_at=NOW,
        )

    def cancel_order(self, exchange_order_id: str):
        raise AssertionError("not used in submission tests")

    def fetch_order_status(self, exchange_order_id: str):
        raise AssertionError("not used in submission tests")


@pytest.fixture
def intent() -> StrategyIntentEvent:
    return StrategyIntentEvent(
        intent_id="intent-001",
        strategy_name="brawler",
        symbol="BTC/USD",
        side=ExecutionSide.BUY,
        quantity=Decimal("0.25"),
        created_at=NOW,
    )


def test_submission_persists_pending_claim_before_gateway_call(
    intent: StrategyIntentEvent,
) -> None:
    store = InMemoryExecutionStore()

    class InspectingGateway(RecordingGateway):
        def submit_order(self, request):
            claim = store.claims[request.idempotency_key]
            assert claim.status is IdempotencyStatus.PENDING
            return super().submit_order(request)

    gateway = InspectingGateway()
    service = LiveExecutionService(gateway=gateway, store=store)

    result = service.submit_intent(intent)

    assert result.report.status is ExecutionStatus.ACCEPTED
    assert len(gateway.requests) == 1
    assert store.claims[result.idempotency_key].status is IdempotencyStatus.ACCEPTED


def test_duplicate_intent_is_never_submitted_twice(intent: StrategyIntentEvent) -> None:
    store = InMemoryExecutionStore()
    gateway = RecordingGateway()
    service = LiveExecutionService(gateway=gateway, store=store)

    first = service.submit_intent(intent)
    second = service.submit_intent(intent)

    assert first.submitted_now is True
    assert second.submitted_now is False
    assert second.exchange_order == first.exchange_order
    assert len(gateway.requests) == 1


def test_successful_acknowledgement_persists_order_and_report(
    intent: StrategyIntentEvent,
) -> None:
    store = InMemoryExecutionStore()
    gateway = RecordingGateway()
    service = LiveExecutionService(gateway=gateway, store=store)

    result = service.submit_intent(intent)

    assert result.exchange_order is not None
    assert result.exchange_order.exchange_order_id == "exchange-order-001"
    assert store.orders[result.idempotency_key] == result.exchange_order
    assert result.report.status is ExecutionStatus.ACCEPTED
    assert result.report.exchange_order_id == "exchange-order-001"


def test_ambiguous_gateway_failure_requires_reconciliation(
    intent: StrategyIntentEvent,
) -> None:
    store = InMemoryExecutionStore()
    gateway = RecordingGateway()
    gateway.raise_on_submit = TimeoutError("gateway response timed out")
    service = LiveExecutionService(gateway=gateway, store=store)

    result = service.submit_intent(intent)

    assert result.submitted_now is True
    assert result.exchange_order is None
    assert result.report.status is ExecutionStatus.UNKNOWN
    assert store.claims[result.idempotency_key].status is IdempotencyStatus.UNKNOWN
    assert len(gateway.requests) == 1


def test_unresolved_claim_does_not_trigger_automatic_resubmission(
    intent: StrategyIntentEvent,
) -> None:
    store = InMemoryExecutionStore()
    gateway = RecordingGateway()
    gateway.raise_on_submit = ConnectionError("connection dropped")
    service = LiveExecutionService(gateway=gateway, store=store)

    first = service.submit_intent(intent)
    gateway.raise_on_submit = None
    second = service.submit_intent(intent)

    assert first.report.status is ExecutionStatus.UNKNOWN
    assert second.submitted_now is False
    assert second.report.status is ExecutionStatus.UNKNOWN
    assert len(gateway.requests) == 1


def test_idempotency_key_is_stable_for_same_intent(intent: StrategyIntentEvent) -> None:
    assert derive_idempotency_key(intent) == derive_idempotency_key(intent)


@pytest.mark.parametrize(
    ("quantity", "error"),
    [
        (Decimal("0"), "quantity"),
        (Decimal("-1"), "quantity"),
        (Decimal("NaN"), "quantity"),
        (Decimal("Infinity"), "quantity"),
    ],
)
def test_invalid_intent_quantity_is_rejected(
    quantity: Decimal,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        StrategyIntentEvent(
            intent_id="intent-001",
            strategy_name="brawler",
            symbol="BTC/USD",
            side=ExecutionSide.BUY,
            quantity=quantity,
            created_at=NOW,
        )


def test_naive_intent_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        StrategyIntentEvent(
            intent_id="intent-001",
            strategy_name="brawler",
            symbol="BTC/USD",
            side=ExecutionSide.BUY,
            quantity=Decimal("1"),
            created_at=datetime(2026, 11, 23, 16, 0),
        )
