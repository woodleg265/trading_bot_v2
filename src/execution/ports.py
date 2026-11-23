from __future__ import annotations

from datetime import datetime
from typing import Protocol

from src.execution.models import (
    ExchangeOrderRecord,
    ExecutionReport,
    IdempotencyClaim,
    OrderSubmissionRequest,
)


class ExecutionGateway(Protocol):
    """Port implemented by an exchange-specific adapter outside this package."""

    def submit_order(
        self,
        request: OrderSubmissionRequest,
    ) -> ExchangeOrderRecord:
        """Submit exactly one request and return its exchange acknowledgement."""

    def cancel_order(self, exchange_order_id: str) -> ExecutionReport:
        """Request cancellation for a known exchange order."""

    def fetch_order_status(self, exchange_order_id: str) -> ExecutionReport:
        """Retrieve the current state of a known exchange order."""


class ExecutionStore(Protocol):
    """Durable persistence port.

    `create_idempotency_claim` must be atomic: concurrent calls for the same
    key must result in exactly one `created=True` result.
    """

    def create_idempotency_claim(
        self,
        claim: IdempotencyClaim,
    ) -> tuple[IdempotencyClaim, bool]:
        """Persist the claim or return the existing claim and False."""

    def get_exchange_order(
        self,
        idempotency_key: str,
    ) -> ExchangeOrderRecord | None:
        """Return the acknowledged exchange record, if one exists."""

    def save_exchange_order(self, record: ExchangeOrderRecord) -> None:
        """Persist the intent-to-exchange-order mapping."""

    def save_execution_report(self, report: ExecutionReport) -> None:
        """Append or persist the latest immutable execution report."""

    def update_idempotency_claim(
        self,
        claim: IdempotencyClaim,
    ) -> None:
        """Persist an existing claim's state transition."""

    def now(self) -> datetime:
        """Return a UTC timestamp supplied by the persistence/application layer."""
