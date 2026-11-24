from __future__ import annotations

from datetime import datetime
from typing import Protocol

from src.execution.models import ExchangeOrderRecord, ExecutionReport
from src.live_reconciliation.models import (
    ExchangeOrderStateSnapshot,
    ReconciliationFinding,
    ReconciliationResolution,
    ReconciliationRun,
)


class ExchangeOrderStateProvider(Protocol):
    """Read-only external exchange-state adapter."""

    def get_order_snapshot(self, *, since: datetime) -> ExchangeOrderStateSnapshot:
        """Return exchange orders observed since the requested UTC time."""


class ExecutionHistoryProvider(Protocol):
    """Read-only durable execution-history adapter."""

    def list_exchange_orders_since(self, *, since: datetime) -> tuple[ExchangeOrderRecord, ...]:
        """Return locally persisted exchange acknowledgements."""

    def list_execution_reports_since(self, *, since: datetime) -> tuple[ExecutionReport, ...]:
        """Return durable execution reports."""


class ReconciliationStore(Protocol):
    """Durable reconciliation audit store."""

    def begin_run(self, run: ReconciliationRun) -> None:
        ...

    def record_finding(self, finding: ReconciliationFinding) -> None:
        ...

    def complete_run(self, run: ReconciliationRun) -> None:
        ...

    def latest_run(self) -> ReconciliationRun | None:
        ...

    def list_unresolved_findings(self) -> tuple[ReconciliationFinding, ...]:
        ...

    def resolve_finding(
        self,
        *,
        finding_id: str,
        resolution: ReconciliationResolution,
        resolved_at: datetime,
        operator_note: str,
    ) -> ReconciliationFinding:
        ...
