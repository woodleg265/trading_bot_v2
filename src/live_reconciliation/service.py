from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from src.execution.models import (
    ExchangeOrderRecord,
    ExecutionReport,
    ExecutionStatus,
)
from src.live_reconciliation.models import (
    ExchangeOrderSnapshot,
    ReconciliationFinding,
    ReconciliationFindingType,
    ReconciliationResult,
    ReconciliationRun,
    ReconciliationStatus,
)
from src.live_reconciliation.ports import (
    ExchangeOrderStateProvider,
    ExecutionHistoryProvider,
    ReconciliationStore,
)
from src.live_risk.models import LiveTradingControlState
from src.live_risk.ports import LiveTradingControlStore


class LiveReconciliationService:
    """Read-only reconciliation service that fails closed on every uncertainty."""

    def __init__(
        self,
        *,
        exchange_order_state_provider: ExchangeOrderStateProvider,
        execution_history_provider: ExecutionHistoryProvider,
        reconciliation_store: ReconciliationStore,
        control_store: LiveTradingControlStore,
        max_exchange_snapshot_age_seconds: int,
    ) -> None:
        if max_exchange_snapshot_age_seconds < 0:
            raise ValueError("max_exchange_snapshot_age_seconds must be non-negative")
        self._exchange_order_state_provider = exchange_order_state_provider
        self._execution_history_provider = execution_history_provider
        self._reconciliation_store = reconciliation_store
        self._control_store = control_store
        self._max_exchange_snapshot_age_seconds = max_exchange_snapshot_age_seconds

    def run(self, *, since: datetime) -> ReconciliationResult:
        if since.tzinfo is None or since.utcoffset() is None:
            raise ValueError("since must be timezone-aware")

        now = self._control_store.now()
        run = ReconciliationRun(
            run_id=str(uuid4()),
            started_at=now,
            completed_at=None,
            status=ReconciliationStatus.RUNNING,
        )

        try:
            self._reconciliation_store.begin_run(run)
        except Exception as error:
            self._require_reconciliation()
            raise RuntimeError("reconciliation run persistence failed") from error

        try:
            local_orders = self._execution_history_provider.list_exchange_orders_since(
                since=since
            )
            reports = self._execution_history_provider.list_execution_reports_since(
                since=since
            )
            exchange_snapshot = self._exchange_order_state_provider.get_order_snapshot(
                since=since
            )
        except Exception as error:
            return self._fail_closed(
                run=run,
                finding_type=ReconciliationFindingType.SOURCE_DATA_UNAVAILABLE,
                message=f"Reconciliation source unavailable: {type(error).__name__}",
            )

        findings = self._compare(
            run_id=run.run_id,
            created_at=now,
            local_orders=local_orders,
            reports=reports,
            exchange_orders=exchange_snapshot.orders,
        )

        if (
            exchange_snapshot.captured_at > now
            or now - exchange_snapshot.captured_at
            > timedelta(seconds=self._max_exchange_snapshot_age_seconds)
        ):
            findings.append(
                self._finding(
                    run_id=run.run_id,
                    finding_type=ReconciliationFindingType.SOURCE_DATA_STALE,
                    message="Exchange order snapshot is stale.",
                    created_at=now,
                )
            )

        try:
            for finding in findings:
                self._reconciliation_store.record_finding(finding)

            completed = replace(
                run,
                completed_at=self._control_store.now(),
                status=(
                    ReconciliationStatus.CLEAN
                    if not findings
                    else ReconciliationStatus.FINDINGS_PRESENT
                ),
                exchange_snapshot_at=exchange_snapshot.captured_at,
            )
            self._reconciliation_store.complete_run(completed)
        except Exception as error:
            self._require_reconciliation()
            raise RuntimeError("reconciliation persistence failed") from error

        if findings:
            self._require_reconciliation()

        return ReconciliationResult(run=completed, findings=tuple(findings))

    def _compare(
        self,
        *,
        run_id: str,
        created_at: datetime,
        local_orders: tuple[ExchangeOrderRecord, ...],
        reports: tuple[ExecutionReport, ...],
        exchange_orders: tuple[ExchangeOrderSnapshot, ...],
    ) -> list[ReconciliationFinding]:
        findings: list[ReconciliationFinding] = []
        local_by_exchange_id = {
            record.exchange_order_id: record for record in local_orders
        }
        exchange_by_id = {order.exchange_order_id: order for order in exchange_orders}
        exchange_idempotency_counts: dict[str, int] = {}

        for order in exchange_orders:
            if order.idempotency_key is not None:
                exchange_idempotency_counts[order.idempotency_key] = (
                    exchange_idempotency_counts.get(order.idempotency_key, 0) + 1
                )

        for key, count in exchange_idempotency_counts.items():
            if count > 1:
                findings.append(
                    self._finding(
                        run_id=run_id,
                        finding_type=ReconciliationFindingType.DUPLICATE_IDEMPOTENCY_KEY,
                        message="Multiple exchange orders share an idempotency key.",
                        created_at=created_at,
                        idempotency_key=key,
                    )
                )

        for record in local_orders:
            exchange_order = exchange_by_id.get(record.exchange_order_id)
            if exchange_order is None:
                findings.append(
                    self._finding(
                        run_id=run_id,
                        finding_type=ReconciliationFindingType.MISSING_EXCHANGE_ORDER,
                        message="Local exchange acknowledgement is absent from exchange state.",
                        created_at=created_at,
                        idempotency_key=record.idempotency_key,
                        exchange_order_id=record.exchange_order_id,
                    )
                )

        for exchange_order in exchange_orders:
            if exchange_order.exchange_order_id not in local_by_exchange_id:
                findings.append(
                    self._finding(
                        run_id=run_id,
                        finding_type=ReconciliationFindingType.UNEXPECTED_EXCHANGE_ORDER,
                        message="Exchange order has no local acknowledgement.",
                        created_at=created_at,
                        idempotency_key=exchange_order.idempotency_key,
                        exchange_order_id=exchange_order.exchange_order_id,
                    )
                )

        latest_reports: dict[str, ExecutionReport] = {}
        for report in reports:
            previous = latest_reports.get(report.idempotency_key)
            if previous is None or report.reported_at >= previous.reported_at:
                latest_reports[report.idempotency_key] = report

        for report in latest_reports.values():
            if report.status is ExecutionStatus.UNKNOWN:
                findings.append(
                    self._finding(
                        run_id=run_id,
                        finding_type=ReconciliationFindingType.UNKNOWN_SUBMISSION,
                        message="Execution report remains UNKNOWN.",
                        created_at=created_at,
                        idempotency_key=report.idempotency_key,
                        exchange_order_id=report.exchange_order_id,
                    )
                )
                continue

            if report.exchange_order_id is None:
                continue

            exchange_order = exchange_by_id.get(report.exchange_order_id)
            if exchange_order is None:
                continue

            if (
                report.is_terminal
                and exchange_order.status in {
                    ExecutionStatus.FILLED,
                    ExecutionStatus.CANCELLED,
                    ExecutionStatus.REJECTED,
                }
                and report.status is not exchange_order.status
            ):
                findings.append(
                    self._finding(
                        run_id=run_id,
                        finding_type=ReconciliationFindingType.EXECUTION_STATUS_MISMATCH,
                        message="Terminal local and exchange statuses differ.",
                        created_at=created_at,
                        idempotency_key=report.idempotency_key,
                        exchange_order_id=report.exchange_order_id,
                    )
                )

            if (
                report.filled_quantity is not None
                and exchange_order.filled_quantity is not None
                and report.filled_quantity != exchange_order.filled_quantity
            ):
                findings.append(
                    self._finding(
                        run_id=run_id,
                        finding_type=ReconciliationFindingType.FILL_QUANTITY_MISMATCH,
                        message="Local and exchange filled quantities differ.",
                        created_at=created_at,
                        idempotency_key=report.idempotency_key,
                        exchange_order_id=report.exchange_order_id,
                    )
                )

        return findings

    def _fail_closed(
        self,
        *,
        run: ReconciliationRun,
        finding_type: ReconciliationFindingType,
        message: str,
    ) -> ReconciliationResult:
        finding = self._finding(
            run_id=run.run_id,
            finding_type=finding_type,
            message=message,
            created_at=self._control_store.now(),
        )
        self._require_reconciliation()
        try:
            self._reconciliation_store.record_finding(finding)
            completed = replace(
                run,
                completed_at=self._control_store.now(),
                status=ReconciliationStatus.FAILED,
            )
            self._reconciliation_store.complete_run(completed)
        except Exception:
            # Lockout was already persisted before this best-effort audit completion.
            completed = replace(
                run,
                completed_at=self._control_store.now(),
                status=ReconciliationStatus.FAILED,
            )
        return ReconciliationResult(run=completed, findings=(finding,))

    def _require_reconciliation(self) -> None:
        state = self._control_store.get_control_state()
        if not state.reconciliation_required:
            self._control_store.save_control_state(
                replace(state, reconciliation_required=True)
            )

    @staticmethod
    def _finding(
        *,
        run_id: str,
        finding_type: ReconciliationFindingType,
        message: str,
        created_at: datetime,
        idempotency_key: str | None = None,
        exchange_order_id: str | None = None,
    ) -> ReconciliationFinding:
        return ReconciliationFinding(
            finding_id=str(uuid4()),
            run_id=run_id,
            finding_type=finding_type,
            message=message,
            created_at=created_at,
            idempotency_key=idempotency_key,
            exchange_order_id=exchange_order_id,
        )
