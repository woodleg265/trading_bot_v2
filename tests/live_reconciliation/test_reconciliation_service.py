from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.execution.models import (
    ExchangeOrderRecord,
    ExecutionReport,
    ExecutionStatus,
)
from src.live_reconciliation.models import (
    ExchangeOrderSnapshot,
    ExchangeOrderStateSnapshot,
    ReconciliationFinding,
    ReconciliationFindingType,
    ReconciliationResolution,
    ReconciliationRun,
    ReconciliationStatus,
)
from src.live_reconciliation.monitoring import OperationalMonitor
from src.live_reconciliation.service import LiveReconciliationService
from src.live_risk.models import (
    LiveAccountSnapshot,
    LiveTradingControlState,
    MarketPriceSnapshot,
)


NOW = datetime(2026, 11, 24, 20, 0, tzinfo=timezone.utc)
SINCE = NOW - timedelta(hours=1)


class FakeControlStore:
    def __init__(
        self,
        state: LiveTradingControlState | None = None,
        now: datetime = NOW,
    ) -> None:
        self.state = state or LiveTradingControlState(
            live_trading_enabled=True,
            kill_switch_engaged=False,
            reconciliation_required=False,
            operationally_healthy=True,
        )
        self.current_time = now
        self.saved_states: list[LiveTradingControlState] = []

    def get_control_state(self) -> LiveTradingControlState:
        return self.state

    def save_control_state(self, state: LiveTradingControlState) -> None:
        self.state = state
        self.saved_states.append(state)

    def save_risk_decision(self, decision: object) -> None:
        raise AssertionError("reconciliation must not save risk decisions")

    def now(self) -> datetime:
        return self.current_time


class FakeHistoryProvider:
    def __init__(
        self,
        *,
        orders: tuple[ExchangeOrderRecord, ...] = (),
        reports: tuple[ExecutionReport, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self.orders = orders
        self.reports = reports
        self.error = error

    def list_exchange_orders_since(
        self, *, since: datetime
    ) -> tuple[ExchangeOrderRecord, ...]:
        if self.error is not None:
            raise self.error
        return self.orders

    def list_execution_reports_since(
        self, *, since: datetime
    ) -> tuple[ExecutionReport, ...]:
        if self.error is not None:
            raise self.error
        return self.reports


class FakeExchangeProvider:
    def __init__(
        self,
        snapshot: ExchangeOrderStateSnapshot,
        *,
        error: Exception | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.error = error

    def get_order_snapshot(self, *, since: datetime) -> ExchangeOrderStateSnapshot:
        if self.error is not None:
            raise self.error
        return self.snapshot


class FakeReconciliationStore:
    def __init__(self) -> None:
        self.begun_runs: list[ReconciliationRun] = []
        self.completed_runs: list[ReconciliationRun] = []
        self.findings: list[ReconciliationFinding] = []
        self.fail_begin = False
        self.fail_record = False
        self.fail_complete = False

    def begin_run(self, run: ReconciliationRun) -> None:
        if self.fail_begin:
            raise OSError("begin failed")
        self.begun_runs.append(run)

    def record_finding(self, finding: ReconciliationFinding) -> None:
        if self.fail_record:
            raise OSError("finding persistence failed")
        self.findings.append(finding)

    def complete_run(self, run: ReconciliationRun) -> None:
        if self.fail_complete:
            raise OSError("completion failed")
        self.completed_runs.append(run)

    def latest_run(self) -> ReconciliationRun | None:
        return self.completed_runs[-1] if self.completed_runs else None

    def list_unresolved_findings(self) -> tuple[ReconciliationFinding, ...]:
        return tuple(
            finding
            for finding in self.findings
            if finding.resolution is ReconciliationResolution.UNRESOLVED
        )

    def resolve_finding(
        self,
        *,
        finding_id: str,
        resolution: ReconciliationResolution,
        resolved_at: datetime,
        operator_note: str,
    ) -> ReconciliationFinding:
        for index, finding in enumerate(self.findings):
            if finding.finding_id == finding_id:
                resolved = replace(finding, resolution=resolution)
                self.findings[index] = resolved
                return resolved
        raise KeyError(finding_id)


class FakeAccountProvider:
    def __init__(self, snapshot: LiveAccountSnapshot, error: Exception | None = None):
        self.snapshot = snapshot
        self.error = error

    def get_account_snapshot(self) -> LiveAccountSnapshot:
        if self.error is not None:
            raise self.error
        return self.snapshot


class FakeMarketProvider:
    def __init__(self, snapshot: MarketPriceSnapshot, error: Exception | None = None):
        self.snapshot = snapshot
        self.error = error

    def get_price_snapshot(self, symbol: str) -> MarketPriceSnapshot:
        if self.error is not None:
            raise self.error
        return self.snapshot


def local_order(
    *,
    key: str = "key-1",
    intent_id: str = "intent-1",
    exchange_order_id: str = "order-1",
) -> ExchangeOrderRecord:
    return ExchangeOrderRecord(
        idempotency_key=key,
        intent_id=intent_id,
        exchange_order_id=exchange_order_id,
        accepted_at=NOW - timedelta(minutes=5),
    )


def exchange_order(
    *,
    key: str | None = "key-1",
    intent_id: str | None = "intent-1",
    exchange_order_id: str = "order-1",
    status: ExecutionStatus = ExecutionStatus.ACCEPTED,
    filled_quantity: Decimal | None = None,
) -> ExchangeOrderSnapshot:
    return ExchangeOrderSnapshot(
        exchange_order_id=exchange_order_id,
        idempotency_key=key,
        intent_id=intent_id,
        status=status,
        observed_at=NOW,
        filled_quantity=filled_quantity,
    )


def report(
    *,
    key: str = "key-1",
    intent_id: str = "intent-1",
    exchange_order_id: str | None = "order-1",
    status: ExecutionStatus = ExecutionStatus.ACCEPTED,
    filled_quantity: Decimal | None = None,
) -> ExecutionReport:
    return ExecutionReport(
        idempotency_key=key,
        intent_id=intent_id,
        exchange_order_id=exchange_order_id,
        status=status,
        reported_at=NOW,
        filled_quantity=filled_quantity,
    )


def service(
    *,
    control: FakeControlStore,
    store: FakeReconciliationStore,
    history: FakeHistoryProvider,
    exchange: FakeExchangeProvider,
    max_age: int = 60,
) -> LiveReconciliationService:
    return LiveReconciliationService(
        control_store=control,
        reconciliation_store=store,
        execution_history_provider=history,
        exchange_order_state_provider=exchange,
        max_exchange_snapshot_age_seconds=max_age,
    )


def snapshot(
    *orders: ExchangeOrderSnapshot,
    captured_at: datetime = NOW,
) -> ExchangeOrderStateSnapshot:
    return ExchangeOrderStateSnapshot(captured_at=captured_at, orders=orders)


def test_clean_run_is_completed_and_does_not_change_control_state() -> None:
    control = FakeControlStore()
    store = FakeReconciliationStore()

    result = service(
        control=control,
        store=store,
        history=FakeHistoryProvider(orders=(local_order(),)),
        exchange=FakeExchangeProvider(snapshot(exchange_order())),
    ).run(since=SINCE)

    assert result.run.status is ReconciliationStatus.CLEAN
    assert result.run.completed_at is not None
    assert result.findings == ()
    assert len(store.begun_runs) == 1
    assert store.completed_runs == [result.run]
    assert control.saved_states == []
    assert control.state.reconciliation_required is False


def test_unknown_execution_report_creates_finding_and_lockout() -> None:
    control = FakeControlStore()
    store = FakeReconciliationStore()

    result = service(
        control=control,
        store=store,
        history=FakeHistoryProvider(
            reports=(report(status=ExecutionStatus.UNKNOWN),),
        ),
        exchange=FakeExchangeProvider(snapshot()),
    ).run(since=SINCE)

    assert result.run.status is ReconciliationStatus.FINDINGS_PRESENT
    assert [item.finding_type for item in result.findings] == [
        ReconciliationFindingType.UNKNOWN_SUBMISSION
    ]
    assert control.state.reconciliation_required is True
    assert store.findings == list(result.findings)


def test_missing_and_unexpected_orders_are_durably_recorded_before_return() -> None:
    control = FakeControlStore()
    store = FakeReconciliationStore()

    result = service(
        control=control,
        store=store,
        history=FakeHistoryProvider(orders=(local_order(),)),
        exchange=FakeExchangeProvider(
            snapshot(
                exchange_order(exchange_order_id="unexpected-order", key="key-2"),
            )
        ),
    ).run(since=SINCE)

    assert {
        item.finding_type for item in result.findings
    } == {
        ReconciliationFindingType.MISSING_EXCHANGE_ORDER,
        ReconciliationFindingType.UNEXPECTED_EXCHANGE_ORDER,
    }
    assert store.findings == list(result.findings)
    assert store.completed_runs[-1] == result.run
    assert control.state.reconciliation_required is True


def test_duplicate_exchange_idempotency_key_requires_reconciliation() -> None:
    control = FakeControlStore()
    store = FakeReconciliationStore()

    result = service(
        control=control,
        store=store,
        history=FakeHistoryProvider(),
        exchange=FakeExchangeProvider(
            snapshot(
                exchange_order(exchange_order_id="order-1"),
                exchange_order(exchange_order_id="order-2"),
            )
        ),
    ).run(since=SINCE)

    assert ReconciliationFindingType.DUPLICATE_IDEMPOTENCY_KEY in {
        item.finding_type for item in result.findings
    }
    assert control.state.reconciliation_required is True


def test_terminal_status_and_fill_quantity_mismatches_are_detected() -> None:
    control = FakeControlStore()
    store = FakeReconciliationStore()

    result = service(
        control=control,
        store=store,
        history=FakeHistoryProvider(
            orders=(local_order(),),
            reports=(
                report(
                    status=ExecutionStatus.FILLED,
                    filled_quantity=Decimal("2"),
                ),
            ),
        ),
        exchange=FakeExchangeProvider(
            snapshot(
                exchange_order(
                    status=ExecutionStatus.CANCELLED,
                    filled_quantity=Decimal("1"),
                )
            )
        ),
    ).run(since=SINCE)

    assert {
        item.finding_type for item in result.findings
    } == {
        ReconciliationFindingType.EXECUTION_STATUS_MISMATCH,
        ReconciliationFindingType.FILL_QUANTITY_MISMATCH,
    }


@pytest.mark.parametrize(
    "captured_at",
    (
        NOW - timedelta(seconds=61),
        NOW + timedelta(seconds=1),
    ),
)
def test_stale_or_future_exchange_snapshot_requires_lockout(
    captured_at: datetime,
) -> None:
    control = FakeControlStore()
    store = FakeReconciliationStore()

    result = service(
        control=control,
        store=store,
        history=FakeHistoryProvider(),
        exchange=FakeExchangeProvider(snapshot(captured_at=captured_at)),
        max_age=60,
    ).run(since=SINCE)

    assert [item.finding_type for item in result.findings] == [
        ReconciliationFindingType.SOURCE_DATA_STALE
    ]
    assert control.state.reconciliation_required is True


def test_source_failure_returns_failed_run_and_persists_lockout() -> None:
    control = FakeControlStore()
    store = FakeReconciliationStore()

    result = service(
        control=control,
        store=store,
        history=FakeHistoryProvider(error=ConnectionError("unavailable")),
        exchange=FakeExchangeProvider(snapshot()),
    ).run(since=SINCE)

    assert result.run.status is ReconciliationStatus.FAILED
    assert result.findings[0].finding_type is ReconciliationFindingType.SOURCE_DATA_UNAVAILABLE
    assert store.findings == list(result.findings)
    assert store.completed_runs[-1].status is ReconciliationStatus.FAILED
    assert control.state.reconciliation_required is True


def test_begin_run_failure_locks_out_and_raises() -> None:
    control = FakeControlStore()
    store = FakeReconciliationStore()
    store.fail_begin = True

    with pytest.raises(RuntimeError, match="run persistence failed"):
        service(
            control=control,
            store=store,
            history=FakeHistoryProvider(),
            exchange=FakeExchangeProvider(snapshot()),
        ).run(since=SINCE)

    assert control.state.reconciliation_required is True


def test_finding_persistence_failure_locks_out_and_raises() -> None:
    control = FakeControlStore()
    store = FakeReconciliationStore()
    store.fail_record = True

    with pytest.raises(RuntimeError, match="reconciliation persistence failed"):
        service(
            control=control,
            store=store,
            history=FakeHistoryProvider(
                reports=(report(status=ExecutionStatus.UNKNOWN),),
            ),
            exchange=FakeExchangeProvider(snapshot()),
        ).run(since=SINCE)

    assert control.state.reconciliation_required is True


def test_operational_monitor_reports_ready_only_when_every_health_gate_passes() -> None:
    control = FakeControlStore()
    reconciliation_store = FakeReconciliationStore()
    reconciliation_store.completed_runs.append(
        ReconciliationRun(
            run_id="run-1",
            started_at=NOW - timedelta(minutes=1),
            completed_at=NOW,
            status=ReconciliationStatus.CLEAN,
            exchange_snapshot_at=NOW,
        )
    )

    monitor = OperationalMonitor(
        control_store=control,
        reconciliation_store=reconciliation_store,
        execution_history_provider=FakeHistoryProvider(),
        account_state_provider=FakeAccountProvider(
            LiveAccountSnapshot(
                captured_at=NOW,
                symbol_quantities={"BTC/USD": Decimal("0")},
                symbol_exposures={"BTC/USD": Decimal("0")},
                total_exposure=Decimal("0"),
                open_order_count=0,
            )
        ),
        market_price_provider=FakeMarketProvider(
            MarketPriceSnapshot(
                symbol="BTC/USD",
                price=Decimal("50000"),
                captured_at=NOW,
            )
        ),
        monitoring_symbol="BTC/USD",
        max_account_snapshot_age_seconds=60,
        max_market_snapshot_age_seconds=60,
    )

    status = monitor.get_status()

    assert status.operationally_ready is True
    assert status.latest_reconciliation_status is ReconciliationStatus.CLEAN
    assert status.unresolved_finding_count == 0
    assert status.unknown_execution_count == 0


@pytest.mark.parametrize(
    "state",
    (
        LiveTradingControlState(
            live_trading_enabled=False,
            kill_switch_engaged=False,
            reconciliation_required=False,
            operationally_healthy=True,
        ),
        LiveTradingControlState(
            live_trading_enabled=True,
            kill_switch_engaged=True,
            reconciliation_required=False,
            operationally_healthy=True,
        ),
        LiveTradingControlState(
            live_trading_enabled=True,
            kill_switch_engaged=False,
            reconciliation_required=True,
            operationally_healthy=True,
        ),
    ),
)
def test_operational_monitor_never_reports_ready_when_control_state_blocks(
    state: LiveTradingControlState,
) -> None:
    control = FakeControlStore(state=state)
    reconciliation_store = FakeReconciliationStore()

    monitor = OperationalMonitor(
        control_store=control,
        reconciliation_store=reconciliation_store,
        execution_history_provider=FakeHistoryProvider(),
        account_state_provider=FakeAccountProvider(
            LiveAccountSnapshot(
                captured_at=NOW,
                symbol_quantities={},
                symbol_exposures={},
                total_exposure=Decimal("0"),
                open_order_count=0,
            )
        ),
        market_price_provider=FakeMarketProvider(
            MarketPriceSnapshot(
                symbol="BTC/USD",
                price=Decimal("50000"),
                captured_at=NOW,
            )
        ),
        monitoring_symbol="BTC/USD",
        max_account_snapshot_age_seconds=60,
        max_market_snapshot_age_seconds=60,
    )

    assert monitor.get_status().operationally_ready is False


def test_operational_monitor_rejects_unknown_reports_and_stale_snapshots() -> None:
    control = FakeControlStore()
    reconciliation_store = FakeReconciliationStore()
    reconciliation_store.completed_runs.append(
        ReconciliationRun(
            run_id="run-1",
            started_at=NOW - timedelta(minutes=1),
            completed_at=NOW,
            status=ReconciliationStatus.CLEAN,
        )
    )

    monitor = OperationalMonitor(
        control_store=control,
        reconciliation_store=reconciliation_store,
        execution_history_provider=FakeHistoryProvider(
            reports=(report(status=ExecutionStatus.UNKNOWN),),
        ),
        account_state_provider=FakeAccountProvider(
            LiveAccountSnapshot(
                captured_at=NOW - timedelta(seconds=61),
                symbol_quantities={},
                symbol_exposures={},
                total_exposure=Decimal("0"),
                open_order_count=0,
            )
        ),
        market_price_provider=FakeMarketProvider(
            MarketPriceSnapshot(
                symbol="BTC/USD",
                price=Decimal("50000"),
                captured_at=NOW,
            )
        ),
        monitoring_symbol="BTC/USD",
        max_account_snapshot_age_seconds=60,
        max_market_snapshot_age_seconds=60,
    )

    status = monitor.get_status()

    assert status.unknown_execution_count == 1
    assert status.operationally_ready is False
