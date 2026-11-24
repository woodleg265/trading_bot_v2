from __future__ import annotations

from datetime import timedelta

from src.execution.models import ExecutionStatus
from src.live_reconciliation.models import OperationalStatus, ReconciliationStatus
from src.live_reconciliation.ports import ExecutionHistoryProvider, ReconciliationStore
from src.live_risk.ports import (
    LiveAccountStateProvider,
    LiveTradingControlStore,
    MarketPriceProvider,
)


class OperationalMonitor:
    """Read-only aggregation; it never changes control state or submits orders."""

    def __init__(
        self,
        *,
        control_store: LiveTradingControlStore,
        reconciliation_store: ReconciliationStore,
        execution_history_provider: ExecutionHistoryProvider,
        account_state_provider: LiveAccountStateProvider,
        market_price_provider: MarketPriceProvider,
        monitoring_symbol: str,
        max_account_snapshot_age_seconds: int,
        max_market_snapshot_age_seconds: int,
    ) -> None:
        self._control_store = control_store
        self._reconciliation_store = reconciliation_store
        self._execution_history_provider = execution_history_provider
        self._account_state_provider = account_state_provider
        self._market_price_provider = market_price_provider
        self._monitoring_symbol = monitoring_symbol
        self._max_account_snapshot_age_seconds = max_account_snapshot_age_seconds
        self._max_market_snapshot_age_seconds = max_market_snapshot_age_seconds

    def get_status(self) -> OperationalStatus:
        now = self._control_store.now()
        state = self._control_store.get_control_state()
        latest_run = self._reconciliation_store.latest_run()
        unresolved = self._reconciliation_store.list_unresolved_findings()

        account_at = None
        market_at = None
        snapshots_fresh = False
        unknown_count = 0

        try:
            account = self._account_state_provider.get_account_snapshot()
            market = self._market_price_provider.get_price_snapshot(
                self._monitoring_symbol
            )
            account_at = account.captured_at
            market_at = market.captured_at
            snapshots_fresh = (
                account_at <= now
                and market_at <= now
                and now - account_at
                <= timedelta(seconds=self._max_account_snapshot_age_seconds)
                and now - market_at
                <= timedelta(seconds=self._max_market_snapshot_age_seconds)
            )
            reports = self._execution_history_provider.list_execution_reports_since(
                since=now - timedelta(days=3650)
            )
            unknown_count = sum(
                report.status is ExecutionStatus.UNKNOWN for report in reports
            )
        except Exception:
            snapshots_fresh = False

        latest_status = latest_run.status if latest_run is not None else None
        ready = (
            state.live_trading_enabled
            and not state.kill_switch_engaged
            and not state.reconciliation_required
            and state.operationally_healthy
            and latest_status is ReconciliationStatus.CLEAN
            and not unresolved
            and unknown_count == 0
            and snapshots_fresh
        )

        return OperationalStatus(
            live_trading_enabled=state.live_trading_enabled,
            kill_switch_engaged=state.kill_switch_engaged,
            reconciliation_required=state.reconciliation_required,
            latest_reconciliation_status=latest_status,
            last_reconciliation_completed_at=(
                latest_run.completed_at if latest_run is not None else None
            ),
            unresolved_finding_count=len(unresolved),
            unknown_execution_count=unknown_count,
            last_account_snapshot_at=account_at,
            last_market_snapshot_at=market_at,
            operationally_ready=ready,
        )
