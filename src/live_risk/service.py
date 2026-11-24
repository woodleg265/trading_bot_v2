from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from decimal import Decimal

from src.execution.models import ExecutionSide, ExecutionStatus, StrategyIntentEvent
from src.execution.service import ExecutionSubmissionResult, LiveExecutionService
from src.live_risk.models import (
    LiveAccountSnapshot,
    LiveRiskLimits,
    LiveTradingControlState,
    MarketPriceSnapshot,
    RiskDecision,
    RiskDecisionStatus,
)
from src.live_risk.ports import (
    LiveAccountStateProvider,
    LiveTradingControlStore,
    MarketPriceProvider,
)


@dataclass(frozen=True, slots=True)
class ControlledExecutionResult:
    risk_decision: RiskDecision
    execution_result: ExecutionSubmissionResult | None

    @property
    def submitted(self) -> bool:
        return self.execution_result is not None


class LiveRiskControlService:
    """Independent, fail-closed pre-trade risk evaluator."""

    def __init__(
        self,
        *,
        limits: LiveRiskLimits,
        control_store: LiveTradingControlStore,
        account_state_provider: LiveAccountStateProvider,
        market_price_provider: MarketPriceProvider,
    ) -> None:
        self._limits = limits
        self._control_store = control_store
        self._account_state_provider = account_state_provider
        self._market_price_provider = market_price_provider

    def evaluate(self, intent: StrategyIntentEvent) -> RiskDecision:
        now = self._control_store.now()
        state = self._control_store.get_control_state()

        reason = self._control_state_rejection_reason(state)
        if reason is not None:
            return self._persist_decision(
                intent=intent,
                status=RiskDecisionStatus.REJECTED,
                reason=reason,
                decided_at=now,
            )

        if intent.symbol not in self._limits.allowed_symbols:
            return self._persist_decision(
                intent=intent,
                status=RiskDecisionStatus.REJECTED,
                reason="SYMBOL_NOT_ALLOWED",
                decided_at=now,
            )

        try:
            account = self._account_state_provider.get_account_snapshot()
            price = self._market_price_provider.get_price_snapshot(intent.symbol)
        except Exception as error:
            return self._persist_decision(
                intent=intent,
                status=RiskDecisionStatus.REJECTED,
                reason=f"DEPENDENCY_UNAVAILABLE:{type(error).__name__}",
                decided_at=now,
            )

        reason = self._freshness_rejection_reason(
            now=now,
            account=account,
            price=price,
        )
        if reason is not None:
            return self._persist_decision(
                intent=intent,
                status=RiskDecisionStatus.REJECTED,
                reason=reason,
                decided_at=now,
            )

        order_notional = intent.quantity * price.price

        reason = self._limit_rejection_reason(
            intent=intent,
            account=account,
            price=price,
            order_notional=order_notional,
        )
        if reason is not None:
            return self._persist_decision(
                intent=intent,
                status=RiskDecisionStatus.REJECTED,
                reason=reason,
                decided_at=now,
                order_notional=order_notional,
            )

        return self._persist_decision(
            intent=intent,
            status=RiskDecisionStatus.APPROVED,
            reason="APPROVED",
            decided_at=now,
            order_notional=order_notional,
        )

    def _persist_decision(
        self,
        *,
        intent: StrategyIntentEvent,
        status: RiskDecisionStatus,
        reason: str,
        decided_at,
        order_notional: Decimal | None = None,
    ) -> RiskDecision:
        decision = RiskDecision(
            intent_id=intent.intent_id,
            status=status,
            reason=reason,
            decided_at=decided_at,
            order_notional=order_notional,
        )
        try:
            self._control_store.save_risk_decision(decision)
        except Exception as error:
            raise RuntimeError(
                "risk decision persistence failed; submission is blocked"
            ) from error
        return decision

    @staticmethod
    def _control_state_rejection_reason(
        state: LiveTradingControlState,
    ) -> str | None:
        if not state.live_trading_enabled:
            return "LIVE_TRADING_DISABLED"
        if state.kill_switch_engaged:
            return "KILL_SWITCH_ENGAGED"
        if state.reconciliation_required:
            return "RECONCILIATION_REQUIRED"
        if not state.operationally_healthy:
            return "OPERATIONAL_HEALTH_UNAVAILABLE"
        return None

    def _freshness_rejection_reason(
        self,
        *,
        now,
        account: LiveAccountSnapshot,
        price: MarketPriceSnapshot,
    ) -> str | None:
        if account.captured_at > now or price.captured_at > now:
            return "FUTURE_DATED_SNAPSHOT"

        if now - account.captured_at > timedelta(
            seconds=self._limits.max_account_snapshot_age_seconds
        ):
            return "ACCOUNT_SNAPSHOT_STALE"

        if now - price.captured_at > timedelta(
            seconds=self._limits.max_market_data_age_seconds
        ):
            return "MARKET_PRICE_STALE"

        return None

    def _limit_rejection_reason(
        self,
        *,
        intent: StrategyIntentEvent,
        account: LiveAccountSnapshot,
        price: MarketPriceSnapshot,
        order_notional: Decimal,
    ) -> str | None:
        if account.open_order_count >= self._limits.max_open_orders:
            return "OPEN_ORDER_LIMIT_REACHED"
        if order_notional > self._limits.max_order_notional:
            return "ORDER_NOTIONAL_LIMIT_EXCEEDED"

        current_symbol_exposure = account.symbol_exposures.get(
            intent.symbol,
            Decimal("0"),
        )
        current_quantity = account.symbol_quantities.get(intent.symbol, Decimal("0"))

        if intent.side is ExecutionSide.BUY:
            projected_symbol_exposure = current_symbol_exposure + order_notional
            projected_total_exposure = account.total_exposure + order_notional
        else:
            reduced_quantity = min(current_quantity, intent.quantity)
            reduced_exposure = reduced_quantity * price.price
            projected_symbol_exposure = max(
                Decimal("0"),
                current_symbol_exposure - reduced_exposure,
            )
            projected_total_exposure = max(
                Decimal("0"),
                account.total_exposure - reduced_exposure,
            )

        if projected_symbol_exposure > self._limits.max_symbol_exposure:
            return "SYMBOL_EXPOSURE_LIMIT_EXCEEDED"
        if projected_total_exposure > self._limits.max_total_exposure:
            return "TOTAL_EXPOSURE_LIMIT_EXCEEDED"
        return None


class ControlledLiveExecutionService:
    """Only route through which live intent submission should be exposed."""

    def __init__(
        self,
        *,
        risk_control_service: LiveRiskControlService,
        live_execution_service: LiveExecutionService,
        control_store: LiveTradingControlStore,
    ) -> None:
        self._risk_control_service = risk_control_service
        self._live_execution_service = live_execution_service
        self._control_store = control_store

    def submit_intent(self, intent: StrategyIntentEvent) -> ControlledExecutionResult:
        decision = self._risk_control_service.evaluate(intent)
        if decision.status is not RiskDecisionStatus.APPROVED:
            return ControlledExecutionResult(
                risk_decision=decision,
                execution_result=None,
            )

        result = self._live_execution_service.submit_intent(intent)

        if result.report.status is ExecutionStatus.UNKNOWN:
            current_state = self._control_store.get_control_state()
            self._control_store.save_control_state(
                replace(current_state, reconciliation_required=True)
            )

        return ControlledExecutionResult(
            risk_decision=decision,
            execution_result=result,
        )
