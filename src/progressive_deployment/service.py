from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

from src.execution.models import ExecutionStatus, StrategyIntentEvent
from src.live_reconciliation.monitoring import OperationalMonitor
from src.live_risk.service import (
    ControlledExecutionResult,
    ControlledLiveExecutionService,
)

from .models import (
    DeploymentDecision,
    DeploymentPolicy,
    DeploymentStage,
    DeploymentState,
)
from .ports import DeploymentStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DeploymentSubmissionResult:
    """
    Outcome of deployment-stage gating.

    execution_result is None when the deployment boundary rejects the intent
    before the live-risk/execution boundary is called.
    """

    deployment_decision: DeploymentDecision
    execution_result: ControlledExecutionResult | None

    @property
    def submitted(self) -> bool:
        return (
            self.execution_result is not None
            and self.execution_result.submitted
        )


class ProgressiveDeploymentService:
    """Orchestrates manually staged, operationally gated live execution."""

    _ACTIVE_STAGES = frozenset(
        {
            DeploymentStage.CANARY,
            DeploymentStage.LIMITED,
        }
    )

    def __init__(
        self,
        *,
        store: DeploymentStore,
        monitor: OperationalMonitor,
        policies: Mapping[DeploymentStage, DeploymentPolicy],
        stage_routes: Mapping[DeploymentStage, ControlledLiveExecutionService],
    ) -> None:
        self._store = store
        self._monitor = monitor
        self._policies = policies
        self._stage_routes = stage_routes

    def submit_intent(
        self,
        intent: StrategyIntentEvent,
    ) -> DeploymentSubmissionResult:
        state = self._store.get_deployment_state()

        if state.stage not in self._ACTIVE_STAGES:
            return self._reject(
                intent,
                state.stage,
                f"Deployment stage {state.stage.value} is not active for trading",
            )

        operational_status = self._monitor.get_status()
        if not operational_status.operationally_ready:
            return self._reject(
                intent,
                state.stage,
                "Operational health check failed",
            )

        policy = self._policies.get(state.stage)
        if policy is None:
            return self._reject(
                intent,
                state.stage,
                f"No policy defined for stage {state.stage.value}",
            )

        if policy.stage is not state.stage:
            return self._reject(
                intent,
                state.stage,
                f"Policy stage does not match active stage {state.stage.value}",
            )

        if intent.symbol not in policy.symbol_allowlist:
            return self._reject(
                intent,
                state.stage,
                f"Symbol {intent.symbol} not in allowlist for {state.stage.value}",
            )

        route = self._stage_routes.get(state.stage)
        if route is None:
            return self._reject(
                intent,
                state.stage,
                f"No execution route configured for stage {state.stage.value}",
            )

        # The approval is durably stored before live-risk or gateway access.
        approval = self._approve(intent, state.stage)

        result = route.submit_intent(intent)

        if (
            result.execution_result is not None
            and result.execution_result.report.status is ExecutionStatus.UNKNOWN
        ):
            self._trigger_rollback(
                f"Unknown execution outcome in {state.stage.value}"
            )

        return DeploymentSubmissionResult(
            deployment_decision=approval,
            execution_result=result,
        )

    def _reject(
        self,
        intent: StrategyIntentEvent,
        stage: DeploymentStage,
        reason: str,
    ) -> DeploymentSubmissionResult:
        decision = DeploymentDecision(
            decision_id=self._store.next_decision_id(),
            intent_id=intent.intent_id,
            stage=stage,
            approved=False,
            reason=reason,
            decided_at=self._store.now(),
        )
        self._store.save_deployment_decision(decision)

        return DeploymentSubmissionResult(
            deployment_decision=decision,
            execution_result=None,
        )

    def _approve(
        self,
        intent: StrategyIntentEvent,
        stage: DeploymentStage,
    ) -> DeploymentDecision:
        decision = DeploymentDecision(
            decision_id=self._store.next_decision_id(),
            intent_id=intent.intent_id,
            stage=stage,
            approved=True,
            reason="Deployment policy satisfied",
            decided_at=self._store.now(),
        )
        self._store.save_deployment_decision(decision)
        return decision

    def _trigger_rollback(self, reason: str) -> None:
        new_state = DeploymentState(
            stage=DeploymentStage.ROLLED_BACK,
            updated_at=self._store.now(),
            operator_note=f"Automatic rollback: {reason}",
        )
        self._store.update_deployment_state(new_state)
        logger.error("Deployment ROLLED_BACK: %s", reason)
