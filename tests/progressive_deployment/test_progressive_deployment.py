from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.execution.models import ExecutionSide, ExecutionStatus, StrategyIntentEvent
from src.live_risk.service import ControlledLiveExecutionService
from src.progressive_deployment.models import (
    DeploymentDecision,
    DeploymentPolicy,
    DeploymentStage,
    DeploymentState,
)
from src.progressive_deployment.service import ProgressiveDeploymentService


class FakeDeploymentStore:
    def __init__(self, stage: DeploymentStage) -> None:
        self.state = DeploymentState(
            stage=stage,
            updated_at=datetime.now(timezone.utc),
        )
        self.decisions: list[DeploymentDecision] = []
        self.updates: list[DeploymentState] = []
        self.events: list[str] = []

    def get_deployment_state(self) -> DeploymentState:
        self.events.append("get_state")
        return self.state

    def update_deployment_state(self, state: DeploymentState) -> None:
        self.events.append("update_state")
        self.state = state
        self.updates.append(state)

    def save_deployment_decision(self, decision: DeploymentDecision) -> None:
        self.events.append(f"save_decision:{decision.approved}")
        self.decisions.append(decision)

    def next_decision_id(self) -> str:
        return f"decision-{len(self.decisions) + 1}"

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class FailingDecisionStore(FakeDeploymentStore):
    def save_deployment_decision(self, decision: DeploymentDecision) -> None:
        self.events.append("save_decision_failed")
        raise RuntimeError("deployment decision persistence unavailable")


class ReadyMonitor:
    def __init__(self, *, ready: bool) -> None:
        self._status = SimpleNamespace(operationally_ready=ready)
        self.calls = 0

    def get_status(self) -> SimpleNamespace:
        self.calls += 1
        return self._status


class RecordingRoute:
    def __init__(self, store: FakeDeploymentStore, result: object) -> None:
        self._store = store
        self._result = result
        self.calls: list[StrategyIntentEvent] = []

    def submit_intent(self, intent: StrategyIntentEvent) -> object:
        self._store.events.append("route_submit")
        self.calls.append(intent)
        return self._result


@pytest.fixture
def intent() -> StrategyIntentEvent:
    return StrategyIntentEvent(
        intent_id="intent-1",
        strategy_name="progressive-deployment-test",
        symbol="BTC/USD",
        side=ExecutionSide.BUY,
        quantity=Decimal("1"),
        created_at=datetime.now(timezone.utc),
    )


def policy_for(stage: DeploymentStage) -> DeploymentPolicy:
    return DeploymentPolicy(
        stage=stage,
        symbol_allowlist=frozenset({"BTC/USD"}),
        max_open_orders=1,
    )


@pytest.mark.parametrize(
    "stage",
    [
        DeploymentStage.DISABLED,
        DeploymentStage.PREFLIGHT,
        DeploymentStage.PAUSED,
        DeploymentStage.ROLLED_BACK,
    ],
)
def test_rejects_all_inactive_stages_without_calling_route(
    intent: StrategyIntentEvent,
    stage: DeploymentStage,
) -> None:
    store = FakeDeploymentStore(stage)
    monitor = ReadyMonitor(ready=True)
    route = MagicMock(spec=ControlledLiveExecutionService)

    service = ProgressiveDeploymentService(
        store=store,
        monitor=monitor,
        policies={stage: policy_for(stage)},
        stage_routes={stage: route},
    )

    result = service.submit_intent(intent)

    assert result.execution_result is None
    assert result.deployment_decision.approved is False
    assert "not active for trading" in result.deployment_decision.reason
    assert len(store.decisions) == 1
    assert store.decisions[0] == result.deployment_decision
    assert monitor.calls == 0
    route.submit_intent.assert_not_called()


def test_rejects_when_operational_readiness_is_false(
    intent: StrategyIntentEvent,
) -> None:
    store = FakeDeploymentStore(DeploymentStage.CANARY)
    monitor = ReadyMonitor(ready=False)
    route = MagicMock(spec=ControlledLiveExecutionService)

    service = ProgressiveDeploymentService(
        store=store,
        monitor=monitor,
        policies={DeploymentStage.CANARY: policy_for(DeploymentStage.CANARY)},
        stage_routes={DeploymentStage.CANARY: route},
    )

    result = service.submit_intent(intent)

    assert result.execution_result is None
    assert result.deployment_decision.approved is False
    assert result.deployment_decision.reason == "Operational health check failed"
    route.submit_intent.assert_not_called()


@pytest.mark.parametrize(
    ("policies", "expected_reason"),
    [
        ({}, "No policy defined"),
        (
            {
                DeploymentStage.CANARY: DeploymentPolicy(
                    stage=DeploymentStage.LIMITED,
                    symbol_allowlist=frozenset({"BTC/USD"}),
                    max_open_orders=1,
                )
            },
            "Policy stage does not match active stage",
        ),
    ],
)
def test_rejects_missing_or_mismatched_policy_without_calling_route(
    intent: StrategyIntentEvent,
    policies: dict[DeploymentStage, DeploymentPolicy],
    expected_reason: str,
) -> None:
    store = FakeDeploymentStore(DeploymentStage.CANARY)
    monitor = ReadyMonitor(ready=True)
    route = MagicMock(spec=ControlledLiveExecutionService)

    service = ProgressiveDeploymentService(
        store=store,
        monitor=monitor,
        policies=policies,
        stage_routes={DeploymentStage.CANARY: route},
    )

    result = service.submit_intent(intent)

    assert result.execution_result is None
    assert result.deployment_decision.approved is False
    assert expected_reason in result.deployment_decision.reason
    route.submit_intent.assert_not_called()


def test_rejects_symbol_outside_stage_allowlist_without_calling_route(
    intent: StrategyIntentEvent,
) -> None:
    store = FakeDeploymentStore(DeploymentStage.CANARY)
    monitor = ReadyMonitor(ready=True)
    route = MagicMock(spec=ControlledLiveExecutionService)

    policy = DeploymentPolicy(
        stage=DeploymentStage.CANARY,
        symbol_allowlist=frozenset({"ETH/USD"}),
        max_open_orders=1,
    )

    service = ProgressiveDeploymentService(
        store=store,
        monitor=monitor,
        policies={DeploymentStage.CANARY: policy},
        stage_routes={DeploymentStage.CANARY: route},
    )

    result = service.submit_intent(intent)

    assert result.execution_result is None
    assert result.deployment_decision.approved is False
    assert "not in allowlist" in result.deployment_decision.reason
    route.submit_intent.assert_not_called()


def test_rejects_missing_execution_route(
    intent: StrategyIntentEvent,
) -> None:
    store = FakeDeploymentStore(DeploymentStage.CANARY)
    monitor = ReadyMonitor(ready=True)

    service = ProgressiveDeploymentService(
        store=store,
        monitor=monitor,
        policies={DeploymentStage.CANARY: policy_for(DeploymentStage.CANARY)},
        stage_routes={},
    )

    result = service.submit_intent(intent)

    assert result.execution_result is None
    assert result.deployment_decision.approved is False
    assert "No execution route configured" in result.deployment_decision.reason
    assert len(store.decisions) == 1


@pytest.mark.parametrize(
    "stage",
    [DeploymentStage.CANARY, DeploymentStage.LIMITED],
)
def test_active_stages_persist_approval_before_routing(
    intent: StrategyIntentEvent,
    stage: DeploymentStage,
) -> None:
    store = FakeDeploymentStore(stage)
    monitor = ReadyMonitor(ready=True)
    controlled_result = SimpleNamespace(execution_result=None)
    route = RecordingRoute(store, controlled_result)

    service = ProgressiveDeploymentService(
        store=store,
        monitor=monitor,
        policies={stage: policy_for(stage)},
        stage_routes={stage: route},
    )

    result = service.submit_intent(intent)

    assert result.deployment_decision.approved is True
    assert result.deployment_decision.reason == "Deployment policy satisfied"
    assert result.execution_result is controlled_result
    assert route.calls == [intent]
    assert store.events.index("save_decision:True") < store.events.index("route_submit")


def test_unknown_execution_outcome_durably_rolls_back(
    intent: StrategyIntentEvent,
) -> None:
    store = FakeDeploymentStore(DeploymentStage.CANARY)
    monitor = ReadyMonitor(ready=True)

    unknown_execution = SimpleNamespace(
        report=SimpleNamespace(status=ExecutionStatus.UNKNOWN)
    )
    controlled_result = SimpleNamespace(execution_result=unknown_execution)
    route = RecordingRoute(store, controlled_result)

    service = ProgressiveDeploymentService(
        store=store,
        monitor=monitor,
        policies={DeploymentStage.CANARY: policy_for(DeploymentStage.CANARY)},
        stage_routes={DeploymentStage.CANARY: route},
    )

    result = service.submit_intent(intent)

    assert result.execution_result is controlled_result
    assert len(route.calls) == 1
    assert len(store.updates) == 1
    assert store.state.stage is DeploymentStage.ROLLED_BACK
    assert "Automatic rollback: Unknown execution outcome in CANARY" == (
        store.state.operator_note
    )
    assert store.events.index("route_submit") < store.events.index("update_state")


def test_decision_persistence_failure_fails_closed_without_routing(
    intent: StrategyIntentEvent,
) -> None:
    store = FailingDecisionStore(DeploymentStage.CANARY)
    monitor = ReadyMonitor(ready=True)
    route = MagicMock(spec=ControlledLiveExecutionService)

    service = ProgressiveDeploymentService(
        store=store,
        monitor=monitor,
        policies={DeploymentStage.CANARY: policy_for(DeploymentStage.CANARY)},
        stage_routes={DeploymentStage.CANARY: route},
    )

    with pytest.raises(RuntimeError, match="deployment decision persistence unavailable"):
        service.submit_intent(intent)

    route.submit_intent.assert_not_called()
