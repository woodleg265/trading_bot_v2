from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.execution.models import _require_utc_timestamp


class DeploymentStage(Enum):
    """Progressive funding and risk stages."""

    DISABLED = "DISABLED"
    PREFLIGHT = "PREFLIGHT"
    CANARY = "CANARY"
    LIMITED = "LIMITED"
    PAUSED = "PAUSED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(frozen=True, slots=True)
class DeploymentState:
    """Durable state of the deployment pipeline."""

    stage: DeploymentStage
    updated_at: datetime
    operator_note: str | None = None

    def __post_init__(self) -> None:
        _require_utc_timestamp(self.updated_at, "updated_at")


@dataclass(frozen=True, slots=True)
class DeploymentPolicy:
    """Constraints for a specific deployment stage."""

    stage: DeploymentStage
    symbol_allowlist: frozenset[str]
    max_open_orders: int


@dataclass(frozen=True, slots=True)
class DeploymentDecision:
    """Audit record of a deployment-level approval or rejection."""

    decision_id: str
    intent_id: str
    stage: DeploymentStage
    approved: bool
    reason: str
    decided_at: datetime

    def __post_init__(self) -> None:
        _require_utc_timestamp(self.decided_at, "decided_at")
