from __future__ import annotations

from datetime import datetime
from typing import Protocol

from .models import DeploymentDecision, DeploymentState


class DeploymentStore(Protocol):
    """Persistence for deployment state and decisions."""

    def get_deployment_state(self) -> DeploymentState:
        """Return the current active deployment stage and metadata."""

    def update_deployment_state(self, state: DeploymentState) -> None:
        """Persist a deployment-stage transition."""

    def save_deployment_decision(self, decision: DeploymentDecision) -> None:
        """Persist an audit trail of an approval or rejection."""

    def next_decision_id(self) -> str:
        """Generate a unique decision identifier."""

    def now(self) -> datetime:
        """Return the current UTC timestamp."""
