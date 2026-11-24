from __future__ import annotations

from datetime import datetime
from typing import Protocol

from src.live_risk.models import (
    LiveAccountSnapshot,
    LiveTradingControlState,
    MarketPriceSnapshot,
    RiskDecision,
)


class LiveTradingControlStore(Protocol):
    """Durable control-plane and audit persistence port."""

    def get_control_state(self) -> LiveTradingControlState:
        """Return the latest durable live-trading control state."""

    def save_control_state(self, state: LiveTradingControlState) -> None:
        """Persist a control-state transition."""

    def save_risk_decision(self, decision: RiskDecision) -> None:
        """Durably persist a risk decision before submission is allowed."""

    def now(self) -> datetime:
        """Return a UTC timestamp from the application or persistence layer."""


class LiveAccountStateProvider(Protocol):
    """Independent account-state source, normally fed by reconciliation."""

    def get_account_snapshot(self) -> LiveAccountSnapshot:
        """Return the latest account and open-order state."""


class MarketPriceProvider(Protocol):
    """Independent market-data source used only for pre-trade valuation."""

    def get_price_snapshot(self, symbol: str) -> MarketPriceSnapshot:
        """Return a current reference price for one symbol."""
