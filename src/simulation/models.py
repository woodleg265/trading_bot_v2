"""Immutable contracts for the isolated virtual trading environment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """A request to simulate an order; never an exchange order."""

    intent_id: str
    symbol: str
    side: Side
    quantity: Decimal
    reference_price: Decimal
    market_data_timestamp: datetime
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.intent_id:
            raise ValueError("intent_id is required")
        if not self.symbol:
            raise ValueError("symbol is required")
        if self.quantity <= Decimal("0"):
            raise ValueError("quantity must be greater than zero")
        if self.reference_price <= Decimal("0"):
            raise ValueError("reference_price must be greater than zero")


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """The deterministic result of evaluating an OrderIntent."""

    approved: bool
    code: str
    reason: str
    evaluated_notional: Decimal


@dataclass(frozen=True, slots=True)
class SimulatedFill:
    """A virtual fill generated solely by the simulation fill model."""

    fill_id: str
    intent_id: str
    symbol: str
    side: Side
    quantity: Decimal
    market_price: Decimal
    fill_price: Decimal
    fee: Decimal
    filled_at: datetime

    def __post_init__(self) -> None:
        if not self.fill_id:
            raise ValueError("fill_id is required")
        if not self.intent_id:
            raise ValueError("intent_id is required")
        if self.quantity <= Decimal("0"):
            raise ValueError("quantity must be greater than zero")
        if self.market_price <= Decimal("0"):
            raise ValueError("market_price must be greater than zero")
        if self.fill_price <= Decimal("0"):
            raise ValueError("fill_price must be greater than zero")
        if self.fee < Decimal("0"):
            raise ValueError("fee cannot be negative")


@dataclass(frozen=True, slots=True)
class SimulationPosition:
    """An immutable virtual position, unrelated to real account holdings."""

    symbol: str
    quantity: Decimal
    average_entry_price: Decimal
    realized_pnl: Decimal
