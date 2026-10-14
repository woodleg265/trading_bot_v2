"""Deterministic risk controls for the isolated virtual trading environment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import FrozenSet, Mapping

from .models import OrderIntent, RiskDecision, Side


ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class SimulationRiskLimits:
    """Immutable hard limits for one simulation run."""

    allowed_symbols: FrozenSet[str]
    max_order_notional: Decimal
    max_position_notional: Decimal
    max_total_exposure: Decimal
    max_market_data_age: timedelta
    estimated_fee_rate: Decimal = ZERO

    def __post_init__(self) -> None:
        if not self.allowed_symbols:
            raise ValueError("allowed_symbols must not be empty")
        if self.max_order_notional <= ZERO:
            raise ValueError("max_order_notional must be greater than zero")
        if self.max_position_notional <= ZERO:
            raise ValueError("max_position_notional must be greater than zero")
        if self.max_total_exposure <= ZERO:
            raise ValueError("max_total_exposure must be greater than zero")
        if self.max_market_data_age < timedelta(0):
            raise ValueError("max_market_data_age cannot be negative")
        if self.estimated_fee_rate < ZERO:
            raise ValueError("estimated_fee_rate cannot be negative")


@dataclass(frozen=True, slots=True)
class SimulationRiskContext:
    """Read-only virtual-account values used to evaluate one order intent."""

    available_cash: Decimal
    position_quantities: Mapping[str, Decimal]
    current_total_exposure: Decimal
    processed_intent_ids: FrozenSet[str]

    def __post_init__(self) -> None:
        if self.available_cash < ZERO:
            raise ValueError("available_cash cannot be negative")
        if self.current_total_exposure < ZERO:
            raise ValueError("current_total_exposure cannot be negative")

        for symbol, quantity in self.position_quantities.items():
            if not symbol:
                raise ValueError("position symbol cannot be empty")
            if quantity < ZERO:
                raise ValueError("position quantity cannot be negative")


class SimulationRiskEngine:
    """Evaluates virtual intents against non-negotiable deterministic limits."""

    def __init__(self, limits: SimulationRiskLimits) -> None:
        self._limits = limits

    def evaluate(
        self,
        intent: OrderIntent,
        *,
        context: SimulationRiskContext,
        evaluated_at: datetime,
    ) -> RiskDecision:
        """Return an approval or a structured rejection; never submit an order."""

        notional = intent.quantity * intent.reference_price

        if not intent.intent_id or not intent.symbol:
            return self._reject(
                "INVALID_INTENT",
                "Intent requires non-empty intent_id and symbol.",
                notional,
            )

        if intent.intent_id in context.processed_intent_ids:
            return self._reject(
                "DUPLICATE_INTENT",
                "Intent ID has already been processed in this simulation run.",
                notional,
            )

        if intent.symbol not in self._limits.allowed_symbols:
            return self._reject(
                "SYMBOL_NOT_ALLOWED",
                "Symbol is not included in the simulation allowlist.",
                notional,
            )

        market_data_age = evaluated_at - intent.market_data_timestamp
        if market_data_age < timedelta(0) or market_data_age > self._limits.max_market_data_age:
            return self._reject(
                "STALE_MARKET_DATA",
                "Market data is outside the permitted simulation freshness window.",
                notional,
            )

        if notional > self._limits.max_order_notional:
            return self._reject(
                "MAX_ORDER_NOTIONAL_EXCEEDED",
                "Intent notional exceeds the configured maximum order notional.",
                notional,
            )

        current_quantity = context.position_quantities.get(intent.symbol, ZERO)

        if intent.side is Side.SELL and intent.quantity > current_quantity:
            return self._reject(
                "INSUFFICIENT_SIMULATED_POSITION",
                "Sell quantity exceeds the available virtual position.",
                notional,
            )

        if intent.side is Side.BUY:
            resulting_position_notional = (
                current_quantity + intent.quantity
            ) * intent.reference_price
        else:
            resulting_position_notional = (
                current_quantity - intent.quantity
            ) * intent.reference_price

        if resulting_position_notional > self._limits.max_position_notional:
            return self._reject(
                "MAX_POSITION_NOTIONAL_EXCEEDED",
                "Resulting virtual position exceeds the configured position limit.",
                notional,
            )

        if intent.side is Side.BUY:
            resulting_total_exposure = context.current_total_exposure + notional
        else:
            resulting_total_exposure = max(ZERO, context.current_total_exposure - notional)

        if resulting_total_exposure > self._limits.max_total_exposure:
            return self._reject(
                "MAX_TOTAL_EXPOSURE_EXCEEDED",
                "Resulting virtual exposure exceeds the configured exposure limit.",
                notional,
            )

        if intent.side is Side.BUY:
            estimated_fee = notional * self._limits.estimated_fee_rate
            estimated_total_cost = notional + estimated_fee

            if estimated_total_cost > context.available_cash:
                return self._reject(
                    "INSUFFICIENT_VIRTUAL_CASH",
                    "Buy cost including estimated fees exceeds available virtual cash.",
                    notional,
                )

        return RiskDecision(
            approved=True,
            code="APPROVED",
            reason="Intent satisfies all virtual simulation risk controls.",
            evaluated_notional=notional,
        )

    @staticmethod
    def _reject(code: str, reason: str, notional: Decimal) -> RiskDecision:
        return RiskDecision(
            approved=False,
            code=code,
            reason=reason,
            evaluated_notional=notional,
        )
