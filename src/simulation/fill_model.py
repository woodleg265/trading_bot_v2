"""Deterministic fill generation for the isolated virtual trading environment."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .models import OrderIntent, RiskDecision, Side, SimulatedFill


ZERO = Decimal("0")
ONE = Decimal("1")


@dataclass(frozen=True, slots=True)
class SimulationFillConfig:
    """Immutable fee and adverse-slippage configuration for a simulation run."""

    fee_rate: Decimal
    slippage_rate: Decimal

    def __post_init__(self) -> None:
        if self.fee_rate < ZERO:
            raise ValueError("fee_rate cannot be negative")
        if self.slippage_rate < ZERO:
            raise ValueError("slippage_rate cannot be negative")


class SimulationFillModel:
    """Creates deterministic virtual fills for risk-approved order intents."""

    def __init__(self, config: SimulationFillConfig) -> None:
        self._config = config

    def create_fill(
        self,
        intent: OrderIntent,
        *,
        risk_decision: RiskDecision,
        run_id: str,
        sequence: int,
    ) -> SimulatedFill:
        """Create a virtual fill without persistence, network, or state changes."""

        if not risk_decision.approved:
            raise ValueError(
                "Cannot create a simulated fill for a rejected risk decision."
            )

        if not run_id:
            raise ValueError("run_id must not be empty")

        if sequence <= 0:
            raise ValueError("sequence must be greater than zero")

        fill_price = self._calculate_fill_price(intent)

        if fill_price <= ZERO:
            raise ValueError("slippage produced a zero or negative fill price")

        executed_notional = intent.quantity * fill_price
        fee = executed_notional * self._config.fee_rate

        return SimulatedFill(
            fill_id=f"{run_id}:fill:{sequence:06d}",
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            market_price=intent.reference_price,
            fill_price=fill_price,
            fee=fee,
            filled_at=intent.created_at,
        )

    def _calculate_fill_price(self, intent: OrderIntent) -> Decimal:
        if intent.side is Side.BUY:
            return intent.reference_price * (ONE + self._config.slippage_rate)

        if intent.side is Side.SELL:
            return intent.reference_price * (ONE - self._config.slippage_rate)

        raise ValueError(f"Unsupported simulation side: {intent.side!r}")
