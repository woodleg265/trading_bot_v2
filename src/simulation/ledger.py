"""Immutable virtual-cash and position transitions for simulations only."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping

from src.simulation.models import Side, SimulatedFill


ZERO = Decimal("0")


def _require_decimal(value: object, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    return value


def _parse_symbol(symbol: str) -> tuple[str, str]:
    if not isinstance(symbol, str):
        raise TypeError("symbol must be a string")

    parts = symbol.split("/")
    if len(parts) != 2:
        raise ValueError("symbol must use BASE/QUOTE format")

    base_asset, quote_asset = (part.strip().upper() for part in parts)
    if not base_asset or not quote_asset:
        raise ValueError("symbol must use non-empty BASE/QUOTE assets")

    return base_asset, quote_asset


@dataclass(frozen=True, slots=True)
class SimulationLedgerState:
    """Immutable virtual portfolio state; unrelated to real balances."""

    cash_balance: Decimal
    cash_currency: str
    position_quantities: Mapping[str, Decimal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_decimal(self.cash_balance, "cash_balance")
        if self.cash_balance < ZERO:
            raise ValueError("cash_balance cannot be negative")

        if not isinstance(self.cash_currency, str) or not self.cash_currency.strip():
            raise ValueError("cash_currency is required")

        normalized_positions: dict[str, Decimal] = {}
        for asset, quantity in self.position_quantities.items():
            if not isinstance(asset, str) or not asset.strip():
                raise ValueError("position asset names must be non-empty strings")
            _require_decimal(quantity, f"position quantity for {asset}")
            if quantity < ZERO:
                raise ValueError("position quantities cannot be negative")
            normalized_positions[asset.strip().upper()] = quantity

        object.__setattr__(self, "cash_currency", self.cash_currency.strip().upper())
        object.__setattr__(
            self,
            "position_quantities",
            MappingProxyType(normalized_positions),
        )


@dataclass(frozen=True, slots=True)
class SimulationLedgerEntry:
    """Deterministic audit-ready record of one virtual ledger transition."""

    entry_id: str
    run_id: str
    sequence: int
    fill_id: str
    symbol: str
    side: Side
    quantity: Decimal
    executed_notional: Decimal
    fee: Decimal
    cash_delta: Decimal
    position_delta: Decimal
    resulting_cash_balance: Decimal
    resulting_position_quantity: Decimal


class SimulationLedger:
    """Applies validated virtual fills without I/O or operational dependencies."""

    def apply_fill(
        self,
        state: SimulationLedgerState,
        fill: SimulatedFill,
        *,
        run_id: str,
        sequence: int,
    ) -> tuple[SimulationLedgerState, SimulationLedgerEntry]:
        if not isinstance(state, SimulationLedgerState):
            raise TypeError("state must be a SimulationLedgerState")
        if not isinstance(fill, SimulatedFill):
            raise TypeError("fill must be a SimulatedFill")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id is required")
        if ":" in run_id or any(char.isspace() for char in run_id):
            raise ValueError("run_id must not contain whitespace or ':'")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= 0:
            raise ValueError("sequence must be a positive integer")

        base_asset, quote_asset = _parse_symbol(fill.symbol)
        if quote_asset != state.cash_currency:
            raise ValueError(
                f"fill quote currency {quote_asset} does not match "
                f"ledger cash currency {state.cash_currency}"
            )

        quantity = _require_decimal(fill.quantity, "fill.quantity")
        fill_price = _require_decimal(fill.fill_price, "fill.fill_price")
        market_price = _require_decimal(fill.market_price, "fill.market_price")
        fee = _require_decimal(fill.fee, "fill.fee")

        if quantity <= ZERO:
            raise ValueError("fill.quantity must be greater than zero")
        if fill_price <= ZERO:
            raise ValueError("fill.fill_price must be greater than zero")
        if market_price <= ZERO:
            raise ValueError("fill.market_price must be greater than zero")
        if fee < ZERO:
            raise ValueError("fill.fee cannot be negative")
        if fill.side not in (Side.BUY, Side.SELL):
            raise ValueError("fill.side must be BUY or SELL")

        executed_notional = quantity * fill_price
        current_position = state.position_quantities.get(base_asset, ZERO)

        if fill.side is Side.BUY:
            total_cost = executed_notional + fee
            if total_cost > state.cash_balance:
                raise ValueError("insufficient virtual cash")

            cash_delta = -total_cost
            position_delta = quantity
        else:
            if quantity > current_position:
                raise ValueError("insufficient simulated position")

            cash_delta = executed_notional - fee
            position_delta = -quantity

        resulting_cash_balance = state.cash_balance + cash_delta
        resulting_position_quantity = current_position + position_delta

        updated_positions = dict(state.position_quantities)
        if resulting_position_quantity == ZERO:
            updated_positions.pop(base_asset, None)
        else:
            updated_positions[base_asset] = resulting_position_quantity

        new_state = SimulationLedgerState(
            cash_balance=resulting_cash_balance,
            cash_currency=state.cash_currency,
            position_quantities=updated_positions,
        )
        normalized_run_id = run_id.strip()
        entry = SimulationLedgerEntry(
            entry_id=f"{normalized_run_id}:ledger:{sequence}",
            run_id=normalized_run_id,
            sequence=sequence,
            fill_id=fill.fill_id,
            symbol=fill.symbol,
            side=fill.side,
            quantity=quantity,
            executed_notional=executed_notional,
            fee=fee,
            cash_delta=cash_delta,
            position_delta=position_delta,
            resulting_cash_balance=resulting_cash_balance,
            resulting_position_quantity=resulting_position_quantity,
        )
        return new_state, entry
