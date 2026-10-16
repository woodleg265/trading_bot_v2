"""Pure, offline simulated allocation ledger.

This module intentionally has no exchange, network, credential, filesystem,
or live-order dependencies. It tracks allocated simulated capital only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Optional
from uuid import uuid4


ZERO = Decimal("0")
QUANTITY_PRECISION = Decimal("0.00000001")


class LedgerError(ValueError):
    """Raised when a requested ledger transition is invalid."""


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class IntentStatus(str, Enum):
    CREATED = "CREATED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class OrderIntent:
    """An immutable request for a simulated order."""

    intent_id: str
    side: OrderSide
    quantity: Decimal
    reference_price: Decimal
    status: IntentStatus
    created_sequence: int


@dataclass(frozen=True)
class LedgerEvent:
    """An immutable, append-only audit event."""

    sequence: int
    event_type: str
    intent_id: Optional[str]
    message: str


@dataclass
class SimulatedPosition:
    """Current long-only simulated position."""

    quantity: Decimal = ZERO
    average_entry_price: Decimal = ZERO

    @property
    def is_open(self) -> bool:
        return self.quantity > ZERO


@dataclass
class AllocationLedger:
    """Tracks a single allocated, long-only simulated trading account."""

    allocated_cash: Decimal
    fee_rate: Decimal
    max_order_notional: Decimal
    quote_currency: str = "USD"
    base_currency: str = "BTC"
    available_cash: Decimal = field(init=False)
    position: SimulatedPosition = field(default_factory=SimulatedPosition, init=False)
    realized_pnl: Decimal = field(default=ZERO, init=False)
    total_fees: Decimal = field(default=ZERO, init=False)
    trade_count: int = field(default=0, init=False)
    _intents: dict[str, OrderIntent] = field(default_factory=dict, init=False)
    _events: list[LedgerEvent] = field(default_factory=list, init=False)
    _sequence: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.allocated_cash = self._as_positive_decimal(
            self.allocated_cash,
            "allocated_cash",
        )
        self.fee_rate = self._as_non_negative_decimal(self.fee_rate, "fee_rate")
        self.max_order_notional = self._as_positive_decimal(
            self.max_order_notional,
            "max_order_notional",
        )
        self.available_cash = self.allocated_cash
        self._record(
            "LEDGER_CREATED",
            None,
            f"Allocated {self.allocated_cash} {self.quote_currency}",
        )

    @property
    def intents(self) -> tuple[OrderIntent, ...]:
        """Return all intents in creation order."""

        return tuple(
            sorted(
                self._intents.values(),
                key=lambda intent: intent.created_sequence,
            )
        )

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        """Return an immutable view of audit events."""

        return tuple(self._events)

    def create_buy_intent(
        self,
        *,
        quantity: Decimal,
        reference_price: Decimal,
    ) -> OrderIntent:
        """Create a buy intent without changing cash or position."""

        if self.position.is_open:
            raise LedgerError("Cannot create a buy intent while a position is open.")

        return self._create_intent(
            side=OrderSide.BUY,
            quantity=quantity,
            reference_price=reference_price,
        )

    def create_sell_intent(
        self,
        *,
        quantity: Decimal,
        reference_price: Decimal,
    ) -> OrderIntent:
        """Create a sell intent for part or all of the current position."""

        if not self.position.is_open:
            raise LedgerError("Cannot create a sell intent without an open position.")

        validated_quantity = self._normalized_quantity(quantity)

        if validated_quantity > self.position.quantity:
            raise LedgerError("Sell quantity exceeds the simulated position quantity.")

        return self._create_intent(
            side=OrderSide.SELL,
            quantity=validated_quantity,
            reference_price=reference_price,
        )

    def fill_intent(
        self,
        intent_id: str,
        *,
        fill_price: Decimal,
    ) -> OrderIntent:
        """Fill an existing intent at an explicitly supplied simulated price."""

        intent = self._get_created_intent(intent_id)
        validated_price = self._as_positive_decimal(fill_price, "fill_price")
        notional = intent.quantity * validated_price
        fee = notional * self.fee_rate

        if intent.side == OrderSide.BUY:
            self._fill_buy(intent, validated_price, notional, fee)
        else:
            self._fill_sell(intent, validated_price, notional, fee)

        filled_intent = OrderIntent(
            intent_id=intent.intent_id,
            side=intent.side,
            quantity=intent.quantity,
            reference_price=intent.reference_price,
            status=IntentStatus.FILLED,
            created_sequence=intent.created_sequence,
        )
        self._intents[intent_id] = filled_intent
        self.trade_count += 1
        self.total_fees += fee

        self._record(
            "INTENT_FILLED",
            intent_id,
            (
                f"{intent.side.value} {intent.quantity} {self.base_currency} "
                f"at {validated_price}; fee={fee}"
            ),
        )
        return filled_intent

    def cancel_intent(self, intent_id: str) -> OrderIntent:
        """Cancel an unfilled intent without modifying cash or position."""

        intent = self._get_created_intent(intent_id)
        cancelled_intent = OrderIntent(
            intent_id=intent.intent_id,
            side=intent.side,
            quantity=intent.quantity,
            reference_price=intent.reference_price,
            status=IntentStatus.CANCELLED,
            created_sequence=intent.created_sequence,
        )
        self._intents[intent_id] = cancelled_intent
        self._record("INTENT_CANCELLED", intent_id, "Intent cancelled.")
        return cancelled_intent

    def equity(self, mark_price: Decimal) -> Decimal:
        """Return simulated cash plus marked-to-market position value."""

        validated_price = self._as_positive_decimal(mark_price, "mark_price")
        return self.available_cash + (self.position.quantity * validated_price)

    def _create_intent(
        self,
        *,
        side: OrderSide,
        quantity: Decimal,
        reference_price: Decimal,
    ) -> OrderIntent:
        validated_quantity = self._normalized_quantity(quantity)
        validated_price = self._as_positive_decimal(
            reference_price,
            "reference_price",
        )
        notional = validated_quantity * validated_price

        if notional > self.max_order_notional:
            raise LedgerError("Order notional exceeds max_order_notional.")

        self._sequence += 1
        intent = OrderIntent(
            intent_id=str(uuid4()),
            side=side,
            quantity=validated_quantity,
            reference_price=validated_price,
            status=IntentStatus.CREATED,
            created_sequence=self._sequence,
        )
        self._intents[intent.intent_id] = intent
        self._record(
            "INTENT_CREATED",
            intent.intent_id,
            (
                f"{side.value} {validated_quantity} {self.base_currency} "
                f"at reference price {validated_price}"
            ),
        )
        return intent

    def _fill_buy(
        self,
        intent: OrderIntent,
        fill_price: Decimal,
        notional: Decimal,
        fee: Decimal,
    ) -> None:
        total_cost = notional + fee

        if total_cost > self.available_cash:
            raise LedgerError("Insufficient allocated simulated cash for buy fill.")

        if self.position.is_open:
            raise LedgerError("Cannot fill a buy while a position is open.")

        self.available_cash -= total_cost
        self.position = SimulatedPosition(
            quantity=intent.quantity,
            average_entry_price=(total_cost / intent.quantity),
        )

    def _fill_sell(
        self,
        intent: OrderIntent,
        fill_price: Decimal,
        notional: Decimal,
        fee: Decimal,
    ) -> None:
        if intent.quantity > self.position.quantity:
            raise LedgerError("Sell fill quantity exceeds the simulated position.")

        proceeds_after_fee = notional - fee
        cost_basis = intent.quantity * self.position.average_entry_price
        self.realized_pnl += proceeds_after_fee - cost_basis
        self.available_cash += proceeds_after_fee

        remaining_quantity = self.position.quantity - intent.quantity

        if remaining_quantity == ZERO:
            self.position = SimulatedPosition()
        else:
            self.position = SimulatedPosition(
                quantity=remaining_quantity,
                average_entry_price=self.position.average_entry_price,
            )

    def _get_created_intent(self, intent_id: str) -> OrderIntent:
        try:
            intent = self._intents[intent_id]
        except KeyError as error:
            raise LedgerError(f"Unknown intent: {intent_id}") from error

        if intent.status is not IntentStatus.CREATED:
            raise LedgerError(
                f"Intent {intent_id} cannot be changed from {intent.status.value}."
            )

        return intent

    def _record(
        self,
        event_type: str,
        intent_id: Optional[str],
        message: str,
    ) -> None:
        self._sequence += 1
        self._events.append(
            LedgerEvent(
                sequence=self._sequence,
                event_type=event_type,
                intent_id=intent_id,
                message=message,
            )
        )

    @staticmethod
    def _as_positive_decimal(value: Decimal, name: str) -> Decimal:
        decimal_value = Decimal(value)

        if decimal_value <= ZERO:
            raise LedgerError(f"{name} must be greater than zero.")

        return decimal_value

    @staticmethod
    def _as_non_negative_decimal(value: Decimal, name: str) -> Decimal:
        decimal_value = Decimal(value)

        if decimal_value < ZERO:
            raise LedgerError(f"{name} cannot be negative.")

        return decimal_value

    @staticmethod
    def _normalized_quantity(value: Decimal) -> Decimal:
        quantity = AllocationLedger._as_positive_decimal(value, "quantity")
        normalized = quantity.quantize(
            QUANTITY_PRECISION,
            rounding=ROUND_DOWN,
        )

        if normalized <= ZERO:
            raise LedgerError("quantity is too small after precision normalization.")

        return normalized
