"""Offline bridge from strategy decisions to the simulated allocation ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum

from src.portfolio.allocation_ledger import (
    AllocationLedger,
    LedgerError,
    OrderIntent,
)
from src.strategies.ema_regime_strategy import DecisionAction, StrategyDecision


ZERO = Decimal("0")
ONE = Decimal("1")


class AdapterOutcome(str, Enum):
    """Result category for applying a strategy decision."""

    EXECUTED = "EXECUTED"
    NO_ACTION = "NO_ACTION"
    DUPLICATE = "DUPLICATE"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class AdapterResult:
    """The outcome of applying one strategy decision."""

    outcome: AdapterOutcome
    decision_key: str | None
    message: str
    intent: OrderIntent | None = None


@dataclass(frozen=True)
class AdapterEvent:
    """Immutable adapter audit event."""

    sequence: int
    decision_key: str | None
    action: DecisionAction
    outcome: AdapterOutcome
    message: str
    intent_id: str | None


@dataclass
class StrategyLedgerAdapter:
    """Applies strategy decisions to one offline simulated allocation ledger.

    Idempotency is held in memory for this offline adapter. A live execution
    service needs durable storage and transactional reconciliation before it
    can safely support restarts or exchange interaction.
    """

    ledger: AllocationLedger
    strategy_id: str
    _processed_keys: set[str] = field(default_factory=set, init=False)
    _events: list[AdapterEvent] = field(default_factory=list, init=False)
    _sequence: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        normalized_strategy_id = self.strategy_id.strip()

        if not normalized_strategy_id:
            raise ValueError("strategy_id cannot be empty.")

        self.strategy_id = normalized_strategy_id

    @property
    def events(self) -> tuple[AdapterEvent, ...]:
        """Return an immutable view of adapter audit events."""

        return tuple(self._events)

    @property
    def processed_keys(self) -> frozenset[str]:
        """Return an immutable view of successfully processed decision keys."""

        return frozenset(self._processed_keys)

    def apply(self, decision: StrategyDecision) -> AdapterResult:
        """Apply one strategy decision without external side effects."""

        decision_key = self._decision_key(decision)

        if decision_key is None:
            return self._reject(
                decision=decision,
                decision_key=None,
                message=(
                    "Executable decisions require a non-empty candle timestamp "
                    "and a finite positive close price."
                ),
            )

        if decision_key in self._processed_keys:
            result = AdapterResult(
                outcome=AdapterOutcome.DUPLICATE,
                decision_key=decision_key,
                message="Decision was already processed.",
            )
            self._record(decision, result)
            return result

        if decision.action in (DecisionAction.HOLD, DecisionAction.BLOCKED):
            return self._complete_no_action(
                decision=decision,
                decision_key=decision_key,
                message=f"{decision.action.value} decisions do not create orders.",
            )

        try:
            price = self._decision_price(decision)

            if decision.action is DecisionAction.ENTER:
                return self._apply_enter(
                    decision=decision,
                    decision_key=decision_key,
                    price=price,
                )

            if decision.action is DecisionAction.EXIT:
                return self._apply_exit(
                    decision=decision,
                    decision_key=decision_key,
                    price=price,
                )

            return self._reject(
                decision=decision,
                decision_key=decision_key,
                message=f"Unsupported decision action: {decision.action!r}",
            )
        except (LedgerError, InvalidOperation, ValueError) as error:
            return self._reject(
                decision=decision,
                decision_key=decision_key,
                message=f"Ledger transition rejected: {error}",
            )

    def _apply_enter(
        self,
        *,
        decision: StrategyDecision,
        decision_key: str,
        price: Decimal,
    ) -> AdapterResult:
        if self.ledger.position.is_open:
            return self._complete_no_action(
                decision=decision,
                decision_key=decision_key,
                message="ENTER ignored because a simulated position is already open.",
            )

        quantity = self._buy_quantity(price)

        if quantity <= ZERO:
            return self._reject(
                decision=decision,
                decision_key=decision_key,
                message="ENTER rejected because the affordable quantity is zero.",
            )

        intent = self.ledger.create_buy_intent(
            quantity=quantity,
            reference_price=price,
        )
        filled_intent = self.ledger.fill_intent(
            intent.intent_id,
            fill_price=price,
        )

        return self._complete_execution(
            decision=decision,
            decision_key=decision_key,
            intent=filled_intent,
        )

    def _apply_exit(
        self,
        *,
        decision: StrategyDecision,
        decision_key: str,
        price: Decimal,
    ) -> AdapterResult:
        if not self.ledger.position.is_open:
            return self._complete_no_action(
                decision=decision,
                decision_key=decision_key,
                message="EXIT ignored because no simulated position is open.",
            )

        filled_intents: list[OrderIntent] = []

        while self.ledger.position.is_open:
            quantity_cap = self.ledger.max_order_notional / price
            quantity = min(self.ledger.position.quantity, quantity_cap)

            if quantity <= ZERO:
                raise ValueError(
                    "EXIT cannot proceed because the calculated sell quantity "
                    "is zero."
                )

            intent = self.ledger.create_sell_intent(
                quantity=quantity,
                reference_price=price,
            )
            filled_intent = self.ledger.fill_intent(
                intent.intent_id,
                fill_price=price,
            )
            filled_intents.append(filled_intent)

        self._processed_keys.add(decision_key)

        result = AdapterResult(
            outcome=AdapterOutcome.EXECUTED,
            decision_key=decision_key,
            message=(
                f"EXIT filled successfully using {len(filled_intents)} "
                "simulated order(s)."
            ),
            intent=filled_intents[-1],
        )
        self._record(decision, result)
        return result

    def _buy_quantity(self, price: Decimal) -> Decimal:
        """Return the largest ledger-valid quantity affordable with fees."""

        affordable_notional = self.ledger.available_cash / (
            ONE + self.ledger.fee_rate
        )
        permitted_notional = min(
            affordable_notional,
            self.ledger.max_order_notional,
        )

        return permitted_notional / price

    def _decision_key(self, decision: StrategyDecision) -> str | None:
        """Return a stable idempotency key, or None for invalid decisions."""

        if not decision.candle_timestamp or not decision.candle_timestamp.strip():
            return None

        try:
            price = self._decision_price(decision)
        except (InvalidOperation, TypeError, ValueError):
            return None

        return "|".join(
            (
                self.strategy_id,
                decision.candle_timestamp.strip(),
                decision.action.value,
                str(price),
            )
        )

    @staticmethod
    def _decision_price(decision: StrategyDecision) -> Decimal:
        """Validate and normalize the decision close price."""

        if decision.close is None:
            raise ValueError("Decision close price is required.")

        price = Decimal(str(decision.close))

        if not price.is_finite() or price <= ZERO:
            raise ValueError(
                "Decision close price must be finite and greater than zero."
            )

        return price

    def _complete_execution(
        self,
        *,
        decision: StrategyDecision,
        decision_key: str,
        intent: OrderIntent,
    ) -> AdapterResult:
        self._processed_keys.add(decision_key)

        result = AdapterResult(
            outcome=AdapterOutcome.EXECUTED,
            decision_key=decision_key,
            message=f"{decision.action.value} filled successfully.",
            intent=intent,
        )
        self._record(decision, result)
        return result

    def _complete_no_action(
        self,
        *,
        decision: StrategyDecision,
        decision_key: str,
        message: str,
    ) -> AdapterResult:
        self._processed_keys.add(decision_key)

        result = AdapterResult(
            outcome=AdapterOutcome.NO_ACTION,
            decision_key=decision_key,
            message=message,
        )
        self._record(decision, result)
        return result

    def _reject(
        self,
        *,
        decision: StrategyDecision,
        decision_key: str | None,
        message: str,
    ) -> AdapterResult:
        result = AdapterResult(
            outcome=AdapterOutcome.REJECTED,
            decision_key=decision_key,
            message=message,
        )
        self._record(decision, result)
        return result

    def _record(
        self,
        decision: StrategyDecision,
        result: AdapterResult,
    ) -> None:
        """Append one immutable adapter audit event."""

        self._sequence += 1

        self._events.append(
            AdapterEvent(
                sequence=self._sequence,
                decision_key=result.decision_key,
                action=decision.action,
                outcome=result.outcome,
                message=result.message,
                intent_id=(
                    result.intent.intent_id
                    if result.intent is not None
                    else None
                ),
            )
        )
