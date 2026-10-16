"""Isolated bridge from strategy decisions to the virtual simulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from hashlib import sha256

from src.simulation.ledger import SimulationLedgerState
from src.simulation.models import OrderIntent, Side
from src.simulation.risk_engine import SimulationRiskContext
from src.simulation.runner import SimulationRunResult, SimulationRunner
from src.strategies.ema_regime_strategy import DecisionAction, StrategyDecision


ZERO = Decimal("0")


class SimulationAdapterOutcome(str, Enum):
    """Result category for one strategy-decision application."""

    EXECUTED = "EXECUTED"
    NO_ACTION = "NO_ACTION"
    DUPLICATE = "DUPLICATE"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class SimulationStrategyAdapterConfig:
    """Explicit strategy-to-simulation sizing configuration."""

    symbol: str
    entry_quantity: Decimal
    max_exit_quantity: Decimal

    def __post_init__(self) -> None:
        normalized_symbol = self.symbol.strip().upper()

        if not normalized_symbol:
            raise ValueError("symbol is required")
        if "/" not in normalized_symbol:
            raise ValueError("symbol must use BASE/QUOTE format")
        if self.entry_quantity <= ZERO:
            raise ValueError("entry_quantity must be greater than zero")
        if self.max_exit_quantity <= ZERO:
            raise ValueError("max_exit_quantity must be greater than zero")

        object.__setattr__(self, "symbol", normalized_symbol)


@dataclass(frozen=True, slots=True)
class SimulationStrategyAdapterResult:
    """The result of applying one strategy decision."""

    outcome: SimulationAdapterOutcome
    decision_key: str | None
    message: str
    run_results: tuple[SimulationRunResult, ...] = ()


@dataclass(frozen=True, slots=True)
class SimulationStrategyAdapterEvent:
    """In-memory adapter event; order lifecycle events remain in JSONL audit."""

    sequence: int
    decision_key: str | None
    action: DecisionAction
    outcome: SimulationAdapterOutcome
    message: str
    intent_ids: tuple[str, ...]


@dataclass
class SimulationStrategyAdapter:
    """Routes strategy decisions exclusively through ``SimulationRunner``.

    This component has no dependency on the legacy ``AllocationLedger`` and
    never imports broker, exchange, or production-execution code.
    """

    runner: SimulationRunner
    config: SimulationStrategyAdapterConfig
    state: SimulationLedgerState
    risk_context: SimulationRiskContext
    run_id: str
    _processed_decision_keys: set[str] = field(default_factory=set, init=False)
    _events: list[SimulationStrategyAdapterEvent] = field(
        default_factory=list,
        init=False,
    )
    _event_sequence: int = field(default=0, init=False)
    _runner_sequence: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        normalized_run_id = self.run_id.strip()

        if not normalized_run_id:
            raise ValueError("run_id is required")
        if ":" in normalized_run_id or any(char.isspace() for char in normalized_run_id):
            raise ValueError("run_id must not contain whitespace or ':'")

        self.run_id = normalized_run_id

    @property
    def events(self) -> tuple[SimulationStrategyAdapterEvent, ...]:
        """Return an immutable snapshot of adapter events."""

        return tuple(self._events)

    @property
    def processed_decision_keys(self) -> frozenset[str]:
        """Return decision keys already handled by this adapter session."""

        return frozenset(self._processed_decision_keys)

    def apply(self, decision: StrategyDecision) -> SimulationStrategyAdapterResult:
        """Apply a strategy decision through the virtual simulation pipeline."""

        decision_key = self._decision_key(decision)

        if decision_key is None:
            return self._complete(
                decision=decision,
                outcome=SimulationAdapterOutcome.REJECTED,
                decision_key=None,
                message=(
                    "Executable decisions require a timezone-aware ISO-8601 "
                    "candle timestamp and a finite positive close price."
                ),
            )

        if decision_key in self._processed_decision_keys:
            return self._complete(
                decision=decision,
                outcome=SimulationAdapterOutcome.DUPLICATE,
                decision_key=decision_key,
                message="Decision was already processed.",
            )

        if decision.action in (DecisionAction.HOLD, DecisionAction.BLOCKED):
            self._processed_decision_keys.add(decision_key)
            return self._complete(
                decision=decision,
                outcome=SimulationAdapterOutcome.NO_ACTION,
                decision_key=decision_key,
                message=f"{decision.action.value} decisions do not create virtual orders.",
            )

        try:
            price = self._decision_price(decision)
            evaluated_at = self._decision_timestamp(decision)

            if decision.action is DecisionAction.ENTER:
                return self._apply_enter(
                    decision=decision,
                    decision_key=decision_key,
                    price=price,
                    evaluated_at=evaluated_at,
                )

            if decision.action is DecisionAction.EXIT:
                return self._apply_exit(
                    decision=decision,
                    decision_key=decision_key,
                    price=price,
                    evaluated_at=evaluated_at,
                )
        except (InvalidOperation, TypeError, ValueError) as error:
            return self._complete(
                decision=decision,
                outcome=SimulationAdapterOutcome.REJECTED,
                decision_key=decision_key,
                message=f"Simulation adapter rejected decision: {error}",
            )

        return self._complete(
            decision=decision,
            outcome=SimulationAdapterOutcome.REJECTED,
            decision_key=decision_key,
            message=f"Unsupported strategy action: {decision.action!r}",
        )

    def _apply_enter(
        self,
        *,
        decision: StrategyDecision,
        decision_key: str,
        price: Decimal,
        evaluated_at: datetime,
    ) -> SimulationStrategyAdapterResult:
        if self._position_quantity() > ZERO:
            self._processed_decision_keys.add(decision_key)
            return self._complete(
                decision=decision,
                outcome=SimulationAdapterOutcome.NO_ACTION,
                decision_key=decision_key,
                message="ENTER ignored because a virtual position is already open.",
            )

        result = self._process_intent(
            decision_key=decision_key,
            side=Side.BUY,
            quantity=self.config.entry_quantity,
            price=price,
            evaluated_at=evaluated_at,
            ordinal=1,
        )
        self._processed_decision_keys.add(decision_key)

        return self._complete_from_runs(
            decision=decision,
            decision_key=decision_key,
            run_results=(result,),
        )

    def _apply_exit(
        self,
        *,
        decision: StrategyDecision,
        decision_key: str,
        price: Decimal,
        evaluated_at: datetime,
    ) -> SimulationStrategyAdapterResult:
        quantity_remaining = self._position_quantity()

        if quantity_remaining <= ZERO:
            self._processed_decision_keys.add(decision_key)
            return self._complete(
                decision=decision,
                outcome=SimulationAdapterOutcome.NO_ACTION,
                decision_key=decision_key,
                message="EXIT ignored because no virtual position is open.",
            )

        run_results: list[SimulationRunResult] = []
        ordinal = 1

        while quantity_remaining > ZERO:
            quantity = min(quantity_remaining, self.config.max_exit_quantity)
            result = self._process_intent(
                decision_key=decision_key,
                side=Side.SELL,
                quantity=quantity,
                price=price,
                evaluated_at=evaluated_at,
                ordinal=ordinal,
            )
            run_results.append(result)

            if not result.risk_decision.approved:
                break

            quantity_remaining = self._position_quantity()
            ordinal += 1

        self._processed_decision_keys.add(decision_key)

        return self._complete_from_runs(
            decision=decision,
            decision_key=decision_key,
            run_results=tuple(run_results),
        )

    def _process_intent(
        self,
        *,
        decision_key: str,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        evaluated_at: datetime,
        ordinal: int,
    ) -> SimulationRunResult:
        intent = OrderIntent(
            intent_id=self._intent_id(decision_key, side, ordinal),
            symbol=self.config.symbol,
            side=side,
            quantity=quantity,
            reference_price=price,
            market_data_timestamp=evaluated_at,
            created_at=evaluated_at,
        )

        result = self.runner.process_intent(
            intent,
            state=self.state,
            risk_context=self.risk_context,
            run_id=self.run_id,
            sequence=self._runner_sequence,
            evaluated_at=evaluated_at,
        )

        self.state = result.state
        self.risk_context = result.risk_context

        # Reserve all three stage sequence numbers whether risk approves or
        # rejects, preventing future event-ID reuse within this session.
        self._runner_sequence += 3
        return result

    def _complete_from_runs(
        self,
        *,
        decision: StrategyDecision,
        decision_key: str,
        run_results: tuple[SimulationRunResult, ...],
    ) -> SimulationStrategyAdapterResult:
        rejected = next(
            (result for result in run_results if not result.risk_decision.approved),
            None,
        )

        if rejected is not None:
            return self._complete(
                decision=decision,
                outcome=SimulationAdapterOutcome.REJECTED,
                decision_key=decision_key,
                message=(
                    "Virtual order rejected by simulation risk controls: "
                    f"{rejected.risk_decision.code}."
                ),
                run_results=run_results,
            )

        return self._complete(
            decision=decision,
            outcome=SimulationAdapterOutcome.EXECUTED,
            decision_key=decision_key,
            message=f"{decision.action.value} completed through virtual simulation.",
            run_results=run_results,
        )

    def _complete(
        self,
        *,
        decision: StrategyDecision,
        outcome: SimulationAdapterOutcome,
        decision_key: str | None,
        message: str,
        run_results: tuple[SimulationRunResult, ...] = (),
    ) -> SimulationStrategyAdapterResult:
        result = SimulationStrategyAdapterResult(
            outcome=outcome,
            decision_key=decision_key,
            message=message,
            run_results=run_results,
        )

        self._event_sequence += 1
        self._events.append(
            SimulationStrategyAdapterEvent(
                sequence=self._event_sequence,
                decision_key=decision_key,
                action=decision.action,
                outcome=outcome,
                message=message,
                intent_ids=tuple(
                    run.fill.intent_id
                    for run in run_results
                    if run.fill is not None
                ),
            )
        )
        return result

    def _position_quantity(self) -> Decimal:
        base_asset = self.config.symbol.split("/", maxsplit=1)[0]
        return self.state.position_quantities.get(base_asset, ZERO)

    def _decision_key(self, decision: StrategyDecision) -> str | None:
        try:
            price = self._decision_price(decision)
            timestamp = self._decision_timestamp(decision)
        except (InvalidOperation, TypeError, ValueError):
            return None

        return "|".join(
            (
                self.run_id,
                self.config.symbol,
                timestamp.isoformat(),
                decision.action.value,
                str(price),
            )
        )

    @staticmethod
    def _decision_price(decision: StrategyDecision) -> Decimal:
        if decision.close is None:
            raise ValueError("Decision close price is required.")

        price = Decimal(str(decision.close))

        if not price.is_finite() or price <= ZERO:
            raise ValueError("Decision close price must be finite and greater than zero.")

        return price

    @staticmethod
    def _decision_timestamp(decision: StrategyDecision) -> datetime:
        if not decision.candle_timestamp or not decision.candle_timestamp.strip():
            raise ValueError("Decision candle timestamp is required.")

        timestamp_text = decision.candle_timestamp.strip()
        if timestamp_text.endswith("Z"):
            timestamp_text = f"{timestamp_text[:-1]}+00:00"

        timestamp = datetime.fromisoformat(timestamp_text)

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("Decision candle timestamp must be timezone-aware.")

        return timestamp

    @staticmethod
    def _intent_id(decision_key: str, side: Side, ordinal: int) -> str:
        digest = sha256(
            f"{decision_key}|{side.value}|{ordinal}".encode("utf-8")
        ).hexdigest()
        return f"strategy-{digest}"
