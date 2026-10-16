"""Deterministic orchestration for isolated virtual trading simulations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from .audit_writer import SimulationAuditWriter
from .fill_model import SimulationFillModel
from .ledger import SimulationLedger, SimulationLedgerEntry, SimulationLedgerState
from .models import OrderIntent, RiskDecision, SimulatedFill, Side
from .risk_engine import SimulationRiskContext, SimulationRiskEngine


ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class SimulationRunResult:
    """The immutable result of processing one virtual order intent."""

    state: SimulationLedgerState
    risk_context: SimulationRiskContext
    risk_decision: RiskDecision
    fill: Optional[SimulatedFill]
    ledger_entry: Optional[SimulationLedgerEntry]


class SimulationRunner:
    """Orchestrates risk, fill, ledger, and audit operations for one intent."""

    def __init__(
        self,
        *,
        risk_engine: SimulationRiskEngine,
        fill_model: SimulationFillModel,
        ledger: SimulationLedger,
        audit_writer: SimulationAuditWriter,
    ) -> None:
        self._risk_engine = risk_engine
        self._fill_model = fill_model
        self._ledger = ledger
        self._audit_writer = audit_writer

    def process_intent(
        self,
        intent: OrderIntent,
        *,
        state: SimulationLedgerState,
        risk_context: SimulationRiskContext,
        run_id: str,
        sequence: int,
        evaluated_at: datetime,
    ) -> SimulationRunResult:
        """Process one intent through risk, fill, ledger, and audit stages."""
        self._validate_inputs(
            intent=intent,
            state=state,
            risk_context=risk_context,
            run_id=run_id,
            sequence=sequence,
            evaluated_at=evaluated_at,
        )

        normalized_run_id = run_id.strip()

        decision = self._risk_engine.evaluate(
            intent,
            context=risk_context,
            evaluated_at=evaluated_at,
        )

        self._audit_writer.append_event(
            run_id=normalized_run_id,
            sequence=sequence,
            event_type="RISK_DECISION",
            occurred_at=evaluated_at,
            payload={
                "intent": intent,
                "risk_decision": decision,
            },
        )

        processed_intent_ids = frozenset(
            set(risk_context.processed_intent_ids) | {intent.intent_id}
        )

        if not decision.approved:
            return SimulationRunResult(
                state=state,
                risk_context=SimulationRiskContext(
                    available_cash=state.cash_balance,
                    position_quantities=dict(risk_context.position_quantities),
                    current_total_exposure=risk_context.current_total_exposure,
                    processed_intent_ids=processed_intent_ids,
                ),
                risk_decision=decision,
                fill=None,
                ledger_entry=None,
            )

        fill_sequence = sequence + 1
        ledger_sequence = sequence + 2

        fill = self._fill_model.create_fill(
            intent,
            risk_decision=decision,
            run_id=normalized_run_id,
            sequence=fill_sequence,
        )

        new_state, ledger_entry = self._ledger.apply_fill(
            state,
            fill,
            run_id=normalized_run_id,
            sequence=ledger_sequence,
        )

        self._audit_writer.append_event(
            run_id=normalized_run_id,
            sequence=fill_sequence,
            event_type="SIMULATED_FILL",
            occurred_at=evaluated_at,
            payload={
                "fill": fill,
                "intent_id": intent.intent_id,
                "risk_code": decision.code,
            },
        )

        self._audit_writer.append_event(
            run_id=normalized_run_id,
            sequence=ledger_sequence,
            event_type="LEDGER_TRANSITION",
            occurred_at=evaluated_at,
            payload={
                "ledger_entry": ledger_entry,
                "resulting_state": new_state,
            },
        )

        updated_positions = dict(risk_context.position_quantities)
        updated_positions[intent.symbol] = ledger_entry.resulting_position_quantity

        if ledger_entry.resulting_position_quantity == ZERO:
            updated_positions.pop(intent.symbol, None)

        if intent.side is Side.BUY:
            updated_exposure = (
                risk_context.current_total_exposure + decision.evaluated_notional
            )
        else:
            updated_exposure = max(
                ZERO,
                risk_context.current_total_exposure - decision.evaluated_notional,
            )

        return SimulationRunResult(
            state=new_state,
            risk_context=SimulationRiskContext(
                available_cash=new_state.cash_balance,
                position_quantities=updated_positions,
                current_total_exposure=updated_exposure,
                processed_intent_ids=processed_intent_ids,
            ),
            risk_decision=decision,
            fill=fill,
            ledger_entry=ledger_entry,
        )

    @staticmethod
    def _validate_inputs(
        *,
        intent: OrderIntent,
        state: SimulationLedgerState,
        risk_context: SimulationRiskContext,
        run_id: str,
        sequence: int,
        evaluated_at: datetime,
    ) -> None:
        if not isinstance(intent, OrderIntent):
            raise TypeError("intent must be an OrderIntent")
        if not isinstance(state, SimulationLedgerState):
            raise TypeError("state must be a SimulationLedgerState")
        if not isinstance(risk_context, SimulationRiskContext):
            raise TypeError("risk_context must be a SimulationRiskContext")

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id is required")
        if ":" in run_id or any(character.isspace() for character in run_id):
            raise ValueError("run_id must not contain whitespace or ':'")

        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= 0:
            raise ValueError("sequence must be a positive integer")

        if not isinstance(evaluated_at, datetime):
            raise TypeError("evaluated_at must be a datetime")
        if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")
