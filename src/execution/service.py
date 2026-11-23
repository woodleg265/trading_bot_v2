from __future__ import annotations

from dataclasses import dataclass, replace

from src.execution.idempotency import derive_idempotency_key
from src.execution.models import (
    ExchangeOrderRecord,
    ExecutionReport,
    ExecutionStatus,
    IdempotencyClaim,
    IdempotencyStatus,
    OrderSubmissionRequest,
    StrategyIntentEvent,
)
from src.execution.ports import ExecutionGateway, ExecutionStore


@dataclass(frozen=True, slots=True)
class ExecutionSubmissionResult:
    """Outcome of a submission attempt without hiding reconciliation needs."""

    idempotency_key: str
    exchange_order: ExchangeOrderRecord | None
    report: ExecutionReport
    submitted_now: bool


class LiveExecutionService:
    """Idempotent, exchange-agnostic order-submission orchestrator.

    This service intentionally does not retry gateway calls. An ambiguous
    outcome is recorded as UNKNOWN and must be reconciled using exchange state
    before any further action is taken.
    """

    def __init__(self, *, gateway: ExecutionGateway, store: ExecutionStore) -> None:
        self._gateway = gateway
        self._store = store

    def submit_intent(
        self,
        intent: StrategyIntentEvent,
    ) -> ExecutionSubmissionResult:
        key = derive_idempotency_key(intent)
        now = self._store.now()
        pending_claim = IdempotencyClaim(
            idempotency_key=key,
            intent_id=intent.intent_id,
            status=IdempotencyStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        claim, created = self._store.create_idempotency_claim(pending_claim)

        if claim.intent_id != intent.intent_id:
            raise ValueError("idempotency key is already associated with another intent")

        if not created:
            return self._existing_claim_result(intent=intent, claim=claim)

        request = OrderSubmissionRequest(
            idempotency_key=key,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
        )

        try:
            exchange_order = self._gateway.submit_order(request)
        except Exception as error:
            return self._record_unknown_submission(
                intent=intent,
                claim=claim,
                error=error,
            )

        if exchange_order.idempotency_key != key:
            raise ValueError("gateway returned an exchange record for another idempotency key")
        if exchange_order.intent_id != intent.intent_id:
            raise ValueError("gateway returned an exchange record for another intent")

        self._store.save_exchange_order(exchange_order)
        accepted_claim = replace(
            claim,
            status=IdempotencyStatus.ACCEPTED,
            updated_at=self._store.now(),
        )
        self._store.update_idempotency_claim(accepted_claim)
        report = ExecutionReport(
            idempotency_key=key,
            intent_id=intent.intent_id,
            status=ExecutionStatus.ACCEPTED,
            exchange_order_id=exchange_order.exchange_order_id,
            reported_at=self._store.now(),
        )
        self._store.save_execution_report(report)
        return ExecutionSubmissionResult(
            idempotency_key=key,
            exchange_order=exchange_order,
            report=report,
            submitted_now=True,
        )

    def _existing_claim_result(
        self,
        *,
        intent: StrategyIntentEvent,
        claim: IdempotencyClaim,
    ) -> ExecutionSubmissionResult:
        exchange_order = self._store.get_exchange_order(claim.idempotency_key)
        status = (
            ExecutionStatus.ACCEPTED
            if exchange_order is not None
            else ExecutionStatus.UNKNOWN
        )
        report = ExecutionReport(
            idempotency_key=claim.idempotency_key,
            intent_id=intent.intent_id,
            status=status,
            exchange_order_id=(
                exchange_order.exchange_order_id if exchange_order is not None else None
            ),
            reported_at=self._store.now(),
            message=(
                "Existing idempotency claim found; reconciliation is required."
                if exchange_order is None
                else "Existing exchange acknowledgement found; no resubmission performed."
            ),
        )
        self._store.save_execution_report(report)
        return ExecutionSubmissionResult(
            idempotency_key=claim.idempotency_key,
            exchange_order=exchange_order,
            report=report,
            submitted_now=False,
        )

    def _record_unknown_submission(
        self,
        *,
        intent: StrategyIntentEvent,
        claim: IdempotencyClaim,
        error: Exception,
    ) -> ExecutionSubmissionResult:
        unknown_claim = replace(
            claim,
            status=IdempotencyStatus.UNKNOWN,
            updated_at=self._store.now(),
        )
        self._store.update_idempotency_claim(unknown_claim)
        report = ExecutionReport(
            idempotency_key=claim.idempotency_key,
            intent_id=intent.intent_id,
            status=ExecutionStatus.UNKNOWN,
            reported_at=self._store.now(),
            message=f"Submission outcome is ambiguous: {type(error).__name__}",
        )
        self._store.save_execution_report(report)
        return ExecutionSubmissionResult(
            idempotency_key=claim.idempotency_key,
            exchange_order=None,
            report=report,
            submitted_now=True,
        )
