"""Transaction-scoped staging for simulation JSONL audit records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .audit_writer import SimulationAuditEvent, SimulationAuditWriter


@dataclass
class StagedSimulationAuditWriter:
    """
    Buffer audit records until an explicit transaction commit.

    It intentionally has the same append_event() call shape used by
    SimulationRunner, allowing the runner to remain unchanged.
    """

    _writer: SimulationAuditWriter
    _pending_events: list[SimulationAuditEvent] = field(default_factory=list)
    _transaction_active: bool = False

    @property
    def root_directory(self) -> Path:
        """Expose the underlying writer's audit root."""
        return self._writer.root_directory

    @property
    def transaction_active(self) -> bool:
        """Whether records are currently being staged."""
        return self._transaction_active

    @property
    def pending_event_count(self) -> int:
        """Number of records waiting for commit."""
        return len(self._pending_events)

    def begin(self) -> None:
        """Start a new audit transaction."""
        if self._transaction_active:
            raise RuntimeError("An audit transaction is already active.")

        self._pending_events.clear()
        self._transaction_active = True

    def append_event(
        self,
        *,
        run_id: str,
        sequence: int,
        event_type: str,
        occurred_at: datetime,
        payload: Mapping[str, Any],
    ) -> Path:
        """
        Validate and stage one audit event.

        Events may only be accepted inside an explicit transaction. This
        prevents accidental direct persistence through a runner configured
        for transactional audit handling.
        """
        if not self._transaction_active:
            raise RuntimeError(
                "Audit events may only be appended inside an active transaction."
            )

        self._writer.validate_event(
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            occurred_at=occurred_at,
            payload=payload,
        )

        self._pending_events.append(
            SimulationAuditEvent(
                run_id=run_id,
                sequence=sequence,
                event_type=event_type,
                occurred_at=occurred_at,
                payload=payload,
            )
        )

        return self._writer.path_for_run(run_id.strip())

    def commit(self) -> Path | None:
        """
        Persist all staged records through one atomic replacement.

        The pending buffer is cleared only after a successful write. If the
        underlying write fails, rollback() can discard the still-pending
        records while the existing audit file remains unchanged.
        """
        self._require_active_transaction()

        output_path = self._writer.append_events_atomically(
            tuple(self._pending_events)
        )
        self._pending_events.clear()
        self._transaction_active = False
        return output_path

    def rollback(self) -> None:
        """Discard all staged records without performing audit-file I/O."""
        self._pending_events.clear()
        self._transaction_active = False

    def _require_active_transaction(self) -> None:
        if not self._transaction_active:
            raise RuntimeError("No audit transaction is active.")
