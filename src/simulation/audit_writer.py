"""Deterministic, isolated audit logging for virtual simulation runs."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any


class SimulationAuditPathError(ValueError):
    """Raised when an audit write would escape data/simulation/."""


class SimulationAuditValidationError(ValueError):
    """Raised when an audit event is invalid."""


_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "api_secret",
    "secret",
    "password",
    "token",
    "private_key",
    "credential",
)


@dataclass(frozen=True, slots=True)
class SimulationAuditEvent:
    """One validated candidate record for a simulation audit trail."""

    run_id: str
    sequence: int
    event_type: str
    occurred_at: datetime
    payload: Mapping[str, Any]


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _sanitize(value: Any) -> Any:
    """Convert values to deterministic JSON-safe, non-sensitive structures."""
    if is_dataclass(value) and not isinstance(value, type):
        return _sanitize(
            {
                field.name: getattr(value, field.name)
                for field in fields(value)
            }
        )

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _is_sensitive_key(str(key)) else _sanitize(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]

    if isinstance(value, set):
        return [_sanitize(item) for item in sorted(value, key=repr)]

    if value is None or isinstance(value, (str, int, bool)):
        return value

    raise SimulationAuditValidationError(
        f"Unsupported audit payload value type: {type(value).__name__}"
    )


class SimulationAuditWriter:
    """Append-only JSONL writer restricted to data/simulation/."""

    def __init__(self, root_directory: Path | str = Path("data/simulation")) -> None:
        configured_root = Path(root_directory)

        if configured_root.is_absolute():
            raise SimulationAuditPathError(
                "Simulation audit root must be a relative path under data/simulation/."
            )

        expected_root = (Path.cwd() / "data" / "simulation").resolve()
        resolved_root = (Path.cwd() / configured_root).resolve()

        if resolved_root != expected_root:
            raise SimulationAuditPathError(
                "Simulation audit output is restricted to data/simulation/."
            )

        self._root_directory = resolved_root

    @property
    def root_directory(self) -> Path:
        return self._root_directory

    def append_event(
        self,
        *,
        run_id: str,
        sequence: int,
        event_type: str,
        occurred_at: datetime,
        payload: Mapping[str, Any],
    ) -> Path:
        """Write one deterministic structured audit record immediately."""
        output_path, serialized = self._serialize_event(
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            occurred_at=occurred_at,
            payload=payload,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(serialized)
            audit_file.write("\n")
            audit_file.flush()
            os.fsync(audit_file.fileno())

        return output_path

    def append_events_atomically(
        self,
        events: Sequence[SimulationAuditEvent],
    ) -> Path | None:
        """
        Append a single run's records with one atomic file replacement.

        All records are validated and serialized before the existing audit
        file is changed. An exception before replacement leaves the prior
        file unchanged.
        """
        if not events:
            return None

        serialized_events: list[str] = []
        output_path: Path | None = None

        for event in events:
            if not isinstance(event, SimulationAuditEvent):
                raise TypeError("events must contain SimulationAuditEvent values")

            candidate_path, serialized = self._serialize_event(
                run_id=event.run_id,
                sequence=event.sequence,
                event_type=event.event_type,
                occurred_at=event.occurred_at,
                payload=event.payload,
            )

            if output_path is None:
                output_path = candidate_path
            elif candidate_path != output_path:
                raise SimulationAuditValidationError(
                    "Atomic audit batches must contain events for one run."
                )

            serialized_events.append(serialized)

        assert output_path is not None
        output_path.parent.mkdir(parents=True, exist_ok=True)

        existing_contents = (
            output_path.read_bytes() if output_path.exists() else b""
        )

        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)

        try:
            with os.fdopen(file_descriptor, "wb") as temporary_file:
                temporary_file.write(existing_contents)

                for serialized in serialized_events:
                    temporary_file.write(serialized.encode("utf-8"))
                    temporary_file.write(b"\n")

                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, output_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

        return output_path

    def validate_event(
        self,
        *,
        run_id: str,
        sequence: int,
        event_type: str,
        occurred_at: datetime,
        payload: Mapping[str, Any],
    ) -> None:
        """Validate an event without writing it to disk."""
        self._serialize_event(
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            occurred_at=occurred_at,
            payload=payload,
        )

    def path_for_run(self, run_id: str) -> Path:
        """Return the validated audit-file path for a run without writing."""
        return self._path_for_run(run_id)

    def _serialize_event(
        self,
        *,
        run_id: str,
        sequence: int,
        event_type: str,
        occurred_at: datetime,
        payload: Mapping[str, Any],
    ) -> tuple[Path, str]:
        if not isinstance(run_id, str) or not run_id.strip():
            raise SimulationAuditValidationError("run_id must be non-empty.")

        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise SimulationAuditValidationError("sequence must be at least 1.")

        if not isinstance(event_type, str) or not event_type.strip():
            raise SimulationAuditValidationError("event_type must be non-empty.")

        if not isinstance(occurred_at, datetime) or occurred_at.tzinfo is None:
            raise SimulationAuditValidationError(
                "occurred_at must be timezone-aware for deterministic audit records."
            )

        if occurred_at.utcoffset() is None:
            raise SimulationAuditValidationError(
                "occurred_at must be timezone-aware for deterministic audit records."
            )

        if not isinstance(payload, Mapping):
            raise SimulationAuditValidationError("payload must be a mapping.")

        normalized_run_id = run_id.strip()
        output_path = self._path_for_run(normalized_run_id)

        record = {
            "event_id": f"{normalized_run_id}:audit:{sequence:06d}",
            "event_type": event_type,
            "occurred_at": occurred_at.isoformat(),
            "payload": _sanitize(payload),
            "run_id": normalized_run_id,
            "sequence": sequence,
        }

        serialized = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

        return output_path, serialized

    def _path_for_run(self, run_id: str) -> Path:
        filename = f"{run_id}.jsonl"

        if (
            Path(run_id).name != run_id
            or "/" in run_id
            or "\\" in run_id
            or run_id in {".", ".."}
        ):
            raise SimulationAuditPathError(
                "run_id must be a filename-safe identifier without path separators."
            )

        candidate = (self._root_directory / filename).resolve()

        if candidate.parent != self._root_directory:
            raise SimulationAuditPathError(
                "Simulation audit output must remain directly inside data/simulation/."
            )

        return candidate
