"""Deterministic, isolated audit logging for virtual simulation runs."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
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
        """Write one deterministic structured audit record for a simulation run."""
        if not run_id or not run_id.strip():
            raise SimulationAuditValidationError("run_id must be non-empty.")

        if sequence < 1:
            raise SimulationAuditValidationError("sequence must be at least 1.")

        if not event_type or not event_type.strip():
            raise SimulationAuditValidationError("event_type must be non-empty.")

        if occurred_at.tzinfo is None:
            raise SimulationAuditValidationError(
                "occurred_at must be timezone-aware for deterministic audit records."
            )

        if not isinstance(payload, Mapping):
            raise SimulationAuditValidationError("payload must be a mapping.")

        output_path = self._path_for_run(run_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "event_id": f"{run_id}:audit:{sequence:06d}",
            "event_type": event_type,
            "occurred_at": occurred_at.isoformat(),
            "payload": _sanitize(payload),
            "run_id": run_id,
            "sequence": sequence,
        }

        serialized = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

        with output_path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(serialized)
            audit_file.write("\n")
            audit_file.flush()
            os.fsync(audit_file.fileno())

        return output_path

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
