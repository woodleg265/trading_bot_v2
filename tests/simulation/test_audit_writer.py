"""Tests for the isolated simulation JSONL audit writer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path

import pytest

from src.simulation.audit_writer import (
    SimulationAuditPathError,
    SimulationAuditValidationError,
    SimulationAuditWriter,
)


class SampleStatus(Enum):
    ACCEPTED = "ACCEPTED"


@pytest.fixture
def audit_path() -> Path:
    path = Path("data/simulation/sim-test-audit-writer.jsonl").resolve()
    path.unlink(missing_ok=True)
    yield path
    path.unlink(missing_ok=True)


def test_append_event_sanitizes_and_appends_jsonl(audit_path: Path) -> None:
    writer = SimulationAuditWriter()
    occurred_at = datetime(2026, 10, 14, 18, 0, tzinfo=timezone.utc)

    output_path = writer.append_event(
        run_id="sim-test-audit-writer",
        sequence=1,
        event_type="FILL_CREATED",
        occurred_at=occurred_at,
        payload={
            "price": Decimal("123.4500"),
            "status": SampleStatus.ACCEPTED,
            "api_secret": "must-not-appear",
            "nested": {
                "access_token": "must-not-appear",
                "quantity": Decimal("2.5"),
            },
        },
    )

    assert output_path == audit_path
    assert output_path.exists()

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert records == [
        {
            "event_id": "sim-test-audit-writer:audit:000001",
            "event_type": "FILL_CREATED",
            "occurred_at": "2026-10-14T18:00:00+00:00",
            "payload": {
                "api_secret": "[REDACTED]",
                "nested": {
                    "access_token": "[REDACTED]",
                    "quantity": "2.5",
                },
                "price": "123.4500",
                "status": "ACCEPTED",
            },
            "run_id": "sim-test-audit-writer",
            "sequence": 1,
        }
    ]


def test_append_event_appends_multiple_deterministic_records(audit_path: Path) -> None:
    writer = SimulationAuditWriter()
    occurred_at = datetime(2026, 10, 14, 18, 0, tzinfo=timezone.utc)

    writer.append_event(
        run_id="sim-test-audit-writer",
        sequence=1,
        event_type="INTENT_CREATED",
        occurred_at=occurred_at,
        payload={},
    )
    writer.append_event(
        run_id="sim-test-audit-writer",
        sequence=2,
        event_type="RISK_APPROVED",
        occurred_at=occurred_at,
        payload={},
    )

    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["event_id"] for record in records] == [
        "sim-test-audit-writer:audit:000001",
        "sim-test-audit-writer:audit:000002",
    ]


@pytest.mark.parametrize(
    "root_directory",
    [
        "/tmp/simulation",
        "data",
        "data/real_account",
        "data/simulation/../real_account",
        "../data/simulation",
    ],
)
def test_rejects_roots_outside_simulation_directory(root_directory: str) -> None:
    with pytest.raises(SimulationAuditPathError):
        SimulationAuditWriter(root_directory)


@pytest.mark.parametrize(
    "run_id",
    [
        "../ownership_ledger",
        "subdirectory/run",
        r"subdirectory\run",
        ".",
        "..",
        "",
        "   ",
    ],
)
def test_rejects_unsafe_or_empty_run_ids(run_id: str) -> None:
    writer = SimulationAuditWriter()

    expected_exception = (
        SimulationAuditValidationError
        if not run_id.strip()
        else SimulationAuditPathError
    )

    with pytest.raises(expected_exception):
        writer.append_event(
            run_id=run_id,
            sequence=1,
            event_type="TEST_EVENT",
            occurred_at=datetime(2026, 10, 14, 18, 0, tzinfo=timezone.utc),
            payload={},
        )


@pytest.mark.parametrize("sequence", [0, -1])
def test_rejects_non_positive_sequence(sequence: int) -> None:
    writer = SimulationAuditWriter()

    with pytest.raises(SimulationAuditValidationError, match="sequence"):
        writer.append_event(
            run_id="sim-test-validation",
            sequence=sequence,
            event_type="TEST_EVENT",
            occurred_at=datetime(2026, 10, 14, 18, 0, tzinfo=timezone.utc),
            payload={},
        )


def test_rejects_naive_timestamp() -> None:
    writer = SimulationAuditWriter()

    with pytest.raises(SimulationAuditValidationError, match="timezone-aware"):
        writer.append_event(
            run_id="sim-test-validation",
            sequence=1,
            event_type="TEST_EVENT",
            occurred_at=datetime(2026, 10, 14, 18, 0),
            payload={},
        )


def test_rejects_non_mapping_payload() -> None:
    writer = SimulationAuditWriter()

    with pytest.raises(SimulationAuditValidationError, match="mapping"):
        writer.append_event(
            run_id="sim-test-validation",
            sequence=1,
            event_type="TEST_EVENT",
            occurred_at=datetime(2026, 10, 14, 18, 0, tzinfo=timezone.utc),
            payload=["not", "a", "mapping"],  # type: ignore[arg-type]
        )


def test_rejects_unsupported_payload_value_types() -> None:
    writer = SimulationAuditWriter()

    with pytest.raises(SimulationAuditValidationError, match="Unsupported"):
        writer.append_event(
            run_id="sim-test-validation",
            sequence=1,
            event_type="TEST_EVENT",
            occurred_at=datetime(2026, 10, 14, 18, 0, tzinfo=timezone.utc),
            payload={"unsupported": object()},
        )
