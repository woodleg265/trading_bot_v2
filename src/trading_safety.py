from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAFETY_STATE_FILE = DATA_DIR / "trading_safety_state.json"

MAX_SNAPSHOT_AGE = timedelta(minutes=15)
MAX_FUTURE_SKEW = timedelta(minutes=1)


class SafetyState(str, Enum):
    HALTED = "HALTED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    OBSERVATION_APPROVED = "OBSERVATION_APPROVED"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def parse_utc_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("Timestamp must be a non-empty ISO-8601 string.")

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include a timezone.")

    return parsed.astimezone(timezone.utc)


def local_data_file(filename: Any, prefix: str) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError("Expected a non-empty local filename.")

    candidate = Path(filename)
    if candidate.name != filename or candidate.suffix != ".json":
        raise ValueError("Expected a local JSON filename without path components.")

    if not filename.startswith(prefix):
        raise ValueError(f"Unexpected filename prefix: {filename}")

    resolved_data_dir = DATA_DIR.resolve()
    resolved_file = (DATA_DIR / filename).resolve()
    if resolved_file.parent != resolved_data_dir:
        raise ValueError("Path must remain directly inside data/.")

    return resolved_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_file = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        temporary_file.chmod(0o600)
        os.replace(temporary_file, path)
        path.chmod(0o600)
    finally:
        if temporary_file.exists():
            temporary_file.unlink()


def set_state(
    state: SafetyState,
    reason: str,
    *,
    snapshot_file: str | None = None,
    snapshot_sha256: str | None = None,
    reconciliation_report_file: str | None = None,
    expires_at_utc: str | None = None,
) -> dict[str, Any]:
    if state not in SafetyState:
        raise ValueError("Unknown safety state.")

    document: dict[str, Any] = {
        "schema_version": 1,
        "state": state.value,
        "updated_at_utc": utc_now_iso(),
        "reason": reason,
    }

    if snapshot_file is not None:
        document["snapshot_file"] = snapshot_file
    if snapshot_sha256 is not None:
        document["snapshot_sha256"] = snapshot_sha256
    if reconciliation_report_file is not None:
        document["reconciliation_report_file"] = reconciliation_report_file
    if expires_at_utc is not None:
        document["expires_at_utc"] = expires_at_utc

    atomic_write_json(SAFETY_STATE_FILE, document)
    return document


def set_halted(reason: str) -> dict[str, Any]:
    return set_state(SafetyState.HALTED, reason)


def set_reconciliation_required(reason: str) -> dict[str, Any]:
    return set_state(SafetyState.RECONCILIATION_REQUIRED, reason)


def approve_observation(
    *,
    snapshot_file: str,
    snapshot_sha256: str,
    reconciliation_report_file: str,
    reason: str,
) -> dict[str, Any]:
    expires_at = utc_now() + MAX_SNAPSHOT_AGE
    return set_state(
        SafetyState.OBSERVATION_APPROVED,
        reason,
        snapshot_file=snapshot_file,
        snapshot_sha256=snapshot_sha256,
        reconciliation_report_file=reconciliation_report_file,
        expires_at_utc=expires_at.isoformat(),
    )


def load_state() -> dict[str, Any]:
    with SAFETY_STATE_FILE.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError("Safety state must be a JSON object.")

    return value


def validate_observation_approval() -> tuple[bool, str, dict[str, Any] | None]:
    try:
        state = load_state()
    except FileNotFoundError:
        return False, "Safety state file is missing; system is halted.", None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return False, f"Safety state is unreadable or invalid: {exc}", None

    if state.get("state") != SafetyState.OBSERVATION_APPROVED.value:
        return False, f"Safety state is not observation-approved: {state.get('state')!r}.", state

    try:
        expires_at = parse_utc_timestamp(state.get("expires_at_utc"))
        if expires_at <= utc_now():
            return False, "Observation approval has expired.", state

        snapshot_path = local_data_file(
            state.get("snapshot_file"),
            "kraken_snapshot_",
        )
        report_path = local_data_file(
            state.get("reconciliation_report_file"),
            "ownership_reconciliation_",
        )

        if not snapshot_path.is_file():
            return False, "Approved snapshot file is missing.", state
        if not report_path.is_file():
            return False, "Approved reconciliation report is missing.", state

        expected_digest = state.get("snapshot_sha256")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            return False, "Safety state has an invalid snapshot digest.", state

        actual_digest = sha256_file(snapshot_path)
        if actual_digest != expected_digest:
            return False, "Approved snapshot digest does not match snapshot bytes.", state

        with snapshot_path.open("r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)

        if not isinstance(snapshot, dict) or not isinstance(report, dict):
            return False, "Approved snapshot or report is not a JSON object.", state

        if report.get("snapshot_sha256") != actual_digest:
            return False, "Reconciliation report digest does not match snapshot bytes.", state

        if report.get("snapshot_file") != snapshot_path.name:
            return False, "Reconciliation report references a different snapshot.", state

        snapshot_time = parse_utc_timestamp(snapshot.get("timestamp_utc"))
        age = utc_now() - snapshot_time
        if age > MAX_SNAPSHOT_AGE:
            return False, "Approved snapshot is stale.", state
        if age < -MAX_FUTURE_SKEW:
            return False, "Approved snapshot is materially future-dated.", state

        if snapshot.get("mode") != "OBSERVATION_ONLY":
            return False, "Approved snapshot is not observation-only.", state
        if snapshot.get("trading_enabled") is not False:
            return False, "Approved snapshot does not explicitly disable trading.", state
        if snapshot.get("open_order_count") != 0:
            return False, "Approved snapshot contains open orders.", state

    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return False, f"Observation approval validation failed: {exc}", state

    return True, "Fresh observation approval is valid.", state
