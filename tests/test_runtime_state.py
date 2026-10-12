from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from src import runtime_state, trading_safety


def write_env(tmp_path: Path, observation: str = "true", trading: str = "false") -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"OBSERVATION_MODE={observation}\nTRADING_ENABLED={trading}\n",
        encoding="utf-8",
    )
    return env_file


def write_snapshot(data_dir: Path, **overrides: object) -> Path:
    snapshot_name = "kraken_snapshot_20261012T000000Z.json"
    snapshot = {
        "timestamp_utc": trading_safety.utc_now_iso(),
        "mode": "OBSERVATION_ONLY",
        "trading_enabled": False,
        "open_order_count": 0,
        "balances": {},
    }
    snapshot.update(overrides)

    snapshot_file = data_dir / snapshot_name
    snapshot_file.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshot_file.chmod(0o600)
    return snapshot_file


def write_report(
    data_dir: Path,
    snapshot_file: Path,
    snapshot_sha256: str,
    **overrides: object,
) -> Path:
    report_name = "ownership_reconciliation_20261012T000000Z.json"
    report = {
        "reconciliation_status": "PASS_WITH_NO_BOT_ALLOCATIONS",
        "classification": "MANUAL_UNALLOCATED",
        "declared_bot_count": 0,
        "snapshot_file": snapshot_file.name,
        "snapshot_sha256": snapshot_sha256,
    }
    report.update(overrides)

    report_file = data_dir / report_name
    report_file.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_file.chmod(0o600)
    return report_file


def establish_valid_approval(data_dir: Path) -> tuple[Path, Path]:
    snapshot_file = write_snapshot(data_dir)
    snapshot_sha256 = trading_safety.sha256_file(snapshot_file)
    report_file = write_report(data_dir, snapshot_file, snapshot_sha256)

    trading_safety.approve_observation(
        snapshot_file=snapshot_file.name,
        snapshot_sha256=snapshot_sha256,
        reconciliation_report_file=report_file.name,
        reason="Test-only fresh observation approval.",
    )
    return snapshot_file, report_file


@pytest.fixture
def isolated_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(mode=0o700)

    safety_state_file = data_dir / "trading_safety_state.json"

    monkeypatch.setattr(runtime_state, "DATA_DIR", data_dir)
    monkeypatch.setattr(runtime_state, "ENV_FILE", write_env(tmp_path))
    monkeypatch.setattr(runtime_state, "SAFETY_STATE_FILE", safety_state_file)

    monkeypatch.setattr(trading_safety, "DATA_DIR", data_dir)
    monkeypatch.setattr(trading_safety, "SAFETY_STATE_FILE", safety_state_file)

    return data_dir


def test_missing_safety_state_halts(isolated_runtime: Path):
    state, reasons, state_file = runtime_state.determine_state()

    assert state == runtime_state.RuntimeState.HALTED
    assert state_file == isolated_runtime / "trading_safety_state.json"
    assert "missing" in reasons[0].lower()


def test_reconciliation_required_state_is_exposed(isolated_runtime: Path):
    trading_safety.set_reconciliation_required("Fresh Kraken observation is required.")

    state, reasons, state_file = runtime_state.determine_state()

    assert state == runtime_state.RuntimeState.RECONCILIATION_REQUIRED
    assert state_file == isolated_runtime / "trading_safety_state.json"
    assert "not observation-approved" in reasons[0].lower()


def test_valid_fresh_approval_is_observation_approved(isolated_runtime: Path):
    establish_valid_approval(isolated_runtime)

    state, reasons, state_file = runtime_state.determine_state()

    assert state == runtime_state.RuntimeState.OBSERVATION_APPROVED
    assert state_file == isolated_runtime / "trading_safety_state.json"
    assert reasons == ["Fresh observation approval is valid."]


@pytest.mark.parametrize(
    ("observation", "trading"),
    [
        ("true", "true"),
        ("true", "yes"),
        ("true", "1"),
        ("false", "false"),
        ("yes", "false"),
        ("1", "false"),
    ],
)
def test_invalid_environment_values_halt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    observation: str,
    trading: str,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir(mode=0o700)
    safety_state_file = data_dir / "trading_safety_state.json"

    monkeypatch.setattr(runtime_state, "DATA_DIR", data_dir)
    monkeypatch.setattr(
        runtime_state,
        "ENV_FILE",
        write_env(tmp_path, observation=observation, trading=trading),
    )
    monkeypatch.setattr(runtime_state, "SAFETY_STATE_FILE", safety_state_file)
    monkeypatch.setattr(trading_safety, "DATA_DIR", data_dir)
    monkeypatch.setattr(trading_safety, "SAFETY_STATE_FILE", safety_state_file)

    establish_valid_approval(data_dir)

    state, reasons, _ = runtime_state.determine_state()

    assert state == runtime_state.RuntimeState.HALTED
    assert reasons


def test_expired_approval_halts(isolated_runtime: Path):
    establish_valid_approval(isolated_runtime)

    state_document = trading_safety.load_state()
    state_document["expires_at_utc"] = (
        trading_safety.utc_now() - timedelta(seconds=1)
    ).isoformat()
    trading_safety.atomic_write_json(trading_safety.SAFETY_STATE_FILE, state_document)

    state, reasons, _ = runtime_state.determine_state()

    assert state == runtime_state.RuntimeState.HALTED
    assert "expired" in reasons[0].lower()


def test_snapshot_digest_mismatch_halts(isolated_runtime: Path):
    snapshot_file, _ = establish_valid_approval(isolated_runtime)
    snapshot_file.write_text('{"tampered": true}\n', encoding="utf-8")

    state, reasons, _ = runtime_state.determine_state()

    assert state == runtime_state.RuntimeState.HALTED
    assert "digest" in reasons[0].lower()


def test_report_digest_mismatch_halts(isolated_runtime: Path):
    snapshot_file, report_file = establish_valid_approval(isolated_runtime)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    report["snapshot_sha256"] = "0" * 64
    report_file.write_text(json.dumps(report), encoding="utf-8")

    state, reasons, _ = runtime_state.determine_state()

    assert state == runtime_state.RuntimeState.HALTED
    assert "digest" in reasons[0].lower()

    # Keep the snapshot variable intentionally used/documented by this test.
    assert snapshot_file.is_file()


def test_open_orders_in_approved_snapshot_halt(isolated_runtime: Path):
    snapshot_file, _ = establish_valid_approval(isolated_runtime)
    snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
    snapshot["open_order_count"] = 1
    snapshot_file.write_text(json.dumps(snapshot), encoding="utf-8")

    state, reasons, _ = runtime_state.determine_state()

    assert state == runtime_state.RuntimeState.HALTED
    assert "digest" in reasons[0].lower()


def test_malformed_safety_state_halts(isolated_runtime: Path):
    safety_state_file = isolated_runtime / "trading_safety_state.json"
    safety_state_file.write_text("{not valid json", encoding="utf-8")

    state, reasons, _ = runtime_state.determine_state()

    assert state == runtime_state.RuntimeState.HALTED
    assert "invalid" in reasons[0].lower() or "unreadable" in reasons[0].lower()
