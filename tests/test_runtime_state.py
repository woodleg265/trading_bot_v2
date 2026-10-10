import json
from pathlib import Path

import pytest

from src import runtime_state


def write_env(tmp_path: Path, observation="true", trading="false") -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"OBSERVATION_MODE={observation}\nTRADING_ENABLED={trading}\n",
        encoding="utf-8",
    )
    return env_file


def write_report(data_dir: Path, **overrides) -> Path:
    snapshot_name = "kraken_snapshot_20261009T000000Z.json"
    (data_dir / snapshot_name).write_text("{}", encoding="utf-8")

    report = {
        "reconciliation_status": "PASS_WITH_NO_BOT_ALLOCATIONS",
        "classification": "MANUAL_UNALLOCATED",
        "declared_bot_count": 0,
        "snapshot_file": snapshot_name,
    }
    report.update(overrides)

    report_file = data_dir / "ownership_reconciliation_20261009T000000Z.json"
    report_file.write_text(json.dumps(report), encoding="utf-8")
    return report_file


@pytest.fixture
def isolated_runtime(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(runtime_state, "DATA_DIR", data_dir)
    monkeypatch.setattr(runtime_state, "ENV_FILE", write_env(tmp_path))
    return data_dir


def test_valid_observation_state(isolated_runtime):
    write_report(isolated_runtime)

    state, reasons, _ = runtime_state.determine_state()

    assert state == runtime_state.RuntimeState.OBSERVATION
    assert "safeguards passed" in reasons[0].lower()


def test_missing_report_pauses(isolated_runtime):
    state, reasons, report_file = runtime_state.determine_state()

    assert state == runtime_state.RuntimeState.PAUSED_RECONCILIATION
    assert report_file is None
    assert "No ownership reconciliation report" in reasons[0]


@pytest.mark.parametrize("observation,trading", [
    ("true", "true"),
    ("true", "yes"),
    ("true", "1"),
    ("false", "false"),
    ("yes", "false"),
    ("1", "false"),
])
def test_non_exact_environment_values_pause(monkeypatch, tmp_path, observation, trading):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(runtime_state, "DATA_DIR", data_dir)
    monkeypatch.setattr(runtime_state, "ENV_FILE", write_env(tmp_path, observation, trading))
    write_report(data_dir)

    state, _, _ = runtime_state.determine_state()

    assert state == runtime_state.RuntimeState.PAUSED_RECONCILIATION


def test_malformed_report_pauses(isolated_runtime):
    report = isolated_runtime / "ownership_reconciliation_20261009T000000Z.json"
    report.write_text("{not valid json", encoding="utf-8")

    state, reasons, _ = runtime_state.determine_state()

    assert state == runtime_state.RuntimeState.PAUSED_RECONCILIATION
    assert "Could not read reconciliation report" in reasons[0]


@pytest.mark.parametrize("field", [
    "reconciliation_status",
    "classification",
    "declared_bot_count",
    "snapshot_file",
])
def test_missing_required_report_field_pauses(isolated_runtime, field):
    report_file = write_report(isolated_runtime)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    del report[field]
    report_file.write_text(json.dumps(report), encoding="utf-8")

    state, reasons, _ = runtime_state.determine_state()

    assert state == runtime_state.RuntimeState.PAUSED_RECONCILIATION
    assert "missing required fields" in reasons[0]


@pytest.mark.parametrize("overrides", [
    {"reconciliation_status": "FAILED"},
    {"classification": "BOT_ALLOCATED"},
    {"declared_bot_count": 1},
    {"snapshot_file": "missing_snapshot.json"},
    {"snapshot_file": "../outside.json"},
])
def test_unsafe_report_values_pause(isolated_runtime, overrides):
    write_report(isolated_runtime, **overrides)

    state, _, _ = runtime_state.determine_state()

    assert state == runtime_state.RuntimeState.PAUSED_RECONCILIATION
