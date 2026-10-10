from pathlib import Path

from src import run_observation


def test_runner_stops_at_first_failed_step(monkeypatch, tmp_path):
    audit_events = []
    calls = []

    monkeypatch.setattr(run_observation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        run_observation,
        "STEPS",
        [
            ("snapshot", Path("snapshot.py")),
            ("reconciliation", Path("reconcile.py")),
            ("runtime_state", Path("runtime.py")),
        ],
    )

    def fake_run_step(name, script):
        calls.append(name)
        if name == "reconciliation":
            return {"name": name, "status": "FAILED", "return_code": 1}
        return {"name": name, "status": "PASSED", "return_code": 0}

    monkeypatch.setattr(run_observation, "run_step", fake_run_step)
    monkeypatch.setattr(
        run_observation,
        "write_audit_event",
        lambda event: audit_events.append(event) or tmp_path / "audit.jsonl",
    )

    assert run_observation.main() == 1
    assert calls == ["snapshot", "reconciliation"]
    assert len(audit_events) == 1
    assert audit_events[0]["overall_status"] == "FAILED"
    assert audit_events[0]["trading_permitted"] is False
    assert audit_events[0]["order_submission_attempted"] is False


def test_runner_success_remains_non_trading(monkeypatch, tmp_path):
    audit_events = []

    monkeypatch.setattr(run_observation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        run_observation,
        "STEPS",
        [("snapshot", Path("snapshot.py"))],
    )
    monkeypatch.setattr(
        run_observation,
        "run_step",
        lambda name, script: {
            "name": name,
            "status": "PASSED",
            "return_code": 0,
        },
    )
    monkeypatch.setattr(
        run_observation,
        "write_audit_event",
        lambda event: audit_events.append(event) or tmp_path / "audit.jsonl",
    )

    assert run_observation.main() == 0
    assert len(audit_events) == 1
    assert audit_events[0]["overall_status"] == "PASSED"
    assert audit_events[0]["trading_permitted"] is False
    assert audit_events[0]["order_submission_attempted"] is False
