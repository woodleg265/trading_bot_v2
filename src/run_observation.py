from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .trading_safety import set_reconciliation_required
except ImportError:  # Supports direct execution: python src/run_observation.py
    from trading_safety import set_reconciliation_required

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
LOG_DIR = PROJECT_ROOT / "logs"

MAX_DIAGNOSTIC_CHARS = 2000

STEPS = [
    ("snapshot", SRC_DIR / "snapshot_kraken.py"),
    ("reconciliation", SRC_DIR / "reconcile_ownership.py"),
    ("runtime_state", SRC_DIR / "runtime_state.py"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_diagnostic(value: str) -> str:
    value = value.replace("\x00", "")
    lines: list[str] = []

    for line in value.splitlines():
        upper = line.upper()
        if any(
            marker in upper
            for marker in (
                "KRAKEN_API_KEY",
                "KRAKEN_API_SECRET",
                "API_KEY=",
                "API_SECRET=",
                "AUTHORIZATION:",
                "BEARER ",
            )
        ):
            lines.append("[redacted sensitive diagnostic line]")
        else:
            lines.append(line)

    result = "\n".join(lines)
    if len(result) > MAX_DIAGNOSTIC_CHARS:
        return result[:MAX_DIAGNOSTIC_CHARS] + "\n[truncated]"
    return result


def write_audit_event(event: dict[str, Any]) -> Path:
    LOG_DIR.mkdir(mode=0o700, exist_ok=True)
    LOG_DIR.chmod(0o700)

    audit_file = LOG_DIR / "observation_runner.jsonl"
    with audit_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True))
        handle.write("\n")

    audit_file.chmod(0o600)
    return audit_file


def run_step(name: str, script: Path) -> dict[str, Any]:
    if not script.is_file():
        return {
            "name": name,
            "script": str(script.relative_to(PROJECT_ROOT)),
            "status": "FAILED",
            "return_code": None,
            "error": "Required script is missing.",
        }

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "script": str(script.relative_to(PROJECT_ROOT)),
            "status": "FAILED",
            "return_code": None,
            "error": "Timed out after 90 seconds.",
        }
    except OSError as exc:
        return {
            "name": name,
            "script": str(script.relative_to(PROJECT_ROOT)),
            "status": "FAILED",
            "return_code": None,
            "error": f"Could not start process: {exc}",
        }

    return {
        "name": name,
        "script": str(script.relative_to(PROJECT_ROOT)),
        "status": "PASSED" if result.returncode == 0 else "FAILED",
        "return_code": result.returncode,
        "stdout": safe_diagnostic(result.stdout),
        "stderr": safe_diagnostic(result.stderr),
    }


def main() -> int:
    started_at = utc_now()
    step_results: list[dict[str, Any]] = []

    print("Starting read-only observation run.")
    print("Trading capability: unavailable")
    print()

    for name, script in STEPS:
        print(f"Running {name}...")
        result = run_step(name, script)
        step_results.append(result)

        if result["status"] != "PASSED":
            reason = f"Observation run failed during {name}."
            set_reconciliation_required(reason)
            print(f"{name}: FAILED")
            if "error" in result:
                print(f"- {result['error']}")

            event = {
                "event_type": "OBSERVATION_RUN",
                "started_at_utc": started_at,
                "finished_at_utc": utc_now(),
                "overall_status": "FAILED",
                "trading_permitted": False,
                "order_submission_attempted": False,
                "steps": step_results,
            }

            audit_file = write_audit_event(event)
            print()
            print("Observation run stopped safely.")
            print(f"Audit event recorded: {audit_file.relative_to(PROJECT_ROOT)}")
            return 1

        print(f"{name}: PASSED")

    event = {
        "event_type": "OBSERVATION_RUN",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "overall_status": "PASSED",
        "trading_permitted": False,
        "order_submission_attempted": False,
        "steps": step_results,
    }

    audit_file = write_audit_event(event)
    print()
    print("Observation run completed successfully.")
    print("Trading permitted: False")
    print("Order submission attempted: False")
    print(f"Audit event recorded: {audit_file.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
