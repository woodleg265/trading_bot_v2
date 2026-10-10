from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"

REQUIRED_REPORT_FIELDS = {
    "reconciliation_status",
    "classification",
    "declared_bot_count",
    "snapshot_file",
}


class RuntimeState(str, Enum):
    PAUSED_RECONCILIATION = "PAUSED_RECONCILIATION"
    OBSERVATION = "OBSERVATION"
    PAPER_TRADING = "PAPER_TRADING"
    READY_FOR_LIVE = "READY_FOR_LIVE"
    LIVE_TRADING = "LIVE_TRADING"


def load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required configuration file is missing: {path}")

    values: dict[str, str] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def latest_reconciliation_report() -> Path | None:
    reports = sorted(
        path for path in DATA_DIR.glob("ownership_reconciliation_*.json")
        if path.is_file()
    )
    return reports[-1] if reports else None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")

    return value


def validate_report(report: dict[str, Any]) -> tuple[bool, str]:
    missing = sorted(REQUIRED_REPORT_FIELDS - report.keys())
    if missing:
        return False, f"Reconciliation report is missing required fields: {', '.join(missing)}."

    if report["reconciliation_status"] != "PASS_WITH_NO_BOT_ALLOCATIONS":
        return False, "Latest ownership reconciliation did not pass with zero bot allocations."

    if report["classification"] != "MANUAL_UNALLOCATED":
        return False, "Latest reconciliation does not classify holdings as manual/unallocated."

    if report["declared_bot_count"] != 0:
        return False, "A bot allocation exists; this initial v2 phase requires zero."

    snapshot_name = report["snapshot_file"]
    if not isinstance(snapshot_name, str) or not snapshot_name:
        return False, "Reconciliation report has an invalid snapshot_file value."

    snapshot_path = DATA_DIR / snapshot_name
    if Path(snapshot_name).name != snapshot_name:
        return False, "Reconciliation report snapshot_file must be a local filename."

    if not snapshot_name.startswith("kraken_snapshot_") or not snapshot_name.endswith(".json"):
        return False, "Reconciliation report references an unexpected snapshot filename."

    if not snapshot_path.is_file():
        return False, "Reconciliation report references a missing snapshot file."

    return True, "Reconciliation report passed strict observation validation."


def determine_state() -> tuple[RuntimeState, list[str], Path | None]:
    try:
        env = load_env_file(ENV_FILE)
    except (OSError, UnicodeDecodeError) as exc:
        return RuntimeState.PAUSED_RECONCILIATION, [str(exc)], None

    if env.get("TRADING_ENABLED") != "false":
        return (
            RuntimeState.PAUSED_RECONCILIATION,
            ["TRADING_ENABLED must be explicitly set to exactly false."],
            None,
        )

    if env.get("OBSERVATION_MODE") != "true":
        return (
            RuntimeState.PAUSED_RECONCILIATION,
            ["OBSERVATION_MODE must be explicitly set to exactly true."],
            None,
        )

    report_file = latest_reconciliation_report()
    if report_file is None:
        return (
            RuntimeState.PAUSED_RECONCILIATION,
            ["No ownership reconciliation report exists in data/."],
            None,
        )

    try:
        report = load_json(report_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return (
            RuntimeState.PAUSED_RECONCILIATION,
            [f"Could not read reconciliation report: {exc}"],
            report_file,
        )

    valid, reason = validate_report(report)
    if not valid:
        return RuntimeState.PAUSED_RECONCILIATION, [reason], report_file

    return (
        RuntimeState.OBSERVATION,
        [
            "Observation safeguards passed. "
            "No trading or order-submission capability exists."
        ],
        report_file,
    )


def main() -> int:
    DATA_DIR.mkdir(mode=0o700, exist_ok=True)

    state, reasons, report_file = determine_state()
    status = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_state": state.value,
        "trading_permitted": False,
        "kraken_interaction_performed": False,
        "latest_reconciliation_report": (
            report_file.name if report_file is not None else None
        ),
        "reasons": reasons,
        "blocked_states": [
            RuntimeState.PAPER_TRADING.value,
            RuntimeState.READY_FOR_LIVE.value,
            RuntimeState.LIVE_TRADING.value,
        ],
    }

    status_file = DATA_DIR / "runtime_status.json"
    temporary_file = DATA_DIR / "runtime_status.json.tmp"

    try:
        with temporary_file.open("w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2, sort_keys=True)
            handle.write("\n")

        temporary_file.chmod(0o600)
        os.replace(temporary_file, status_file)
        status_file.chmod(0o600)
    except OSError as exc:
        print(f"Unable to write runtime status safely: {exc}")
        return 1

    print(f"Runtime state: {state.value}")
    print("Trading permitted: False")
    print("Kraken interaction performed: False")
    for reason in reasons:
        print(f"- {reason}")
    print(f"Status file: {status_file.relative_to(PROJECT_ROOT)}")

    return 0 if state == RuntimeState.OBSERVATION else 1


if __name__ == "__main__":
    raise SystemExit(main())
