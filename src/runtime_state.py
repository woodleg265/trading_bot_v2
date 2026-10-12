from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

try:
    from .trading_safety import SAFETY_STATE_FILE, validate_observation_approval
except ImportError:  # Supports direct execution: python src/runtime_state.py
    from trading_safety import SAFETY_STATE_FILE, validate_observation_approval

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"


class RuntimeState(str, Enum):
    HALTED = "HALTED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    OBSERVATION_APPROVED = "OBSERVATION_APPROVED"


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


def determine_state() -> tuple[RuntimeState, list[str], Path | None]:
    try:
        env = load_env_file(ENV_FILE)
    except (OSError, UnicodeDecodeError) as exc:
        return RuntimeState.HALTED, [str(exc)], None

    if env.get("TRADING_ENABLED") != "false":
        return (
            RuntimeState.HALTED,
            ["TRADING_ENABLED must be explicitly set to exactly false."],
            None,
        )

    if env.get("OBSERVATION_MODE") != "true":
        return (
            RuntimeState.HALTED,
            ["OBSERVATION_MODE must be explicitly set to exactly true."],
            None,
        )

    valid, reason, state_document = validate_observation_approval()
    if valid:
        return RuntimeState.OBSERVATION_APPROVED, [reason], SAFETY_STATE_FILE

    persisted_state = (
        state_document.get("state") if isinstance(state_document, dict) else None
    )
    if persisted_state == RuntimeState.RECONCILIATION_REQUIRED.value:
        return RuntimeState.RECONCILIATION_REQUIRED, [reason], SAFETY_STATE_FILE

    return RuntimeState.HALTED, [reason], SAFETY_STATE_FILE


def main() -> int:
    DATA_DIR.mkdir(mode=0o700, exist_ok=True)

    state, reasons, state_file = determine_state()
    status = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_state": state.value,
        "trading_permitted": False,
        "kraken_interaction_performed": False,
        "trading_safety_state_file": (
            state_file.name if state_file is not None else None
        ),
        "reasons": reasons,
        "blocked_states": ["LIVE_TRADING", "READY_FOR_LIVE", "PAPER_TRADING"],
    }

    status_file = DATA_DIR / "runtime_status.json"
    temporary_file = DATA_DIR / "runtime_status.json.tmp"

    try:
        with temporary_file.open("w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

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

    return 0 if state == RuntimeState.OBSERVATION_APPROVED else 1


if __name__ == "__main__":
    raise SystemExit(main())
