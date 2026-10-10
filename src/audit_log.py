from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
AUDIT_LOG = LOG_DIR / "audit_events.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_event(event_type: str, details: dict[str, Any]) -> None:
    """
    Append one structured, timestamped event.

    Never include API keys, API secrets, or other credentials in `details`.
    """
    LOG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)

    event = {
        "timestamp_utc": utc_now(),
        "event_type": event_type,
        "details": details,
    }

    with AUDIT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")

    AUDIT_LOG.chmod(0o600)
