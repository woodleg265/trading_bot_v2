from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .trading_safety import (
        MAX_FUTURE_SKEW,
        MAX_SNAPSHOT_AGE,
        approve_observation,
        atomic_write_json,
        parse_utc_timestamp,
        set_reconciliation_required,
        sha256_file,
    )
except ImportError:  # Supports direct execution: python src/reconcile_ownership.py
    from trading_safety import (
        MAX_FUTURE_SKEW,
        MAX_SNAPSHOT_AGE,
        approve_observation,
        atomic_write_json,
        parse_utc_timestamp,
        set_reconciliation_required,
        sha256_file,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_FILE = PROJECT_ROOT / "config" / "ownership_ledger.json"


def latest_snapshot_file() -> Path:
    snapshots = sorted(
        path for path in DATA_DIR.glob("kraken_snapshot_*.json") if path.is_file()
    )
    if not snapshots:
        raise FileNotFoundError("No Kraken snapshot found in data/.")
    return snapshots[-1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def fail(reason: str) -> int:
    set_reconciliation_required(reason)
    print(f"RECONCILIATION FAILED: {reason}", file=sys.stderr)
    return 1


def main() -> int:
    try:
        snapshot_file = latest_snapshot_file()
        snapshot = load_json(snapshot_file)
        ledger = load_json(CONFIG_FILE)

        if snapshot.get("mode") != "OBSERVATION_ONLY":
            return fail("Snapshot was not created in observation mode.")

        if snapshot.get("trading_enabled") is not False:
            return fail("Snapshot indicates trading was enabled.")

        snapshot_time = parse_utc_timestamp(snapshot.get("timestamp_utc"))
        now = datetime.now(timezone.utc)
        age = now - snapshot_time

        if age > MAX_SNAPSHOT_AGE:
            return fail("Snapshot is older than 15 minutes.")
        if age < -MAX_FUTURE_SKEW:
            return fail("Snapshot is materially future-dated.")

        open_order_count = snapshot.get("open_order_count")
        if not isinstance(open_order_count, int) or isinstance(open_order_count, bool):
            return fail("Snapshot open_order_count must be an integer.")
        if open_order_count != 0:
            return fail("Snapshot contains open orders; manual review is required.")

        policy = ledger.get("account_policy")
        if not isinstance(policy, dict):
            return fail("Ownership ledger is missing account_policy.")

        if policy.get("default_classification") != "MANUAL_UNALLOCATED":
            return fail("Ownership policy does not classify funds as MANUAL_UNALLOCATED.")
        if policy.get("bot_may_trade_unallocated_funds") is not False:
            return fail("Ownership policy must prohibit trading unallocated funds.")
        if policy.get("reconciliation_required_before_trading") is not True:
            return fail("Ownership policy must require reconciliation.")

        bots = ledger.get("bots")
        if not isinstance(bots, list) or bots:
            return fail("Expected no bot allocations in the initial ownership ledger.")

        balances = snapshot.get("balances_nonzero", {})
        if not isinstance(balances, dict):
            return fail("Snapshot balances_nonzero must be an object.")

        snapshot_digest = sha256_file(snapshot_file)

        report = {
            "schema_version": 1,
            "report_timestamp_utc": now.isoformat(),
            "snapshot_file": snapshot_file.name,
            "snapshot_sha256": snapshot_digest,
            "classification": "MANUAL_UNALLOCATED",
            "declared_bot_count": 0,
            "nonzero_balance_entries": len(balances),
            "open_order_count": open_order_count,
            "reconciliation_status": "PASS_WITH_NO_BOT_ALLOCATIONS",
            "notes": [
                "No v2 bot has been allocated any exchange funds.",
                "All current Kraken balances are classified as manual/unallocated.",
                "No future bot may trade these funds until an explicit allocation is added.",
                "This command is read-only and does not interact with Kraken.",
            ],
        }

        report_name = (
            "ownership_reconciliation_"
            + now.strftime("%Y%m%dT%H%M%SZ")
            + ".json"
        )
        report_file = DATA_DIR / report_name
        atomic_write_json(report_file, report)

        approve_observation(
            snapshot_file=snapshot_file.name,
            snapshot_sha256=snapshot_digest,
            reconciliation_report_file=report_file.name,
            reason="Fresh zero-open-order observation reconciliation passed.",
        )

    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return fail(str(exc))

    print("Ownership reconciliation: PASS")
    print("Declared v2 bot allocations: 0")
    print(f"Current non-zero balance entries: {len(balances)}")
    print("Current open orders: 0")
    print("Classification: MANUAL_UNALLOCATED")
    print(f"Report file: {report_file.name}")
    print("Trading permitted: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
