from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_FILE = PROJECT_ROOT / "config" / "ownership_ledger.json"


def latest_snapshot_file() -> Path:
    snapshots = sorted(DATA_DIR.glob("kraken_snapshot_*.json"))
    if not snapshots:
        raise FileNotFoundError("No Kraken snapshot found in data/.")

    return snapshots[-1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    try:
        snapshot_file = latest_snapshot_file()
        snapshot = load_json(snapshot_file)
        ledger = load_json(CONFIG_FILE)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"RECONCILIATION FAILED: {exc}", file=sys.stderr)
        return 1

    if snapshot.get("mode") != "OBSERVATION_ONLY":
        print("RECONCILIATION FAILED: Snapshot was not created in observation mode.")
        return 1

    if snapshot.get("trading_enabled") is not False:
        print("RECONCILIATION FAILED: Snapshot indicates trading was enabled.")
        return 1

    bots = ledger.get("bots", [])
    if bots:
        print("RECONCILIATION FAILED: Expected no bot allocations in initial ledger.")
        return 1

    balances = snapshot.get("balances_nonzero", {})
    open_order_count = snapshot.get("open_order_count", 0)

    report = {
        "report_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_file": snapshot_file.name,
        "classification": "MANUAL_UNALLOCATED",
        "declared_bot_count": 0,
        "nonzero_balance_entries": len(balances),
        "open_order_count": open_order_count,
        "reconciliation_status": "PASS_WITH_NO_BOT_ALLOCATIONS",
        "notes": [
            "No v2 bot has been allocated any exchange funds.",
            "All current Kraken balances are classified as manual/unallocated.",
            "No future bot may trade these funds until an explicit allocation is added.",
            "This command is read-only and does not interact with Kraken."
        ]
    }

    report_name = (
        "ownership_reconciliation_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + ".json"
    )
    report_file = DATA_DIR / report_name

    with report_file.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    report_file.chmod(0o600)

    print("Ownership reconciliation: PASS")
    print("Declared v2 bot allocations: 0")
    print(f"Current non-zero balance entries: {len(balances)}")
    print(f"Current open orders: {open_order_count}")
    print("Classification: MANUAL_UNALLOCATED")
    print(f"Report file: {report_file.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
