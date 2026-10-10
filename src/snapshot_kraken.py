from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import krakenex
from dotenv import load_dotenv

from audit_log import append_event


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def is_nonzero(value: str) -> bool:
    try:
        return Decimal(value) != Decimal("0")
    except InvalidOperation:
        raise RuntimeError("Kraken returned an invalid numeric balance value.")


def main() -> int:
    load_dotenv(ENV_FILE)

    if os.getenv("OBSERVATION_MODE", "").strip().lower() != "true":
        print("SAFETY STOP: OBSERVATION_MODE must be true.", file=sys.stderr)
        return 1

    if os.getenv("TRADING_ENABLED", "").strip().lower() != "false":
        print("SAFETY STOP: TRADING_ENABLED must be false.", file=sys.stderr)
        return 1

    try:
        api = krakenex.API(
            key=required_env("KRAKEN_API_KEY"),
            secret=required_env("KRAKEN_API_SECRET"),
        )

        balance_response = api.query_private("Balance")
        balance_errors = balance_response.get("error", [])
        if balance_errors:
            raise RuntimeError("; ".join(balance_errors))

        orders_response = api.query_private("OpenOrders")
        order_errors = orders_response.get("error", [])
        if order_errors:
            raise RuntimeError("; ".join(order_errors))

        balances = balance_response.get("result", {})
        nonzero_balances = {
            asset: amount
            for asset, amount in balances.items()
            if is_nonzero(amount)
        }

        open_orders = orders_response.get("result", {}).get("open", {})

    except Exception as exc:
        append_event(
            "KRAKEN_SNAPSHOT_FAILED",
            {
                "reason": str(exc),
                "observation_mode": True,
                "trading_enabled": False,
            },
        )
        print(f"SNAPSHOT FAILED: {exc}", file=sys.stderr)
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    DATA_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    snapshot_file = DATA_DIR / f"kraken_snapshot_{timestamp}.json"

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "OBSERVATION_ONLY",
        "trading_enabled": False,
        "balances_nonzero": nonzero_balances,
        "open_order_count": len(open_orders),
        "open_orders": open_orders,
    }

    with snapshot_file.open("x", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
        handle.write("\n")

    snapshot_file.chmod(0o600)

    append_event(
        "KRAKEN_SNAPSHOT_CREATED",
        {
            "snapshot_file": snapshot_file.name,
            "nonzero_balance_entries": len(nonzero_balances),
            "open_order_count": len(open_orders),
            "observation_mode": True,
            "trading_enabled": False,
        },
    )

    print("Observation snapshot created successfully.")
    print(f"Non-zero balance entries: {len(nonzero_balances)}")
    print(f"Open orders: {len(open_orders)}")
    print(f"Snapshot file: {snapshot_file.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
