from __future__ import annotations

import os
import sys
from pathlib import Path

import krakenex
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> int:
    load_dotenv(ENV_FILE)

    observation_mode = os.getenv("OBSERVATION_MODE", "").strip().lower()
    trading_enabled = os.getenv("TRADING_ENABLED", "").strip().lower()

    if observation_mode != "true":
        print("SAFETY STOP: OBSERVATION_MODE must be true.", file=sys.stderr)
        return 1

    if trading_enabled != "false":
        print("SAFETY STOP: TRADING_ENABLED must be false.", file=sys.stderr)
        return 1

    try:
        api_key = required_env("KRAKEN_API_KEY")
        api_secret = required_env("KRAKEN_API_SECRET")
    except RuntimeError as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 1

    api = krakenex.API(key=api_key, secret=api_secret)

    print("Mode: OBSERVATION ONLY")
    print("Trading: DISABLED")
    print("Checking Kraken read-only connectivity...")

    try:
        balance_response = api.query_private("Balance")
    except Exception as exc:
        print(f"CONNECTION ERROR: Could not contact Kraken: {exc}", file=sys.stderr)
        return 1

    errors = balance_response.get("error", [])
    if errors:
        print("KRAKEN API ERROR:", "; ".join(errors), file=sys.stderr)
        return 1

    balances = balance_response.get("result", {})
    nonzero_balances = {
        asset: amount
        for asset, amount in balances.items()
        if float(amount) != 0.0
    }

    print("Kraken authentication: SUCCESS")
    print(f"Non-zero balance entries returned: {len(nonzero_balances)}")

    try:
        orders_response = api.query_private("OpenOrders")
    except Exception as exc:
        print(f"WARNING: Balance check succeeded, but OpenOrders failed: {exc}")
        return 0

    order_errors = orders_response.get("error", [])
    if order_errors:
        print(
            "WARNING: Balance check succeeded, but OpenOrders returned: "
            + "; ".join(order_errors)
        )
        return 0

    open_orders = orders_response.get("result", {}).get("open", {})
    print(f"Open orders returned: {len(open_orders)}")
    print("Read-only connectivity check completed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
