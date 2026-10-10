#!/usr/bin/env python3
"""
Public, market-data-only observation.

Safety properties:
- Uses only Kraken's public ticker endpoint.
- Does not load API credentials.
- Does not import private-account or trading modules.
- Does not create, sign, or submit orders.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "market_observations"
DEFAULT_PAIR = "XBTUSD"
KRAKEN_TICKER_URL = "https://api.kraken.com/0/public/Ticker"


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso_timestamp():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_market_pair():
    pair = os.getenv("MARKET_OBSERVATION_PAIR", DEFAULT_PAIR).strip().upper()
    if not pair:
        raise ValueError("MARKET_OBSERVATION_PAIR must not be empty")
    if not pair.isalnum():
        raise ValueError(
            "MARKET_OBSERVATION_PAIR must contain only letters and numbers"
        )
    return pair


def fetch_public_ticker(pair):
    query = urlencode({"pair": pair})
    url = f"{KRAKEN_TICKER_URL}?{query}"

    try:
        with urlopen(url, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Public ticker request returned HTTP {response.status}"
                )
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(
            f"Public ticker request failed with HTTP {exc.code}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Public ticker network failure: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Public ticker response was not valid JSON"
        ) from exc

    validate_ticker_payload(payload, pair)
    return payload


def validate_ticker_payload(payload, requested_pair):
    if not isinstance(payload, dict):
        raise ValueError("Ticker payload must be a JSON object")

    errors = payload.get("error")
    if not isinstance(errors, list):
        raise ValueError("Ticker payload has an invalid error field")
    if errors:
        raise ValueError(f"Kraken returned ticker errors: {errors}")

    results = payload.get("result")
    if not isinstance(results, dict) or not results:
        raise ValueError(f"No ticker result returned for {requested_pair}")

    ticker_name, ticker = next(iter(results.items()))
    if not isinstance(ticker, dict):
        raise ValueError("Ticker result entry must be an object")

    last_trade = ticker.get("c")
    if not isinstance(last_trade, list) or not last_trade:
        raise ValueError("Ticker result is missing last-trade price")

    try:
        price = float(last_trade[0])
    except (TypeError, ValueError) as exc:
        raise ValueError("Ticker last-trade price is invalid") from exc

    if price <= 0:
        raise ValueError("Ticker last-trade price must be positive")

    return ticker_name, price


def write_observation(pair, payload):
    ticker_name, last_price = validate_ticker_payload(payload, pair)

    record = {
        "schema_version": 1,
        "observed_at_utc": utc_iso_timestamp(),
        "mode": "PUBLIC_MARKET_DATA_ONLY",
        "trading_permitted": False,
        "order_submission_attempted": False,
        "requested_pair": pair,
        "kraken_result_pair": ticker_name,
        "last_trade_price": last_price,
        "ticker": payload["result"][ticker_name],
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(DATA_DIR, 0o700)

    output_path = DATA_DIR / f"kraken_ticker_{pair}_{utc_timestamp()}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(output_path, flags, 0o600)

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")

    os.chmod(output_path, 0o600)
    return output_path, record


def main():
    try:
        pair = get_market_pair()
        payload = fetch_public_ticker(pair)
        output_path, record = write_observation(pair, payload)
    except Exception as exc:
        print(f"MARKET_OBSERVATION_FAILED: {exc}", file=sys.stderr)
        return 1

    print("MARKET_OBSERVATION_PASSED")
    print(f"pair={record['requested_pair']}")
    print(f"last_trade_price={record['last_trade_price']}")
    print(f"record={output_path.relative_to(PROJECT_ROOT)}")
    print("trading_permitted=false")
    print("order_submission_attempted=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
