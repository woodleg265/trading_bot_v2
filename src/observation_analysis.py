"""Read-only analysis of locally stored Kraken market observations.

This module reads JSON files from data/market_observations and produces
historical price summaries. It never makes network requests and never changes
observation files.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATA_DIR = Path("data/market_observations")
FILE_PREFIX = "kraken_ticker_"
FILE_SUFFIX = ".json"


def find_observation_files(data_dir=DEFAULT_DATA_DIR):
    """Return sorted regular observation JSON files, excluding symlinks."""
    directory = Path(data_dir)

    if not directory.is_dir():
        return []

    files = []

    for path in directory.iterdir():
        if path.is_symlink():
            continue

        if not path.is_file():
            continue

        if not path.name.startswith(FILE_PREFIX):
            continue

        if not path.name.endswith(FILE_SUFFIX):
            continue

        files.append(path)

    return sorted(files)


def parse_utc_timestamp(value):
    """Validate and return a UTC ISO-8601 timestamp ending in Z."""
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("observed_at_utc must be a UTC timestamp ending in Z")

    try:
        timestamp = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("observed_at_utc is invalid") from exc

    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise ValueError("observed_at_utc must be UTC")

    return timestamp


def load_observation(path):
    """Load, validate, and normalize one local observation file."""
    try:
        with Path(path).open(encoding="utf-8") as file_handle:
            record = json.load(file_handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read valid JSON: {exc}") from exc

    if not isinstance(record, dict):
        raise ValueError("observation must be a JSON object")

    pair = record.get("requested_pair")
    if not isinstance(pair, str) or not pair or not pair.isalnum():
        raise ValueError("requested_pair must be a non-empty alphanumeric string")

    observed_at_utc = record.get("observed_at_utc")
    observed_at = parse_utc_timestamp(observed_at_utc)

    price = record.get("last_trade_price")
    if isinstance(price, bool):
        raise ValueError("last_trade_price must be a positive number")

    try:
        price = float(price)
    except (TypeError, ValueError) as exc:
        raise ValueError("last_trade_price must be a positive number") from exc

    if not math.isfinite(price) or price <= 0:
        raise ValueError("last_trade_price must be a positive finite number")

    return {
        "requested_pair": pair,
        "observed_at_utc": observed_at_utc,
        "observed_at": observed_at,
        "last_trade_price": price,
    }


def analyze_observations(data_dir=DEFAULT_DATA_DIR):
    """Analyze local observation files without modifying them."""
    grouped_records = {}
    skipped_file_count = 0

    for path in find_observation_files(data_dir):
        try:
            record = load_observation(path)
        except ValueError:
            skipped_file_count += 1
            continue

        grouped_records.setdefault(record["requested_pair"], []).append(record)

    pairs = {}

    for pair, records in sorted(grouped_records.items()):
        records.sort(key=lambda record: record["observed_at"])
        prices = [record["last_trade_price"] for record in records]
        latest_record = records[-1]

        pairs[pair] = {
            "count": len(records),
            "earliest_observed_at_utc": records[0]["observed_at_utc"],
            "latest_observed_at_utc": latest_record["observed_at_utc"],
            "minimum_price": min(prices),
            "maximum_price": max(prices),
            "average_price": sum(prices) / len(prices),
            "latest_price": latest_record["last_trade_price"],
        }

    return {
        "observation_count": sum(summary["count"] for summary in pairs.values()),
        "skipped_file_count": skipped_file_count,
        "pairs": pairs,
    }


def main():
    """Print a JSON report without changing any files."""
    parser = argparse.ArgumentParser(
        description="Analyze local Kraken market observation files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing observation JSON files.",
    )
    arguments = parser.parse_args()

    report = analyze_observations(arguments.data_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
