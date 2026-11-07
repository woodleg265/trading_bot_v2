from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.strategies.ema_regime_strategy import Candle


REQUIRED_CANDLE_FIELDS = frozenset(
    {"timestamp", "open", "high", "low", "close", "volume"}
)


def load_candles_csv(path: Path) -> tuple[Candle, ...]:
    """Load and strictly validate OHLCV candles from a CSV file."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)

            if reader.fieldnames is None:
                raise ValueError(f"Candle CSV has no header row: {path}")

            missing_fields = REQUIRED_CANDLE_FIELDS.difference(reader.fieldnames)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(
                    f"Candle CSV is missing required columns: {missing}"
                )

            return candles_from_records(reader)

    except FileNotFoundError as exc:
        raise ValueError(f"Candle CSV file does not exist: {path}") from exc


def candles_from_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[Candle, ...]:
    """Convert external candle records into validated immutable candles."""
    candles = tuple(_candle_from_record(record) for record in records)

    if not candles:
        raise ValueError("At least one candle record is required")

    _validate_strictly_increasing_timestamps(candles)
    return candles


def _candle_from_record(record: Mapping[str, Any]) -> Candle:
    missing_fields = REQUIRED_CANDLE_FIELDS.difference(record)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Candle record is missing required fields: {missing}")

    timestamp = _parse_utc_timestamp(record["timestamp"])

    try:
        open_price = float(record["open"])
        high = float(record["high"])
        low = float(record["low"])
        close = float(record["close"])
        volume = float(record["volume"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Candle open, high, low, close, and volume must be numeric"
        ) from exc

    _validate_ohlcv(
        open_price=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )

    return Candle(
        timestamp=timestamp.isoformat().replace("+00:00", "Z"),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _parse_utc_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Candle timestamp must be a non-empty ISO-8601 string")

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"

    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Invalid candle timestamp: {value!r}"
        ) from exc

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(
            f"Candle timestamp must be timezone-aware UTC: {value!r}"
        )

    if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise ValueError(
            f"Candle timestamp must use UTC (+00:00 or Z): {value!r}"
        )

    return timestamp.astimezone(timezone.utc)


def _validate_ohlcv(
    *,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> None:
    if open_price <= 0:
        raise ValueError("Candle open must be greater than zero")
    if high <= 0:
        raise ValueError("Candle high must be greater than zero")
    if low <= 0:
        raise ValueError("Candle low must be greater than zero")
    if close <= 0:
        raise ValueError("Candle close must be greater than zero")
    if volume < 0:
        raise ValueError("Candle volume must be non-negative")

    if low > min(open_price, close):
        raise ValueError("Candle low cannot exceed open or close")

    if high < max(open_price, close):
        raise ValueError("Candle high cannot be below open or close")

    if low > high:
        raise ValueError("Candle low cannot exceed candle high")


def _validate_strictly_increasing_timestamps(
    candles: tuple[Candle, ...],
) -> None:
    previous_timestamp: datetime | None = None

    for candle in candles:
        timestamp = _parse_utc_timestamp(candle.timestamp)

        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(
                "Candle timestamps must be strictly increasing and unique"
            )

        previous_timestamp = timestamp
