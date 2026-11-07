from __future__ import annotations

from pathlib import Path

import pytest

from src.data.historical_candles import candles_from_records, load_candles_csv


def candle_record(
    *,
    timestamp: str = "2025-01-01T00:00:00Z",
    open: str = "100.0",
    high: str = "105.0",
    low: str = "95.0",
    close: str = "102.0",
    volume: str = "12.5",
) -> dict[str, str]:
    return {
        "timestamp": timestamp,
        "open": open,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def test_candles_from_records_returns_immutable_validated_candles() -> None:
    candles = candles_from_records(
        [
            candle_record(timestamp="2025-01-01T00:00:00Z"),
            candle_record(timestamp="2025-01-01T00:05:00Z", close="103.0"),
        ]
    )

    assert isinstance(candles, tuple)
    assert len(candles) == 2
    assert candles[0].timestamp == "2025-01-01T00:00:00Z"
    assert candles[0].open == 100.0
    assert candles[1].close == 103.0


def test_candles_from_records_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="At least one candle"):
        candles_from_records([])


@pytest.mark.parametrize(
    ("mutations", "error"),
    [
        (
            {"timestamp": "2025-01-01T00:00:00"},
            "timezone-aware UTC",
        ),
        (
            {"timestamp": "2025-01-01T01:00:00+01:00"},
            "must use UTC",
        ),
        (
            {"timestamp": "not-a-timestamp"},
            "Invalid candle timestamp",
        ),
        (
            {"open": "0"},
            "open must be greater than zero",
        ),
        (
            {"volume": "-0.01"},
            "volume must be non-negative",
        ),
        (
            {"low": "103.0"},
            "low cannot exceed open or close",
        ),
        (
            {"high": "101.0"},
            "high cannot be below open or close",
        ),
        (
            {"close": "not-a-number"},
            "must be numeric",
        ),
    ],
)
def test_candles_from_records_rejects_invalid_candle_values(
    mutations: dict[str, str],
    error: str,
) -> None:
    record = candle_record()
    record.update(mutations)

    with pytest.raises(ValueError, match=error):
        candles_from_records([record])


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
        ["2025-01-01T00:05:00Z", "2025-01-01T00:00:00Z"],
    ],
)
def test_candles_from_records_rejects_non_increasing_timestamps(
    timestamps: list[str],
) -> None:
    records = [candle_record(timestamp=timestamp) for timestamp in timestamps]

    with pytest.raises(ValueError, match="strictly increasing"):
        candles_from_records(records)


def test_candles_from_records_rejects_missing_required_field() -> None:
    record = candle_record()
    del record["volume"]

    with pytest.raises(ValueError, match="missing required fields: volume"):
        candles_from_records([record])


def test_load_candles_csv_loads_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "candles.csv"
    path.write_text(
        "\n".join(
            [
                "timestamp,open,high,low,close,volume",
                "2025-01-01T00:00:00Z,100,105,95,102,12.5",
                "2025-01-01T00:05:00Z,102,106,100,104,15.0",
            ]
        ),
        encoding="utf-8",
    )

    candles = load_candles_csv(path)

    assert len(candles) == 2
    assert candles[0].close == 102.0
    assert candles[1].timestamp == "2025-01-01T00:05:00Z"


def test_load_candles_csv_rejects_missing_required_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candles.csv"
    path.write_text(
        "\n".join(
            [
                "timestamp,open,high,low,close",
                "2025-01-01T00:00:00Z,100,105,95,102",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns: volume"):
        load_candles_csv(path)


def test_load_candles_csv_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_candles_csv(tmp_path / "missing.csv")
