import json
import os
from datetime import datetime, timezone

import pytest

from src import observation_analysis


def write_observation(
    path,
    *,
    pair="XBTUSD",
    observed_at_utc="2026-10-10T00:00:00Z",
    price=82624.5,
):
    path.write_text(
        json.dumps(
            {
                "observed_at_utc": observed_at_utc,
                "requested_pair": pair,
                "kraken_result_pair": pair,
                "last_trade_price": price,
                "ticker": {},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_find_observation_files_returns_matching_regular_json_files(tmp_path):
    first = write_observation(tmp_path / "kraken_ticker_XBTUSD_one.json")
    second = write_observation(tmp_path / "kraken_ticker_ETHUSD_two.json")

    (tmp_path / "notes.txt").write_text("not an observation", encoding="utf-8")
    (tmp_path / "other.json").write_text("{}", encoding="utf-8")
    (tmp_path / "kraken_ticker_XBTUSD.txt").write_text(
        "not JSON",
        encoding="utf-8",
    )

    result = observation_analysis.find_observation_files(tmp_path)

    assert result == [second, first]


def test_find_observation_files_ignores_directories(tmp_path):
    directory = tmp_path / "kraken_ticker_directory.json"
    directory.mkdir()

    assert observation_analysis.find_observation_files(tmp_path) == []


def test_find_observation_files_ignores_symlinks(tmp_path):
    target = write_observation(tmp_path / "target.json")
    symlink = tmp_path / "kraken_ticker_link.json"

    try:
        symlink.symlink_to(target)
    except OSError:
        pytest.skip("Symlinks are unavailable in this environment")

    assert observation_analysis.find_observation_files(tmp_path) == []


def test_find_observation_files_returns_empty_list_for_missing_directory(tmp_path):
    missing_directory = tmp_path / "does_not_exist"

    assert observation_analysis.find_observation_files(missing_directory) == []


def test_load_observation_returns_normalized_record(tmp_path):
    path = write_observation(
        tmp_path / "kraken_ticker_XBTUSD_20261010T000000Z.json",
        pair="XBTUSD",
        observed_at_utc="2026-10-10T00:00:00Z",
        price="82624.5",
    )

    result = observation_analysis.load_observation(path)

    assert result["requested_pair"] == "XBTUSD"
    assert result["observed_at_utc"] == "2026-10-10T00:00:00Z"
    assert result["observed_at"] == datetime(2026, 10, 10, tzinfo=timezone.utc)
    assert result["last_trade_price"] == 82624.5


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("not json", "could not read valid JSON"),
        ("[]", "observation must be a JSON object"),
        ('{"requested_pair": ""}', "requested_pair"),
        ('{"requested_pair": "XBT-USD"}', "requested_pair"),
        (
            '{"requested_pair": "XBTUSD", "observed_at_utc": "invalid", '
            '"last_trade_price": 1}',
            "observed_at_utc",
        ),
        (
            '{"requested_pair": "XBTUSD", '
            '"observed_at_utc": "2026-10-10T00:00:00+00:00", '
            '"last_trade_price": 1}',
            "observed_at_utc",
        ),
        (
            '{"requested_pair": "XBTUSD", '
            '"observed_at_utc": "2026-10-10T00:00:00Z", '
            '"last_trade_price": 0}',
            "last_trade_price",
        ),
        (
            '{"requested_pair": "XBTUSD", '
            '"observed_at_utc": "2026-10-10T00:00:00Z", '
            '"last_trade_price": -1}',
            "last_trade_price",
        ),
        (
            '{"requested_pair": "XBTUSD", '
            '"observed_at_utc": "2026-10-10T00:00:00Z", '
            '"last_trade_price": true}',
            "last_trade_price",
        ),
    ],
)
def test_load_observation_rejects_invalid_records(tmp_path, contents, message):
    path = tmp_path / "kraken_ticker_invalid.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        observation_analysis.load_observation(path)


def test_analyze_observations_returns_empty_report_for_missing_directory(tmp_path):
    result = observation_analysis.analyze_observations(tmp_path / "missing")

    assert result == {
        "observation_count": 0,
        "skipped_file_count": 0,
        "pairs": {},
    }


def test_analyze_observations_summarizes_prices_by_pair(tmp_path):
    write_observation(
        tmp_path / "kraken_ticker_XBTUSD_first.json",
        pair="XBTUSD",
        observed_at_utc="2026-10-10T00:00:00Z",
        price=82000.0,
    )
    write_observation(
        tmp_path / "kraken_ticker_XBTUSD_latest.json",
        pair="XBTUSD",
        observed_at_utc="2026-10-10T02:00:00Z",
        price=83000.0,
    )
    write_observation(
        tmp_path / "kraken_ticker_ETHUSD_one.json",
        pair="ETHUSD",
        observed_at_utc="2026-10-10T01:00:00Z",
        price=2000.0,
    )

    result = observation_analysis.analyze_observations(tmp_path)

    assert result["observation_count"] == 3
    assert result["skipped_file_count"] == 0

    assert result["pairs"]["XBTUSD"] == {
        "count": 2,
        "earliest_observed_at_utc": "2026-10-10T00:00:00Z",
        "latest_observed_at_utc": "2026-10-10T02:00:00Z",
        "minimum_price": 82000.0,
        "maximum_price": 83000.0,
        "average_price": 82500.0,
        "latest_price": 83000.0,
    }

    assert result["pairs"]["ETHUSD"] == {
        "count": 1,
        "earliest_observed_at_utc": "2026-10-10T01:00:00Z",
        "latest_observed_at_utc": "2026-10-10T01:00:00Z",
        "minimum_price": 2000.0,
        "maximum_price": 2000.0,
        "average_price": 2000.0,
        "latest_price": 2000.0,
    }


def test_analyze_observations_skips_malformed_matching_files(tmp_path):
    write_observation(
        tmp_path / "kraken_ticker_XBTUSD_valid.json",
        price=82624.5,
    )

    invalid_json = tmp_path / "kraken_ticker_invalid_json.json"
    invalid_json.write_text("{not valid json", encoding="utf-8")

    invalid_price = tmp_path / "kraken_ticker_invalid_price.json"
    invalid_price.write_text(
        json.dumps(
            {
                "requested_pair": "ETHUSD",
                "observed_at_utc": "2026-10-10T00:00:00Z",
                "last_trade_price": -5,
            }
        ),
        encoding="utf-8",
    )

    result = observation_analysis.analyze_observations(tmp_path)

    assert result["observation_count"] == 1
    assert result["skipped_file_count"] == 2
    assert set(result["pairs"]) == {"XBTUSD"}


def test_analyze_observations_does_not_modify_files(tmp_path):
    path = write_observation(
        tmp_path / "kraken_ticker_XBTUSD_20261010T000000Z.json",
        price=82624.5,
    )

    before_contents = path.read_bytes()
    before_stat = path.stat()

    result = observation_analysis.analyze_observations(tmp_path)

    after_contents = path.read_bytes()
    after_stat = path.stat()

    assert result["observation_count"] == 1
    assert before_contents == after_contents
    assert before_stat.st_size == after_stat.st_size
    assert before_stat.st_mtime_ns == after_stat.st_mtime_ns
    assert os.listdir(tmp_path) == [path.name]

