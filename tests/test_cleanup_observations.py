import os
import time
from pathlib import Path

import pytest

from src.cleanup_observations import (
    cleanup_observations,
    find_expired_observations,
    is_observation_file,
)


SECONDS_PER_DAY = 24 * 60 * 60
REFERENCE_TIME = 2_000_000_000.0


def create_file(path: Path, modified_time: float) -> Path:
    path.write_text('{"result": {}}', encoding="utf-8")
    os.utime(path, (modified_time, modified_time))
    return path


def test_is_observation_file_accepts_matching_regular_file(tmp_path):
    path = create_file(
        tmp_path / "kraken_ticker_BTC_USD_20260101.json",
        REFERENCE_TIME,
    )

    assert is_observation_file(path) is True


@pytest.mark.parametrize(
    "filename",
    [
        "other_file.json",
        "kraken_ticker_BTC_USD.txt",
        "kraken_trade_BTC_USD.json",
        "kraken_ticker_.json.bak",
    ],
)
def test_is_observation_file_rejects_nonmatching_names(tmp_path, filename):
    path = create_file(tmp_path / filename, REFERENCE_TIME)

    assert is_observation_file(path) is False


def test_is_observation_file_rejects_directory(tmp_path):
    directory = tmp_path / "kraken_ticker_directory.json"
    directory.mkdir()

    assert is_observation_file(directory) is False


def test_find_expired_observations_returns_only_old_matching_files(tmp_path):
    old_file = create_file(
        tmp_path / "kraken_ticker_BTC_USD_old.json",
        REFERENCE_TIME - (31 * SECONDS_PER_DAY),
    )
    recent_file = create_file(
        tmp_path / "kraken_ticker_BTC_USD_recent.json",
        REFERENCE_TIME - (29 * SECONDS_PER_DAY),
    )
    unrelated_old_file = create_file(
        tmp_path / "notes.json",
        REFERENCE_TIME - (60 * SECONDS_PER_DAY),
    )

    result = find_expired_observations(
        tmp_path,
        days=30,
        now=REFERENCE_TIME,
    )

    assert result == [old_file]
    assert recent_file.exists()
    assert unrelated_old_file.exists()


def test_find_expired_observations_returns_empty_for_missing_directory(tmp_path):
    missing_directory = tmp_path / "does_not_exist"

    result = find_expired_observations(
        missing_directory,
        days=30,
        now=REFERENCE_TIME,
    )

    assert result == []


@pytest.mark.parametrize("days", [0, -1])
def test_find_expired_observations_rejects_invalid_retention_days(tmp_path, days):
    with pytest.raises(ValueError, match="at least 1"):
        find_expired_observations(tmp_path, days=days, now=REFERENCE_TIME)


def test_find_expired_observations_rejects_file_as_data_directory(tmp_path):
    data_file = create_file(tmp_path / "not_a_directory", REFERENCE_TIME)

    with pytest.raises(ValueError, match="not a directory"):
        find_expired_observations(data_file, days=30, now=REFERENCE_TIME)


def test_cleanup_dry_run_reports_file_without_deleting(tmp_path, capsys):
    old_file = create_file(
        tmp_path / "kraken_ticker_ETH_USD_old.json",
        REFERENCE_TIME - (31 * SECONDS_PER_DAY),
    )

    result = cleanup_observations(
        tmp_path,
        days=30,
        apply=False,
        now=REFERENCE_TIME,
    )

    output = capsys.readouterr().out

    assert result == [old_file]
    assert old_file.exists()
    assert "Would delete:" in output
    assert "Dry run only" in output


def test_cleanup_apply_deletes_only_expired_matching_file(tmp_path, capsys):
    old_file = create_file(
        tmp_path / "kraken_ticker_ETH_USD_old.json",
        REFERENCE_TIME - (31 * SECONDS_PER_DAY),
    )
    recent_file = create_file(
        tmp_path / "kraken_ticker_ETH_USD_recent.json",
        REFERENCE_TIME - (5 * SECONDS_PER_DAY),
    )
    unrelated_file = create_file(
        tmp_path / "important_notes.json",
        REFERENCE_TIME - (100 * SECONDS_PER_DAY),
    )

    result = cleanup_observations(
        tmp_path,
        days=30,
        apply=True,
        now=REFERENCE_TIME,
    )

    output = capsys.readouterr().out

    assert result == [old_file]
    assert not old_file.exists()
    assert recent_file.exists()
    assert unrelated_file.exists()
    assert "Deleting:" in output


def test_cleanup_reports_when_nothing_is_expired(tmp_path, capsys):
    result = cleanup_observations(
        tmp_path,
        days=30,
        apply=False,
        now=REFERENCE_TIME,
    )

    output = capsys.readouterr().out

    assert result == []
    assert "No expired market observation files found." in output


def test_symlink_is_not_selected_for_cleanup(tmp_path):
    target = create_file(
        tmp_path / "target.json",
        REFERENCE_TIME - (60 * SECONDS_PER_DAY),
    )
    symlink = tmp_path / "kraken_ticker_link.json"
    symlink.symlink_to(target)

    result = find_expired_observations(
        tmp_path,
        days=30,
        now=REFERENCE_TIME,
    )

    assert result == []
    assert symlink.exists()
    assert target.exists()
