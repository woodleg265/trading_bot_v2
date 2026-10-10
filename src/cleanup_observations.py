"""Safely remove old Kraken market-observation files.

This module is local-data-only:
- no network access
- no credential loading
- no trading functionality
- dry-run by default
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


DEFAULT_DATA_DIRECTORY = Path("data/market_observations")
FILE_PREFIX = "kraken_ticker_"
FILE_SUFFIX = ".json"


def is_observation_file(path: Path) -> bool:
    """Return True only for regular matching observation JSON files."""
    return (
        not path.is_symlink()
        and path.is_file()
        and path.name.startswith(FILE_PREFIX)
        and path.name.endswith(FILE_SUFFIX)
    )


def find_expired_observations(
    data_directory: Path,
    days: int,
    now: float | None = None,
) -> list[Path]:
    """Find matching regular files older than the retention period."""
    if days < 1:
        raise ValueError("Retention days must be at least 1.")

    if not data_directory.exists():
        return []

    if not data_directory.is_dir():
        raise ValueError(f"Data path is not a directory: {data_directory}")

    current_time = time.time() if now is None else now
    cutoff = current_time - (days * 24 * 60 * 60)

    expired = []

    for path in data_directory.iterdir():
        if is_observation_file(path) and path.stat().st_mtime < cutoff:
            expired.append(path)

    return sorted(expired)


def cleanup_observations(
    data_directory: Path,
    days: int,
    apply: bool = False,
    now: float | None = None,
) -> list[Path]:
    """List expired files, or delete them only with apply=True."""
    expired = find_expired_observations(data_directory, days, now=now)

    if not expired:
        print("No expired market observation files found.")
        return []

    action = "Deleting" if apply else "Would delete"

    for path in expired:
        print(f"{action}: {path}")
        if apply:
            path.unlink()

    if not apply:
        print("Dry run only: no files were deleted.")

    return expired


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Safely clean up old Kraken market observation files."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Retention period in whole days; minimum 1; default 30.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete expired files; otherwise only report them.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line cleanup utility."""
    arguments = parse_arguments()

    if arguments.days < 1:
        print("Error: --days must be at least 1.")
        return 2

    try:
        cleanup_observations(
            DEFAULT_DATA_DIRECTORY,
            arguments.days,
            apply=arguments.apply,
        )
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
