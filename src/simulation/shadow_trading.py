"""Closed-candle shadow trading through the virtual simulation pipeline.

This module is intentionally broker-free. It accepts already-finalized market
candles from an injected source, evaluates Brawler deterministically, and
routes resulting decisions only through SimulationSession.

It does not import exchange clients, credentials, order submission modules, or
live-trading infrastructure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from src.simulation.session import SimulationSession
from src.simulation.strategy_adapter import SimulationStrategyAdapterResult
from src.strategies.brawler_strategy import (
    BrawlerEvaluation,
    BrawlerSettings,
    evaluate_brawler_strategy,
)
from src.strategies.ema_regime_strategy import Candle


class TradingMode(str, Enum):
    """Explicit execution mode for the shadow runner."""

    SHADOW = "SHADOW"


class ClosedCandleSource(Protocol):
    """Source of finalized, chronologically ordered market candles.

    Implementations must return only candles that are closed and immutable.
    The source is deliberately responsible only for market-data retrieval; it
    receives neither a session nor a strategy decision and cannot submit an
    order through this interface.
    """

    def fetch_closed_5m_candles(self) -> Sequence[Candle]:
        """Return finalized five-minute candles in ascending timestamp order."""

    def fetch_closed_1h_candles(self) -> Sequence[Candle]:
        """Return finalized hourly candles in ascending timestamp order."""


@dataclass(frozen=True, slots=True)
class ShadowTradingCheckpoint:
    """Durable identity of the latest consumed five-minute candle."""

    last_processed_5m_timestamp: str | None = None


class ShadowCheckpointStore(Protocol):
    """Minimal persistence boundary for shadow progress."""

    def load(self) -> ShadowTradingCheckpoint:
        """Return the current consumed-candle checkpoint."""

    def save(self, checkpoint: ShadowTradingCheckpoint) -> None:
        """Persist the new consumed-candle checkpoint."""


@dataclass(slots=True)
class InMemoryShadowCheckpointStore:
    """Simple deterministic checkpoint store for tests and local processes."""

    checkpoint: ShadowTradingCheckpoint = ShadowTradingCheckpoint()

    def load(self) -> ShadowTradingCheckpoint:
        return self.checkpoint

    def save(self, checkpoint: ShadowTradingCheckpoint) -> None:
        self.checkpoint = checkpoint


@dataclass(frozen=True, slots=True)
class ShadowTradingStep:
    """One closed-candle evaluation submitted to the simulation session."""

    candle_timestamp: str
    evaluation: BrawlerEvaluation
    result: SimulationStrategyAdapterResult


@dataclass(frozen=True, slots=True)
class ShadowTradingRunResult:
    """Outcome of one shadow polling cycle."""

    checkpoint_before: ShadowTradingCheckpoint
    checkpoint_after: ShadowTradingCheckpoint
    steps: tuple[ShadowTradingStep, ...]

    @property
    def processed_candle_count(self) -> int:
        return len(self.steps)

    @property
    def is_noop(self) -> bool:
        return not self.steps


class ShadowTradingRunner:
    """Consume unseen finalized candles through one virtual simulation session."""

    def __init__(
        self,
        *,
        mode: TradingMode,
        session: SimulationSession,
        candle_source: ClosedCandleSource,
        checkpoint_store: ShadowCheckpointStore,
        settings: BrawlerSettings | None = None,
    ) -> None:
        if mode is not TradingMode.SHADOW:
            raise ValueError("ShadowTradingRunner requires TradingMode.SHADOW")

        if not isinstance(session, SimulationSession):
            raise TypeError("session must be a SimulationSession")

        self._session = session
        self._candle_source = candle_source
        self._checkpoint_store = checkpoint_store
        self._settings = settings or BrawlerSettings()

    def run_once(self) -> ShadowTradingRunResult:
        """Evaluate every unseen finalized five-minute candle exactly once.

        The source should include sufficient prior history for Brawler
        indicators on every poll. Only candles strictly later than the durable
        checkpoint are submitted to the session.

        The checkpoint advances only after `process_brawler_evaluation`
        returns. Unexpected evaluation or session failures leave that candle
        uncheckpointed so a subsequent run can retry it.
        """

        candles_5m = tuple(self._candle_source.fetch_closed_5m_candles())
        candles_1h = tuple(self._candle_source.fetch_closed_1h_candles())

        _validate_chronological_candles(candles_5m, label="closed 5m candles")
        _validate_chronological_candles(candles_1h, label="closed 1h candles")

        checkpoint_before = self._checkpoint_store.load()
        checkpoint_timestamp = _parse_timestamp(
            checkpoint_before.last_processed_5m_timestamp
        )

        unseen_start_index = _first_unseen_index(
            candles_5m,
            checkpoint_timestamp=checkpoint_timestamp,
        )

        if unseen_start_index == len(candles_5m):
            return ShadowTradingRunResult(
                checkpoint_before=checkpoint_before,
                checkpoint_after=checkpoint_before,
                steps=(),
            )

        hourly_end_index = 0
        steps: list[ShadowTradingStep] = []
        checkpoint_after = checkpoint_before

        for index, candle_5m in enumerate(candles_5m):
            current_timestamp = _parse_timestamp(candle_5m.timestamp)

            while (
                hourly_end_index < len(candles_1h)
                and _parse_timestamp(candles_1h[hourly_end_index].timestamp)
                <= current_timestamp
            ):
                hourly_end_index += 1

            if index < unseen_start_index:
                continue

            evaluation = evaluate_brawler_strategy(
                candles_5m=candles_5m[: index + 1],
                candles_1h=candles_1h[:hourly_end_index],
                campaign=self._session.brawler_campaign,
                settings=self._settings,
            )

            if evaluation.decision.candle_timestamp is not None:
                result = self._session.process_brawler_evaluation(evaluation)
                steps.append(
                    ShadowTradingStep(
                        candle_timestamp=candle_5m.timestamp,
                        evaluation=evaluation,
                        result=result,
                    )
                )

            checkpoint_after = ShadowTradingCheckpoint(
                last_processed_5m_timestamp=candle_5m.timestamp
            )
            self._checkpoint_store.save(checkpoint_after)

        return ShadowTradingRunResult(
            checkpoint_before=checkpoint_before,
            checkpoint_after=checkpoint_after,
            steps=tuple(steps),
        )


def _first_unseen_index(
    candles: Sequence[Candle],
    *,
    checkpoint_timestamp: datetime | None,
) -> int:
    """Return the first five-minute candle strictly later than the checkpoint."""

    if checkpoint_timestamp is None:
        return 0

    for index, candle in enumerate(candles):
        if _parse_timestamp(candle.timestamp) > checkpoint_timestamp:
            return index

    return len(candles)


def _validate_chronological_candles(
    candles: Sequence[Candle],
    *,
    label: str,
) -> None:
    """Reject duplicate, naive, or reverse-ordered timestamps."""

    previous_timestamp: datetime | None = None

    for index, candle in enumerate(candles):
        timestamp = _parse_timestamp(candle.timestamp)

        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(
                f"{label} must have strictly increasing timestamps; "
                f"index {index} is not later than index {index - 1}."
            )

        previous_timestamp = timestamp


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp and normalize it to UTC."""

    if value is None:
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"Candle timestamp must be a valid ISO-8601 value: {value!r}"
        ) from error

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            f"Candle timestamp must include timezone information: {value!r}"
        )

    return parsed.astimezone(timezone.utc)
