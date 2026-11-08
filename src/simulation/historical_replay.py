"""Deterministic historical replay utilities for closed-candle strategies."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from datetime import datetime, timezone

from src.simulation.backtest_events import (
    BacktestEvent,
    build_backtest_event,
    capture_backtest_event_state,
)
from src.simulation.session import SimulationSession
from src.simulation.strategy_adapter import SimulationStrategyAdapterResult
from src.strategies.brawler_strategy import (
    BrawlerSettings,
    evaluate_brawler_strategy,
)
from src.strategies.ema_regime_strategy import Candle


def run_brawler_historical_replay(
    *,
    session: SimulationSession,
    candles_5m: Sequence[Candle],
    candles_1h: Sequence[Candle],
    settings: BrawlerSettings | None = None,
    start_index: int = 0,
    event_sink: MutableSequence[BacktestEvent] | None = None,
) -> tuple[SimulationStrategyAdapterResult, ...]:
    """Replay ordered, finalized 5-minute candles through Brawler.

    For every processed 5-minute candle, Brawler receives:

    * every finalized 5-minute candle through the current candle;
    * only 1-hour candles whose timestamp is no later than the current
      5-minute candle timestamp;
    * the campaign state currently committed by ``session``.

    ``start_index`` supports restart-safe continuation. The strategy still
    receives the complete prior 5-minute history needed for indicators, but
    only candles from ``start_index`` onward are evaluated and submitted to
    the simulation session.

    The session owns campaign-state persistence. Therefore, a rejected ENTER
    or VAULT does not incorrectly advance Brawler's campaign during replay.

    When ``event_sink`` is provided, one immutable audit event is appended
    for every evaluation submitted to the simulation session. Warm-up
    evaluations with no executable timestamp do not produce events.
    """

    _validate_chronological_candles(candles_5m, label="candles_5m")
    _validate_chronological_candles(candles_1h, label="candles_1h")
    _validate_start_index(start_index, candle_count=len(candles_5m))

    effective_settings = settings or BrawlerSettings()
    results: list[SimulationStrategyAdapterResult] = []

    # This starts at zero deliberately. When resuming at a later 5-minute
    # index, all hourly candles closed by that first resumed candle must still
    # be included in Brawler's available hourly history.
    hourly_end_index = 0

    for index in range(start_index, len(candles_5m)):
        candle_5m = candles_5m[index]
        current_timestamp = _parse_timestamp(candle_5m.timestamp)

        while (
            hourly_end_index < len(candles_1h)
            and _parse_timestamp(candles_1h[hourly_end_index].timestamp)
            <= current_timestamp
        ):
            hourly_end_index += 1

        evaluation = evaluate_brawler_strategy(
            candles_5m=candles_5m[: index + 1],
            candles_1h=candles_1h[:hourly_end_index],
            campaign=session.brawler_campaign,
            settings=effective_settings,
        )

        # Insufficient-history evaluations have no executable candle timestamp.
        # They must not enter the simulation adapter.
        if evaluation.decision.candle_timestamp is None:
            continue

        state_before = capture_backtest_event_state(session)
        result = session.process_brawler_evaluation(evaluation)
        state_after = capture_backtest_event_state(session)

        results.append(result)

        if event_sink is not None:
            event_sink.append(
                build_backtest_event(
                    sequence=len(event_sink) + 1,
                    evaluation=evaluation,
                    result=result,
                    state_before=state_before,
                    state_after=state_after,
                )
            )

    return tuple(results)


def _validate_start_index(
    start_index: int,
    *,
    candle_count: int,
) -> None:
    """Reject invalid replay continuation indexes."""

    if isinstance(start_index, bool) or not isinstance(start_index, int):
        raise TypeError("start_index must be an integer")

    if start_index < 0 or start_index > candle_count:
        raise ValueError(
            "start_index must be between 0 and the number of 5-minute candles"
        )


def _validate_chronological_candles(
    candles: Sequence[Candle],
    *,
    label: str,
) -> None:
    """Reject duplicate, naive, or reverse-ordered candle timestamps."""

    previous_timestamp: datetime | None = None

    for index, candle in enumerate(candles):
        timestamp = _parse_timestamp(candle.timestamp)

        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(
                f"{label} must have strictly increasing timestamps; "
                f"index {index} is not later than index {index - 1}."
            )

        previous_timestamp = timestamp


def _parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC."""

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
