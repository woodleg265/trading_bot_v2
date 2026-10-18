"""Pure, offline long-only EMA/RSI/ADX regime strategy.

This module intentionally has no exchange, credential, network, file-system,
or order-execution dependencies. It evaluates only the closed candles supplied
by its caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Sequence


class DecisionAction(str, Enum):
    ENTER = "ENTER"
    EXIT = "EXIT"
    HOLD = "HOLD"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class Candle:
    """A validated, already-closed OHLCV candle."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        values = (self.open, self.high, self.low, self.close, self.volume)

        if not all(isfinite(value) for value in values):
            raise ValueError("Candle values must be finite numbers.")

        if self.high < self.low:
            raise ValueError("Candle high cannot be lower than candle low.")

        if self.high < max(self.open, self.close):
            raise ValueError("Candle high must be at least open and close.")

        if self.low > min(self.open, self.close):
            raise ValueError("Candle low must be at most open and close.")

        if self.volume < 0:
            raise ValueError("Candle volume cannot be negative.")


@dataclass(frozen=True)
class StrategyDecision:
    """A decision and the facts used to produce it."""

    action: DecisionAction
    reason: str
    candle_timestamp: str | None
    close: float | None
    rsi: float | None
    ema_fast: float | None
    ema_slow: float | None
    ema_200: float | None
    adx: float | None
    plus_di: float | None
    minus_di: float | None


@dataclass(frozen=True)
class StrategySettings:
    """Configuration for the long-only closed-candle regime strategy."""

    fast_ema_period: int = 12
    slow_ema_period: int = 26
    long_ema_period: int = 200
    rsi_period: int = 14
    rsi_entry_floor: float = 50.0
    rsi_entry_ceiling: float = 65.0
    rsi_exit_threshold: float = 70.0
    adx_period: int = 14
    adx_entry_threshold: float = 20.0
    crash_adx_threshold: float = 25.0

    def __post_init__(self) -> None:
        periods = (
            self.fast_ema_period,
            self.slow_ema_period,
            self.long_ema_period,
            self.rsi_period,
            self.adx_period,
        )

        if any(period <= 0 for period in periods):
            raise ValueError("All indicator periods must be positive.")

        if self.fast_ema_period >= self.slow_ema_period:
            raise ValueError(
                "fast_ema_period must be less than slow_ema_period."
            )

        if self.slow_ema_period >= self.long_ema_period:
            raise ValueError(
                "slow_ema_period must be less than long_ema_period."
            )

        rsi_thresholds = (
            self.rsi_entry_floor,
            self.rsi_entry_ceiling,
            self.rsi_exit_threshold,
        )

        if any(not isfinite(value) for value in rsi_thresholds):
            raise ValueError("RSI thresholds must be finite.")

        if not 0.0 <= self.rsi_entry_floor < self.rsi_entry_ceiling <= 100.0:
            raise ValueError(
                "RSI entry thresholds must satisfy "
                "0 <= floor < ceiling <= 100."
            )

        if not 0.0 <= self.rsi_exit_threshold <= 100.0:
            raise ValueError("rsi_exit_threshold must be between 0 and 100.")

        if (
            not isfinite(self.adx_entry_threshold)
            or self.adx_entry_threshold < 0.0
        ):
            raise ValueError(
                "adx_entry_threshold must be finite and non-negative."
            )

        if (
            not isfinite(self.crash_adx_threshold)
            or self.crash_adx_threshold < 0.0
        ):
            raise ValueError(
                "crash_adx_threshold must be finite and non-negative."
            )


def ema(values: Sequence[float], period: int) -> list[float]:
    """Return EMA values using the first observation as the initial EMA."""

    if period <= 0:
        raise ValueError("EMA period must be positive.")

    if not values:
        return []

    result = [float(values[0])]
    multiplier = 2.0 / (period + 1.0)

    for value in values[1:]:
        result.append((float(value) - result[-1]) * multiplier + result[-1])

    return result


def rsi(values: Sequence[float], period: int = 14) -> list[float | None]:
    """Return Wilder RSI values.

    Values before enough price history exists are None. A sequence with no
    losses produces RSI 100; a sequence with no gains produces RSI 0; a flat
    sequence produces RSI 50.
    """

    if period <= 0:
        raise ValueError("RSI period must be positive.")

    result: list[float | None] = [None] * len(values)

    if len(values) <= period:
        return result

    gains: list[float] = []
    losses: list[float] = []

    for previous, current in zip(values, values[1:]):
        difference = float(current) - float(previous)
        gains.append(max(difference, 0.0))
        losses.append(max(-difference, 0.0))

    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period
    result[period] = _rsi_from_averages(average_gain, average_loss)

    for index in range(period + 1, len(values)):
        gain = gains[index - 1]
        loss = losses[index - 1]

        average_gain = ((average_gain * (period - 1)) + gain) / period
        average_loss = ((average_loss * (period - 1)) + loss) / period

        result[index] = _rsi_from_averages(average_gain, average_loss)

    return result


def _rsi_from_averages(average_gain: float, average_loss: float) -> float:
    if average_gain == 0.0 and average_loss == 0.0:
        return 50.0

    if average_loss == 0.0:
        return 100.0

    if average_gain == 0.0:
        return 0.0

    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def directional_movement(
    previous: Candle,
    current: Candle,
) -> tuple[float, float, float]:
    """Return true range, positive DM, and negative DM."""

    upward_move = current.high - previous.high
    downward_move = previous.low - current.low

    plus_dm = upward_move if upward_move > downward_move and upward_move > 0 else 0.0
    minus_dm = (
        downward_move
        if downward_move > upward_move and downward_move > 0
        else 0.0
    )

    true_range = max(
        current.high - current.low,
        abs(current.high - previous.close),
        abs(current.low - previous.close),
    )

    return true_range, plus_dm, minus_dm


def adx(
    candles: Sequence[Candle],
    period: int = 14,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Return ADX, +DI, and -DI using Wilder smoothing."""

    if period <= 0:
        raise ValueError("ADX period must be positive.")

    length = len(candles)
    adx_values: list[float | None] = [None] * length
    plus_di_values: list[float | None] = [None] * length
    minus_di_values: list[float | None] = [None] * length

    if length <= period:
        return adx_values, plus_di_values, minus_di_values

    true_ranges: list[float] = []
    plus_dms: list[float] = []
    minus_dms: list[float] = []

    for previous, current in zip(candles, candles[1:]):
        true_range, plus_dm, minus_dm = directional_movement(previous, current)
        true_ranges.append(true_range)
        plus_dms.append(plus_dm)
        minus_dms.append(minus_dm)

    smoothed_tr = sum(true_ranges[:period])
    smoothed_plus_dm = sum(plus_dms[:period])
    smoothed_minus_dm = sum(minus_dms[:period])

    dx_values: list[tuple[int, float]] = []

    for candle_index in range(period, length):
        if candle_index > period:
            source_index = candle_index - 1
            smoothed_tr = (
                smoothed_tr
                - (smoothed_tr / period)
                + true_ranges[source_index]
            )
            smoothed_plus_dm = (
                smoothed_plus_dm
                - (smoothed_plus_dm / period)
                + plus_dms[source_index]
            )
            smoothed_minus_dm = (
                smoothed_minus_dm
                - (smoothed_minus_dm / period)
                + minus_dms[source_index]
            )

        if smoothed_tr == 0.0:
            plus_di = 0.0
            minus_di = 0.0
        else:
            plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
            minus_di = 100.0 * smoothed_minus_dm / smoothed_tr

        plus_di_values[candle_index] = plus_di
        minus_di_values[candle_index] = minus_di

        denominator = plus_di + minus_di
        dx = (
            0.0
            if denominator == 0.0
            else 100.0 * abs(plus_di - minus_di) / denominator
        )
        dx_values.append((candle_index, dx))

        if len(dx_values) == period:
            adx_values[candle_index] = sum(item[1] for item in dx_values) / period
        elif len(dx_values) > period:
            previous_adx = adx_values[candle_index - 1]
            assert previous_adx is not None
            adx_values[candle_index] = (
                (previous_adx * (period - 1)) + dx
            ) / period

    return adx_values, plus_di_values, minus_di_values


def evaluate_ema_regime_strategy(
    candles: Sequence[Candle],
    settings: StrategySettings | None = None,
) -> StrategyDecision:
    """Evaluate the long-only EMA/RSI/ADX regime strategy.

    The caller is responsible for supplying chronologically ordered,
    deduplicated, closed candles only.
    """

    settings = settings or StrategySettings()

    required_candles = max(
        settings.long_ema_period + 1,
        settings.rsi_period + 2,
        (settings.adx_period * 2) + 1,
    )

    if len(candles) < required_candles:
        return StrategyDecision(
            action=DecisionAction.BLOCKED,
            reason=(
                "INSUFFICIENT_CLOSED_CANDLES: "
                f"need at least {required_candles}, got {len(candles)}"
            ),
            candle_timestamp=None,
            close=None,
            rsi=None,
            ema_fast=None,
            ema_slow=None,
            ema_200=None,
            adx=None,
            plus_di=None,
            minus_di=None,
        )

    closes = [candle.close for candle in candles]

    fast_values = ema(closes, settings.fast_ema_period)
    slow_values = ema(closes, settings.slow_ema_period)
    long_values = ema(closes, settings.long_ema_period)
    rsi_values = rsi(closes, settings.rsi_period)
    adx_values, plus_di_values, minus_di_values = adx(candles, settings.adx_period)

    current_index = len(candles) - 1
    previous_index = current_index - 1

    current_rsi = rsi_values[current_index]
    previous_rsi = rsi_values[previous_index]
    current_adx = adx_values[current_index]
    current_plus_di = plus_di_values[current_index]
    current_minus_di = minus_di_values[current_index]

    if None in (
        current_rsi,
        previous_rsi,
        current_adx,
        current_plus_di,
        current_minus_di,
    ):
        return StrategyDecision(
            action=DecisionAction.BLOCKED,
            reason="INDICATORS_NOT_READY",
            candle_timestamp=candles[current_index].timestamp,
            close=closes[current_index],
            rsi=current_rsi,
            ema_fast=fast_values[current_index],
            ema_slow=slow_values[current_index],
            ema_200=long_values[current_index],
            adx=current_adx,
            plus_di=current_plus_di,
            minus_di=current_minus_di,
        )

    fast_crossed_up = (
        fast_values[previous_index] <= slow_values[previous_index]
        and fast_values[current_index] > slow_values[current_index]
    )
    fast_crossed_down = (
        fast_values[previous_index] >= slow_values[previous_index]
        and fast_values[current_index] < slow_values[current_index]
    )

    close = closes[current_index]
    above_long_trend = close > long_values[current_index]
    below_long_trend = close < long_values[current_index]

    trend_is_bearish = current_minus_di > current_plus_di
    crash_regime = (
        current_adx >= settings.crash_adx_threshold
        and trend_is_bearish
    )
    entry_momentum_ok = (
        settings.rsi_entry_floor <= current_rsi < settings.rsi_entry_ceiling
    )
    entry_trend_ok = (
        current_adx >= settings.adx_entry_threshold
        and current_plus_di > current_minus_di
    )
    overbought_reversal = (
        previous_rsi > settings.rsi_exit_threshold
        and current_rsi <= settings.rsi_exit_threshold
    )

    if fast_crossed_down:
        return _decision(
            DecisionAction.EXIT,
            "EXIT_EMA_CROSS_DOWN",
            candles[current_index],
            current_rsi,
            fast_values[current_index],
            slow_values[current_index],
            long_values[current_index],
            current_adx,
            current_plus_di,
            current_minus_di,
        )

    if below_long_trend:
        return _decision(
            DecisionAction.EXIT,
            "EXIT_LONG_TREND_LOST",
            candles[current_index],
            current_rsi,
            fast_values[current_index],
            slow_values[current_index],
            long_values[current_index],
            current_adx,
            current_plus_di,
            current_minus_di,
        )

    if overbought_reversal:
        return _decision(
            DecisionAction.EXIT,
            "EXIT_RSI_OVERBOUGHT_REVERSAL",
            candles[current_index],
            current_rsi,
            fast_values[current_index],
            slow_values[current_index],
            long_values[current_index],
            current_adx,
            current_plus_di,
            current_minus_di,
        )

    if crash_regime:
        return _decision(
            DecisionAction.BLOCKED,
            "BLOCKED_BEARISH_HIGH_ADX_REGIME",
            candles[current_index],
            current_rsi,
            fast_values[current_index],
            slow_values[current_index],
            long_values[current_index],
            current_adx,
            current_plus_di,
            current_minus_di,
        )

    if (
        fast_crossed_up
        and above_long_trend
        and entry_momentum_ok
        and entry_trend_ok
    ):
        return _decision(
            DecisionAction.ENTER,
            "ENTER_BULLISH_EMA_CROSS_WITH_TREND_MOMENTUM_CONFIRMATION",
            candles[current_index],
            current_rsi,
            fast_values[current_index],
            slow_values[current_index],
            long_values[current_index],
            current_adx,
            current_plus_di,
            current_minus_di,
        )

    return _decision(
        DecisionAction.HOLD,
        "NO_ENTRY_OR_EXIT_CONDITION",
        candles[current_index],
        current_rsi,
        fast_values[current_index],
        slow_values[current_index],
        long_values[current_index],
        current_adx,
        current_plus_di,
        current_minus_di,
    )


def _decision(
    action: DecisionAction,
    reason: str,
    candle: Candle,
    current_rsi: float,
    current_fast_ema: float,
    current_slow_ema: float,
    current_long_ema: float,
    current_adx: float,
    current_plus_di: float,
    current_minus_di: float,
) -> StrategyDecision:
    """Build a decision that records the final closed-candle indicator values."""

    return StrategyDecision(
        action=action,
        reason=reason,
        candle_timestamp=candle.timestamp,
        close=candle.close,
        rsi=current_rsi,
        ema_fast=current_fast_ema,
        ema_slow=current_slow_ema,
        ema_200=current_long_ema,
        adx=current_adx,
        plus_di=current_plus_di,
        minus_di=current_minus_di,
    )
