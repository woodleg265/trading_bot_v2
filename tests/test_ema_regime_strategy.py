import pytest

from src.strategies.ema_regime_strategy import (
    Candle,
    DecisionAction,
    directional_movement,
    evaluate_ema_regime_strategy,
    rsi,
)


def candle(
    index: int,
    close: float,
    *,
    high: float | None = None,
    low: float | None = None,
) -> Candle:
    return Candle(
        timestamp=f"2026-01-01T00:{index:02d}:00Z",
        open=close,
        high=high if high is not None else close + 1.0,
        low=low if low is not None else close - 1.0,
        close=close,
        volume=100.0,
    )


def test_candle_rejects_invalid_ohlc_range():
    with pytest.raises(ValueError, match="high cannot be lower"):
        Candle(
            timestamp="2026-01-01T00:00:00Z",
            open=100.0,
            high=90.0,
            low=95.0,
            close=100.0,
            volume=1.0,
        )


def test_rsi_returns_none_without_sufficient_history():
    assert rsi([100.0] * 14, period=14) == [None] * 14


def test_rsi_is_100_when_there_are_no_losses():
    values = [float(value) for value in range(1, 20)]

    result = rsi(values, period=14)

    assert result[-1] == 100.0


def test_rsi_is_zero_when_there_are_no_gains():
    values = [float(value) for value in range(20, 1, -1)]

    result = rsi(values, period=14)

    assert result[-1] == 0.0


def test_rsi_is_50_for_a_flat_series():
    result = rsi([100.0] * 20, period=14)

    assert result[-1] == 50.0


def test_negative_directional_movement_uses_previous_low_minus_current_low():
    previous = candle(0, 100.0, high=105.0, low=95.0)
    current = candle(1, 98.0, high=104.0, low=90.0)

    true_range, plus_dm, minus_dm = directional_movement(previous, current)

    assert true_range == 14.0
    assert plus_dm == 0.0
    assert minus_dm == 5.0


def test_positive_directional_movement_wins_when_upward_move_is_larger():
    previous = candle(0, 100.0, high=105.0, low=95.0)
    current = candle(1, 104.0, high=110.0, low=93.0)

    _, plus_dm, minus_dm = directional_movement(previous, current)

    assert plus_dm == 5.0
    assert minus_dm == 0.0


def test_strategy_blocks_when_not_enough_closed_candles():
    candles = [candle(index, 100.0 + index) for index in range(200)]

    decision = evaluate_ema_regime_strategy(candles)

    assert decision.action == DecisionAction.BLOCKED
    assert decision.reason.startswith("INSUFFICIENT_CLOSED_CANDLES")


def test_strategy_is_deterministic_for_identical_candles():
    candles = [
        candle(
            index,
            100.0 + (index * 0.2),
            high=101.0 + (index * 0.2),
            low=99.0 + (index * 0.2),
        )
        for index in range(240)
    ]

    first = evaluate_ema_regime_strategy(candles)
    second = evaluate_ema_regime_strategy(candles)

    assert first == second
    assert first.action in {
        DecisionAction.ENTER,
        DecisionAction.EXIT,
        DecisionAction.HOLD,
        DecisionAction.BLOCKED,
    }
    assert first.candle_timestamp == candles[-1].timestamp


def test_strategy_does_not_modify_input_candles():
    candles = [candle(index, 100.0 + index) for index in range(240)]
    original = list(candles)

    evaluate_ema_regime_strategy(candles)

    assert candles == original
