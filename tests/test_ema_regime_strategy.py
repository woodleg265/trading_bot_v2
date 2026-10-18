import pytest

import src.strategies.ema_regime_strategy as strategy_module
from src.strategies.ema_regime_strategy import (
    Candle,
    DecisionAction,
    StrategySettings,
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


def candles(count: int = 240, close: float = 100.0) -> list[Candle]:
    return [candle(index, close + (index * 0.1)) for index in range(count)]


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


def test_settings_require_fast_ema_to_be_shorter_than_slow_ema():
    with pytest.raises(ValueError, match="fast_ema_period"):
        StrategySettings(fast_ema_period=26, slow_ema_period=26)


def test_settings_require_slow_ema_to_be_shorter_than_long_ema():
    with pytest.raises(ValueError, match="slow_ema_period"):
        StrategySettings(slow_ema_period=200, long_ema_period=200)


def test_settings_require_valid_rsi_entry_range():
    with pytest.raises(ValueError, match="RSI entry thresholds"):
        StrategySettings(rsi_entry_floor=65.0, rsi_entry_ceiling=65.0)


def test_strategy_blocks_when_not_enough_closed_candles():
    insufficient_candles = [candle(index, 100.0 + index) for index in range(200)]

    decision = evaluate_ema_regime_strategy(insufficient_candles)

    assert decision.action == DecisionAction.BLOCKED
    assert decision.reason.startswith("INSUFFICIENT_CLOSED_CANDLES")


def test_strategy_is_deterministic_for_identical_candles():
    market_candles = [
        candle(
            index,
            100.0 + (index * 0.2),
            high=101.0 + (index * 0.2),
            low=99.0 + (index * 0.2),
        )
        for index in range(240)
    ]

    first = evaluate_ema_regime_strategy(market_candles)
    second = evaluate_ema_regime_strategy(market_candles)

    assert first == second
    assert first.candle_timestamp == market_candles[-1].timestamp


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


def test_strategy_does_not_modify_input_candles():
    market_candles = candles()
    original = list(market_candles)

    evaluate_ema_regime_strategy(market_candles)

    assert market_candles == original


def mock_ema_values(
    monkeypatch,
    settings: StrategySettings,
    *,
    previous_fast: float,
    current_fast: float,
    previous_slow: float,
    current_slow: float,
    previous_long: float,
    current_long: float,
) -> None:
    values_by_period = {
        settings.fast_ema_period: (previous_fast, current_fast),
        settings.slow_ema_period: (previous_slow, current_slow),
        settings.long_ema_period: (previous_long, current_long),
    }

    monkeypatch.setattr(
        strategy_module,
        "ema",
        lambda values, period: (
            [None] * (len(values) - 2)
            + [
                values_by_period[period][0],
                values_by_period[period][1],
            ]
        ),
    )


def mock_rsi_values(
    monkeypatch,
    *,
    previous_rsi: float,
    current_rsi: float,
) -> None:
    monkeypatch.setattr(
        strategy_module,
        "rsi",
        lambda values, period: (
            [None] * (len(values) - 2)
            + [
                previous_rsi,
                current_rsi,
            ]
        ),
    )


def mock_adx_values(
    monkeypatch,
    *,
    current_adx: float,
    current_plus_di: float,
    current_minus_di: float,
) -> None:
    monkeypatch.setattr(
        strategy_module,
        "adx",
        lambda market_data, period: (
            [None] * (len(market_data) - 1) + [current_adx],
            [None] * (len(market_data) - 1) + [current_plus_di],
            [None] * (len(market_data) - 1) + [current_minus_di],
        ),
    )


def test_strategy_enters_in_confirmed_bullish_regime(monkeypatch):
    settings = StrategySettings()
    market_candles = candles()

    mock_ema_values(
        monkeypatch,
        settings,
        previous_fast=100.0,
        current_fast=110.0,
        previous_slow=105.0,
        current_slow=105.0,
        previous_long=99.0,
        current_long=100.0,
    )
    mock_rsi_values(monkeypatch, previous_rsi=54.0, current_rsi=55.0)
    mock_adx_values(
        monkeypatch,
        current_adx=30.0,
        current_plus_di=25.0,
        current_minus_di=10.0,
    )

    decision = evaluate_ema_regime_strategy(market_candles, settings)

    assert decision.action == DecisionAction.ENTER


def test_strategy_exits_on_bearish_ema_regime(monkeypatch):
    settings = StrategySettings()
    market_candles = candles()

    mock_ema_values(
        monkeypatch,
        settings,
        previous_fast=110.0,
        current_fast=95.0,
        previous_slow=105.0,
        current_slow=105.0,
        previous_long=90.0,
        current_long=90.0,
    )
    mock_rsi_values(monkeypatch, previous_rsi=55.0, current_rsi=55.0)
    mock_adx_values(
        monkeypatch,
        current_adx=20.0,
        current_plus_di=20.0,
        current_minus_di=10.0,
    )

    decision = evaluate_ema_regime_strategy(market_candles, settings)

    assert decision.action == DecisionAction.EXIT


def test_strategy_exits_when_price_falls_below_long_ema(monkeypatch):
    settings = StrategySettings()
    market_candles = candles(close=100.0)

    mock_ema_values(
        monkeypatch,
        settings,
        previous_fast=100.0,
        current_fast=110.0,
        previous_slow=105.0,
        current_slow=105.0,
        previous_long=99.0,
        current_long=150.0,
    )
    mock_rsi_values(monkeypatch, previous_rsi=55.0, current_rsi=55.0)
    mock_adx_values(
        monkeypatch,
        current_adx=20.0,
        current_plus_di=20.0,
        current_minus_di=10.0,
    )

    decision = evaluate_ema_regime_strategy(market_candles, settings)

    assert decision.action == DecisionAction.EXIT


def test_strategy_exits_on_rsi_overbought_reversal(monkeypatch):
    settings = StrategySettings()
    market_candles = candles()

    mock_ema_values(
        monkeypatch,
        settings,
        previous_fast=100.0,
        current_fast=110.0,
        previous_slow=105.0,
        current_slow=105.0,
        previous_long=99.0,
        current_long=100.0,
    )
    mock_rsi_values(
        monkeypatch,
        previous_rsi=settings.rsi_exit_threshold + 5.0,
        current_rsi=settings.rsi_exit_threshold,
    )
    mock_adx_values(
        monkeypatch,
        current_adx=20.0,
        current_plus_di=20.0,
        current_minus_di=10.0,
    )

    decision = evaluate_ema_regime_strategy(market_candles, settings)

    assert decision.action == DecisionAction.EXIT


def test_strategy_blocks_during_bearish_high_adx_crash_regime(monkeypatch):
    settings = StrategySettings()
    market_candles = candles()

    mock_ema_values(
        monkeypatch,
        settings,
        previous_fast=100.0,
        current_fast=110.0,
        previous_slow=105.0,
        current_slow=105.0,
        previous_long=99.0,
        current_long=100.0,
    )
    mock_rsi_values(monkeypatch, previous_rsi=54.0, current_rsi=55.0)
    mock_adx_values(
        monkeypatch,
        current_adx=settings.crash_adx_threshold + 1.0,
        current_plus_di=10.0,
        current_minus_di=30.0,
    )

    decision = evaluate_ema_regime_strategy(market_candles, settings)

    assert decision.action == DecisionAction.BLOCKED
