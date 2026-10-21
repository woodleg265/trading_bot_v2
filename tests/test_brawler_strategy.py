import pytest

import src.strategies.brawler_strategy as brawler_module
from src.strategies.brawler_strategy import (
    BrawlerCampaignState,
    BrawlerSettings,
    evaluate_brawler_strategy,
)
from src.strategies.ema_regime_strategy import Candle, DecisionAction


def candle(index: int, close: float, volume: float = 100.0) -> Candle:
    return Candle(
        timestamp=f"2026-01-01T00:{index:02d}:00Z",
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=volume,
    )


def market_candles() -> tuple[list[Candle], list[Candle]]:
    candles_5m = [
        candle(0, 100.0),
        candle(1, 100.0),
        candle(2, 100.0),
        candle(3, 100.0),
        candle(4, 100.0),
        candle(5, 100.0, volume=150.0),
    ]
    candles_1h = [
        candle(10, 100.0),
        candle(11, 100.0),
        candle(12, 100.0),
    ]
    return candles_5m, candles_1h


def settings() -> BrawlerSettings:
    return BrawlerSettings(
        rsi_period=2,
        rsi_dip_lookback=2,
        volume_lookback=2,
        rsi_dip_threshold=55.0,
        rsi_rebound_threshold=60.0,
        macro_rsi_floor=50.0,
        volume_multiplier=1.2,
        profit_target_fraction=0.03,
        exit_rsi_threshold=75.0,
        rsi_peak_pullback=5.0,
        vault_drawdown_fraction=0.015,
        cooldown_minutes=15,
    )


def mock_rsi(monkeypatch, *, rsi_5m: list[float | None], rsi_1h: list[float | None]):
    monkeypatch.setattr(
        brawler_module,
        "rsi",
        lambda values, period: rsi_5m if len(values) == 6 else rsi_1h,
    )


def test_settings_reject_invalid_rsi_range():
    with pytest.raises(ValueError, match="rsi_dip_threshold"):
        BrawlerSettings(rsi_dip_threshold=60.0, rsi_rebound_threshold=60.0)


def test_campaign_requires_entry_price_and_timestamp_together():
    with pytest.raises(ValueError, match="both be present"):
        BrawlerCampaignState(entry_price=100.0)


def test_blocks_when_not_enough_5m_candles():
    candles_5m, candles_1h = market_candles()

    result = evaluate_brawler_strategy(
        candles_5m[:-2],
        candles_1h,
        settings=settings(),
    )

    assert result.decision.action == DecisionAction.BLOCKED
    assert result.decision.reason.startswith("INSUFFICIENT_5M_CLOSED_CANDLES")


def test_enters_on_dip_rebound_volume_and_macro_confirmation(monkeypatch):
    candles_5m, candles_1h = market_candles()
    mock_rsi(
        monkeypatch,
        rsi_5m=[None, None, None, 50.0, 54.0, 61.0],
        rsi_1h=[None, None, 65.0],
    )

    result = evaluate_brawler_strategy(candles_5m, candles_1h, settings=settings())

    assert result.decision.action == DecisionAction.ENTER
    assert result.next_campaign.entry_price == 100.0
    assert result.next_campaign.rsi_peak == 61.0


def test_does_not_enter_without_prior_rsi_dip(monkeypatch):
    candles_5m, candles_1h = market_candles()
    mock_rsi(
        monkeypatch,
        rsi_5m=[None, None, None, 56.0, 57.0, 61.0],
        rsi_1h=[None, None, 65.0],
    )

    result = evaluate_brawler_strategy(candles_5m, candles_1h, settings=settings())

    assert result.decision.action == DecisionAction.HOLD
    assert result.decision.reason == "HOLD_NO_RSI_DIP"


def test_does_not_enter_without_volume_confirmation(monkeypatch):
    candles_5m, candles_1h = market_candles()
    candles_5m[-1] = candle(5, 100.0, volume=100.0)
    mock_rsi(
        monkeypatch,
        rsi_5m=[None, None, None, 50.0, 54.0, 61.0],
        rsi_1h=[None, None, 65.0],
    )

    result = evaluate_brawler_strategy(candles_5m, candles_1h, settings=settings())

    assert result.decision.action == DecisionAction.HOLD
    assert result.decision.reason == "HOLD_INSUFFICIENT_VOLUME"


def test_active_campaign_updates_rsi_peak_while_holding(monkeypatch):
    candles_5m, candles_1h = market_candles()
    mock_rsi(
        monkeypatch,
        rsi_5m=[None, None, None, 50.0, 54.0, 72.0],
        rsi_1h=[None, None, 65.0],
    )
    campaign = BrawlerCampaignState(
        entry_price=100.0,
        entry_timestamp="2026-01-01T00:00:00Z",
        rsi_peak=68.0,
    )

    result = evaluate_brawler_strategy(
        candles_5m,
        candles_1h,
        campaign=campaign,
        settings=settings(),
    )

    assert result.decision.action == DecisionAction.HOLD
    assert result.next_campaign.rsi_peak == 72.0


def test_exits_at_profit_target_when_rsi_is_overbought(monkeypatch):
    candles_5m, candles_1h = market_candles()
    candles_5m[-1] = candle(5, 103.0, volume=150.0)
    mock_rsi(
        monkeypatch,
        rsi_5m=[None, None, None, 50.0, 54.0, 76.0],
        rsi_1h=[None, None, 65.0],
    )
    campaign = BrawlerCampaignState(
        entry_price=100.0,
        entry_timestamp="2026-01-01T00:00:00Z",
        rsi_peak=74.0,
    )

    result = evaluate_brawler_strategy(
        candles_5m,
        candles_1h,
        campaign=campaign,
        settings=settings(),
    )

    assert result.decision.action == DecisionAction.EXIT
    assert result.next_campaign == BrawlerCampaignState()


def test_exits_at_profit_target_after_rsi_peak_pullback(monkeypatch):
    candles_5m, candles_1h = market_candles()
    candles_5m[-1] = candle(5, 103.0, volume=150.0)
    mock_rsi(
        monkeypatch,
        rsi_5m=[None, None, None, 50.0, 54.0, 70.0],
        rsi_1h=[None, None, 65.0],
    )
    campaign = BrawlerCampaignState(
        entry_price=100.0,
        entry_timestamp="2026-01-01T00:00:00Z",
        rsi_peak=76.0,
    )

    result = evaluate_brawler_strategy(
        candles_5m,
        candles_1h,
        campaign=campaign,
        settings=settings(),
    )

    assert result.decision.action == DecisionAction.EXIT
    assert result.decision.reason == "EXIT_PROFIT_TARGET_AND_RSI_PEAK_PULLBACK"


def test_drawdown_vaults_without_an_exit(monkeypatch):
    candles_5m, candles_1h = market_candles()
    candles_5m[-1] = candle(5, 98.5, volume=150.0)
    mock_rsi(
        monkeypatch,
        rsi_5m=[None, None, None, 50.0, 54.0, 40.0],
        rsi_1h=[None, None, 65.0],
    )
    campaign = BrawlerCampaignState(
        entry_price=100.0,
        entry_timestamp="2026-01-01T00:00:00Z",
        rsi_peak=65.0,
    )

    result = evaluate_brawler_strategy(
        candles_5m,
        candles_1h,
        campaign=campaign,
        settings=settings(),
    )

    assert result.decision.action == DecisionAction.VAULT
    assert result.decision.action != DecisionAction.EXIT
    assert result.next_campaign.entry_price is None
    assert result.next_campaign.cooldown_until == "2026-01-01T00:20:00Z"


def test_active_cooldown_blocks_even_a_valid_entry(monkeypatch):
    candles_5m, candles_1h = market_candles()
    mock_rsi(
        monkeypatch,
        rsi_5m=[None, None, None, 50.0, 54.0, 61.0],
        rsi_1h=[None, None, 65.0],
    )
    campaign = BrawlerCampaignState(
        cooldown_until="2026-01-01T00:10:00Z"
    )

    result = evaluate_brawler_strategy(
        candles_5m,
        candles_1h,
        campaign=campaign,
        settings=settings(),
    )

    assert result.decision.action == DecisionAction.BLOCKED
    assert result.decision.reason == "BLOCKED_COOLDOWN_ACTIVE"


def test_evaluator_is_deterministic_and_does_not_modify_inputs(monkeypatch):
    candles_5m, candles_1h = market_candles()
    original_5m = list(candles_5m)
    original_1h = list(candles_1h)
    mock_rsi(
        monkeypatch,
        rsi_5m=[None, None, None, 50.0, 54.0, 61.0],
        rsi_1h=[None, None, 65.0],
    )

    first = evaluate_brawler_strategy(candles_5m, candles_1h, settings=settings())
    second = evaluate_brawler_strategy(candles_5m, candles_1h, settings=settings())

    assert first == second
    assert candles_5m == original_5m
    assert candles_1h == original_1h
