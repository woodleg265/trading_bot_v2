"""Pure, offline long-only Brawler mean-reversion strategy.

This module intentionally has no exchange, credential, network, file-system,
ledger, or order-execution dependencies. It evaluates only closed candles
supplied by its caller and returns an explicit decision plus next campaign
state.

A VAULT decision is not a sell. It ends active Brawler campaign management,
starts a cooldown, and leaves inventory handling to a later adapter layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Sequence

from src.strategies.ema_regime_strategy import Candle, DecisionAction, rsi


@dataclass(frozen=True)
class BrawlerSettings:
    """Validated configuration for the long-only Brawler strategy."""

    rsi_period: int = 14
    rsi_dip_lookback: int = 3
    volume_lookback: int = 3
    rsi_dip_threshold: float = 55.0
    rsi_rebound_threshold: float = 60.0
    macro_rsi_floor: float = 0.0
    volume_multiplier: float = 1.25
    profit_target_fraction: float = 0.03
    exit_rsi_threshold: float = 75.0
    rsi_peak_pullback: float = 5.0
    vault_drawdown_fraction: float = 0.015
    cooldown_minutes: int = 15

    def __post_init__(self) -> None:
        integer_values = (
            self.rsi_period,
            self.rsi_dip_lookback,
            self.volume_lookback,
            self.cooldown_minutes,
        )

        if any(not isinstance(value, int) or value <= 0 for value in integer_values):
            raise ValueError(
                "RSI period, lookbacks, and cooldown minutes must be "
                "positive integers."
            )

        rsi_values = (
            self.rsi_dip_threshold,
            self.rsi_rebound_threshold,
            self.macro_rsi_floor,
            self.exit_rsi_threshold,
        )

        if any(not isfinite(value) or not 0.0 <= value <= 100.0 for value in rsi_values):
            raise ValueError("RSI thresholds must be finite values from 0 to 100.")

        if self.rsi_dip_threshold >= self.rsi_rebound_threshold:
            raise ValueError(
                "rsi_dip_threshold must be less than rsi_rebound_threshold."
            )

        positive_values = (
            self.volume_multiplier,
            self.profit_target_fraction,
            self.rsi_peak_pullback,
            self.vault_drawdown_fraction,
        )

        if any(not isfinite(value) or value <= 0.0 for value in positive_values):
            raise ValueError(
                "Volume multiplier, profit target, RSI pullback, and vault "
                "drawdown must be finite values greater than zero."
            )


@dataclass(frozen=True)
class BrawlerCampaignState:
    """Persistent strategy state for one Brawler campaign."""

    entry_price: float | None = None
    entry_timestamp: str | None = None
    rsi_peak: float | None = None
    cooldown_until: str | None = None

    def __post_init__(self) -> None:
        has_entry_price = self.entry_price is not None
        has_entry_timestamp = self.entry_timestamp is not None

        if has_entry_price != has_entry_timestamp:
            raise ValueError(
                "entry_price and entry_timestamp must either both be present "
                "or both be absent."
            )

        if self.entry_price is not None:
            if not isfinite(self.entry_price) or self.entry_price <= 0.0:
                raise ValueError(
                    "entry_price must be a finite value greater than zero."
                )
            _parse_timestamp(self.entry_timestamp)

        if self.rsi_peak is not None:
            if not isfinite(self.rsi_peak) or not 0.0 <= self.rsi_peak <= 100.0:
                raise ValueError("rsi_peak must be a finite value from 0 to 100.")

        if self.cooldown_until is not None:
            _parse_timestamp(self.cooldown_until)

        if self.is_active and self.cooldown_until is not None:
            raise ValueError(
                "An active Brawler campaign cannot have an active cooldown."
            )

    @property
    def is_active(self) -> bool:
        return self.entry_price is not None


@dataclass(frozen=True)
class BrawlerDecision:
    """A Brawler decision and the market facts used to produce it."""

    action: DecisionAction
    reason: str
    candle_timestamp: str | None
    close: float | None
    rsi_5m: float | None
    rsi_1h: float | None
    volume_ratio: float | None
    entry_price: float | None
    rsi_peak: float | None


@dataclass(frozen=True)
class BrawlerEvaluation:
    """A decision together with the immutable next campaign state."""

    decision: BrawlerDecision
    next_campaign: BrawlerCampaignState


def evaluate_brawler_strategy(
    candles_5m: Sequence[Candle],
    candles_1h: Sequence[Candle],
    campaign: BrawlerCampaignState | None = None,
    settings: BrawlerSettings | None = None,
) -> BrawlerEvaluation:
    """Evaluate Brawler using finalized 5-minute and 1-hour candles only."""

    effective_settings = settings or BrawlerSettings()
    current_campaign = campaign or BrawlerCampaignState()

    minimum_5m_candles = (
        effective_settings.rsi_period
        + max(
            effective_settings.rsi_dip_lookback,
            effective_settings.volume_lookback,
        )
        + 1
    )
    minimum_1h_candles = effective_settings.rsi_period + 1

    if len(candles_5m) < minimum_5m_candles:
        return _blocked(
            reason=(
                "INSUFFICIENT_5M_CLOSED_CANDLES:"
                f"need={minimum_5m_candles},have={len(candles_5m)}"
            ),
            campaign=current_campaign,
        )

    if len(candles_1h) < minimum_1h_candles:
        return _blocked(
            reason=(
                "INSUFFICIENT_1H_CLOSED_CANDLES:"
                f"need={minimum_1h_candles},have={len(candles_1h)}"
            ),
            campaign=current_campaign,
        )

    latest_candle = candles_5m[-1]
    latest_timestamp = _parse_timestamp(latest_candle.timestamp)

    if current_campaign.cooldown_until is not None:
        cooldown_until = _parse_timestamp(current_campaign.cooldown_until)

        if latest_timestamp <= cooldown_until:
            return _evaluation(
                action=DecisionAction.BLOCKED,
                reason="BLOCKED_COOLDOWN_ACTIVE",
                candle=latest_candle,
                campaign=current_campaign,
            )

        current_campaign = BrawlerCampaignState()

    closes_5m = [candle.close for candle in candles_5m]
    closes_1h = [candle.close for candle in candles_1h]

    rsi_5m_values = rsi(closes_5m, effective_settings.rsi_period)
    rsi_1h_values = rsi(closes_1h, effective_settings.rsi_period)

    current_rsi_5m = rsi_5m_values[-1]
    current_rsi_1h = rsi_1h_values[-1]

    if current_rsi_5m is None or current_rsi_1h is None:
        return _blocked(
            reason="INSUFFICIENT_RSI_HISTORY",
            campaign=current_campaign,
            candle=latest_candle,
        )

    previous_volumes = [
        candle.volume
        for candle in candles_5m[-(effective_settings.volume_lookback + 1):-1]
    ]
    average_volume = sum(previous_volumes) / len(previous_volumes)
    volume_ratio = (
        latest_candle.volume / average_volume
        if average_volume > 0.0
        else None
    )

    if current_campaign.is_active:
        updated_peak = max(current_campaign.rsi_peak or current_rsi_5m, current_rsi_5m)
        entry_price = current_campaign.entry_price
        assert entry_price is not None

        drawdown_price = entry_price * (
            1.0 - effective_settings.vault_drawdown_fraction
        )

        if latest_candle.close <= drawdown_price:
            cooldown_until = latest_timestamp + timedelta(
                minutes=effective_settings.cooldown_minutes
            )
            next_campaign = BrawlerCampaignState(
                cooldown_until=_format_timestamp(cooldown_until)
            )
            return _evaluation(
                action=DecisionAction.VAULT,
                reason="VAULT_DRAWNDOWN_TRIGGERED",
                candle=latest_candle,
                campaign=current_campaign,
                rsi_5m=current_rsi_5m,
                rsi_1h=current_rsi_1h,
                volume_ratio=volume_ratio,
                rsi_peak=updated_peak,
                next_campaign=next_campaign,
            )

        profit_fraction = (latest_candle.close - entry_price) / entry_price
        is_overbought_exit = current_rsi_5m >= effective_settings.exit_rsi_threshold
        is_peak_pullback_exit = (
            updated_peak >= effective_settings.exit_rsi_threshold
            and current_rsi_5m
            <= updated_peak - effective_settings.rsi_peak_pullback
        )

        if profit_fraction >= effective_settings.profit_target_fraction:
            if is_overbought_exit:
                return _evaluation(
                    action=DecisionAction.EXIT,
                    reason="EXIT_PROFIT_TARGET_AND_OVERBOUGHT_RSI",
                    candle=latest_candle,
                    campaign=current_campaign,
                    rsi_5m=current_rsi_5m,
                    rsi_1h=current_rsi_1h,
                    volume_ratio=volume_ratio,
                    rsi_peak=updated_peak,
                    next_campaign=BrawlerCampaignState(),
                )

            if is_peak_pullback_exit:
                return _evaluation(
                    action=DecisionAction.EXIT,
                    reason="EXIT_PROFIT_TARGET_AND_RSI_PEAK_PULLBACK",
                    candle=latest_candle,
                    campaign=current_campaign,
                    rsi_5m=current_rsi_5m,
                    rsi_1h=current_rsi_1h,
                    volume_ratio=volume_ratio,
                    rsi_peak=updated_peak,
                    next_campaign=BrawlerCampaignState(),
                )

        next_campaign = BrawlerCampaignState(
            entry_price=entry_price,
            entry_timestamp=current_campaign.entry_timestamp,
            rsi_peak=updated_peak,
        )
        return _evaluation(
            action=DecisionAction.HOLD,
            reason="HOLD_ACTIVE_CAMPAIGN",
            candle=latest_candle,
            campaign=current_campaign,
            rsi_5m=current_rsi_5m,
            rsi_1h=current_rsi_1h,
            volume_ratio=volume_ratio,
            rsi_peak=updated_peak,
            next_campaign=next_campaign,
        )

    historical_rsi_values = rsi_5m_values[
        -(effective_settings.rsi_dip_lookback + 1):-1
    ]
    saw_rsi_dip = any(
        value is not None and value <= effective_settings.rsi_dip_threshold
        for value in historical_rsi_values
    )

    if not saw_rsi_dip:
        return _evaluation(
            action=DecisionAction.HOLD,
            reason="HOLD_NO_RSI_DIP",
            candle=latest_candle,
            campaign=current_campaign,
            rsi_5m=current_rsi_5m,
            rsi_1h=current_rsi_1h,
            volume_ratio=volume_ratio,
        )

    if current_rsi_5m < effective_settings.rsi_rebound_threshold:
        return _evaluation(
            action=DecisionAction.HOLD,
            reason="HOLD_NO_RSI_REBOUND",
            candle=latest_candle,
            campaign=current_campaign,
            rsi_5m=current_rsi_5m,
            rsi_1h=current_rsi_1h,
            volume_ratio=volume_ratio,
        )

    if volume_ratio is None or volume_ratio < effective_settings.volume_multiplier:
        return _evaluation(
            action=DecisionAction.HOLD,
            reason="HOLD_INSUFFICIENT_VOLUME",
            candle=latest_candle,
            campaign=current_campaign,
            rsi_5m=current_rsi_5m,
            rsi_1h=current_rsi_1h,
            volume_ratio=volume_ratio,
        )

    if current_rsi_1h < effective_settings.macro_rsi_floor:
        return _evaluation(
            action=DecisionAction.HOLD,
            reason="HOLD_MACRO_RSI_FILTER",
            candle=latest_candle,
            campaign=current_campaign,
            rsi_5m=current_rsi_5m,
            rsi_1h=current_rsi_1h,
            volume_ratio=volume_ratio,
        )

    next_campaign = BrawlerCampaignState(
        entry_price=latest_candle.close,
        entry_timestamp=latest_candle.timestamp,
        rsi_peak=current_rsi_5m,
    )
    return _evaluation(
        action=DecisionAction.ENTER,
        reason="ENTER_RSI_REBOUND_WITH_VOLUME_CONFIRMATION",
        candle=latest_candle,
        campaign=current_campaign,
        rsi_5m=current_rsi_5m,
        rsi_1h=current_rsi_1h,
        volume_ratio=volume_ratio,
        rsi_peak=current_rsi_5m,
        next_campaign=next_campaign,
    )


def _blocked(
    *,
    reason: str,
    campaign: BrawlerCampaignState,
    candle: Candle | None = None,
) -> BrawlerEvaluation:
    return _evaluation(
        action=DecisionAction.BLOCKED,
        reason=reason,
        candle=candle,
        campaign=campaign,
    )


def _evaluation(
    *,
    action: DecisionAction,
    reason: str,
    candle: Candle | None,
    campaign: BrawlerCampaignState,
    rsi_5m: float | None = None,
    rsi_1h: float | None = None,
    volume_ratio: float | None = None,
    rsi_peak: float | None = None,
    next_campaign: BrawlerCampaignState | None = None,
) -> BrawlerEvaluation:
    return BrawlerEvaluation(
        decision=BrawlerDecision(
            action=action,
            reason=reason,
            candle_timestamp=candle.timestamp if candle is not None else None,
            close=candle.close if candle is not None else None,
            rsi_5m=rsi_5m,
            rsi_1h=rsi_1h,
            volume_ratio=volume_ratio,
            entry_price=campaign.entry_price,
            rsi_peak=rsi_peak,
        ),
        next_campaign=next_campaign or campaign,
    )


def _parse_timestamp(value: str | None) -> datetime:
    if value is None:
        raise ValueError("Timestamp cannot be None.")

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Timestamp must be a valid ISO-8601 value.") from error

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp must include timezone information.")

    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
