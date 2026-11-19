from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

import src.simulation.shadow_trading as shadow_trading
from src.simulation.fill_model import SimulationFillConfig
from src.simulation.risk_engine import SimulationRiskLimits
from src.simulation.session import SimulationSession, SimulationSessionConfig
from src.simulation.shadow_trading import (
    InMemoryShadowCheckpointStore,
    ShadowTradingCheckpoint,
    ShadowTradingRunner,
    TradingMode,
)
from src.simulation.strategy_adapter import (
    SimulationAdapterOutcome,
    SimulationStrategyAdapterConfig,
)
from src.strategies.brawler_strategy import (
    BrawlerCampaignState,
    BrawlerDecision,
    BrawlerEvaluation,
)
from src.strategies.ema_regime_strategy import Candle, DecisionAction


class StaticClosedCandleSource:
    """Deterministic closed-candle source used only by shadow tests."""

    def __init__(
        self,
        candles_5m: tuple[Candle, ...],
        candles_1h: tuple[Candle, ...],
    ) -> None:
        self._candles_5m = candles_5m
        self._candles_1h = candles_1h

    def fetch_closed_5m_candles(self) -> tuple[Candle, ...]:
        return self._candles_5m

    def fetch_closed_1h_candles(self) -> tuple[Candle, ...]:
        return self._candles_1h


def make_candle(timestamp: str, close: float = 100.0) -> Candle:
    return Candle(
        timestamp=timestamp,
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=100.0,
    )


@pytest.fixture
def session() -> SimulationSession:
    return SimulationSession(
        SimulationSessionConfig(
            run_id="shadow-run-001",
            strategy_adapter=SimulationStrategyAdapterConfig(
                symbol="BTC/USD",
                entry_quantity=Decimal("1"),
                max_exit_quantity=Decimal("1"),
            ),
            risk_limits=SimulationRiskLimits(
                allowed_symbols=frozenset({"BTC/USD"}),
                max_order_notional=Decimal("1000"),
                max_position_notional=Decimal("1000"),
                max_total_exposure=Decimal("1000"),
                max_market_data_age=timedelta(minutes=1),
                estimated_fee_rate=Decimal("0"),
            ),
            fill_config=SimulationFillConfig(
                fee_rate=Decimal("0"),
                slippage_rate=Decimal("0"),
            ),
            initial_cash_balance=Decimal("1000"),
            cash_currency="USD",
        )
    )


def make_evaluation(
    *,
    action: DecisionAction,
    candle: Candle,
    campaign: BrawlerCampaignState,
) -> BrawlerEvaluation:
    next_campaign = campaign

    if action is DecisionAction.ENTER:
        next_campaign = BrawlerCampaignState(
            entry_price=candle.close,
            entry_timestamp=candle.timestamp,
            rsi_peak=60.0,
        )

    return BrawlerEvaluation(
        decision=BrawlerDecision(
            action=action,
            reason="test decision",
            candle_timestamp=candle.timestamp,
            close=candle.close,
            rsi_5m=55.0,
            rsi_1h=55.0,
            volume_ratio=1.0,
            entry_price=campaign.entry_price,
            rsi_peak=campaign.rsi_peak,
        ),
        next_campaign=next_campaign,
    )


def test_shadow_runner_requires_explicit_shadow_mode(
    session: SimulationSession,
) -> None:
    source = StaticClosedCandleSource((), ())
    checkpoint_store = InMemoryShadowCheckpointStore()

    with pytest.raises(ValueError, match="requires TradingMode.SHADOW"):
        ShadowTradingRunner(
            mode="LIVE",  # type: ignore[arg-type]
            session=session,
            candle_source=source,
            checkpoint_store=checkpoint_store,
        )


def test_shadow_runner_processes_each_unseen_candle_once(
    monkeypatch: pytest.MonkeyPatch,
    session: SimulationSession,
) -> None:
    candles_5m = (
        make_candle("2026-10-16T14:00:00+00:00"),
        make_candle("2026-10-16T14:05:00+00:00"),
    )
    candles_1h = (make_candle("2026-10-16T14:00:00+00:00"),)

    source = StaticClosedCandleSource(candles_5m, candles_1h)
    checkpoint_store = InMemoryShadowCheckpointStore()
    evaluated_timestamps: list[str] = []

    def fake_evaluate_brawler_strategy(
        *,
        candles_5m: tuple[Candle, ...],
        candles_1h: tuple[Candle, ...],
        campaign: BrawlerCampaignState,
        settings: object,
    ) -> BrawlerEvaluation:
        current_candle = candles_5m[-1]
        evaluated_timestamps.append(current_candle.timestamp)

        return make_evaluation(
            action=DecisionAction.HOLD,
            candle=current_candle,
            campaign=campaign,
        )

    monkeypatch.setattr(
        shadow_trading,
        "evaluate_brawler_strategy",
        fake_evaluate_brawler_strategy,
    )

    runner = ShadowTradingRunner(
        mode=TradingMode.SHADOW,
        session=session,
        candle_source=source,
        checkpoint_store=checkpoint_store,
    )

    first_result = runner.run_once()
    second_result = runner.run_once()

    assert first_result.processed_candle_count == 2
    assert first_result.is_noop is False
    assert second_result.is_noop is True
    assert second_result.processed_candle_count == 0

    assert evaluated_timestamps == [
        "2026-10-16T14:00:00+00:00",
        "2026-10-16T14:05:00+00:00",
    ]

    assert checkpoint_store.load() == ShadowTradingCheckpoint(
        last_processed_5m_timestamp="2026-10-16T14:05:00+00:00"
    )


def test_shadow_runner_routes_enter_only_through_virtual_session(
    monkeypatch: pytest.MonkeyPatch,
    session: SimulationSession,
) -> None:
    candle_5m = make_candle("2026-10-16T14:00:00+00:00")
    source = StaticClosedCandleSource((candle_5m,), ())
    checkpoint_store = InMemoryShadowCheckpointStore()

    def fake_evaluate_brawler_strategy(
        *,
        candles_5m: tuple[Candle, ...],
        candles_1h: tuple[Candle, ...],
        campaign: BrawlerCampaignState,
        settings: object,
    ) -> BrawlerEvaluation:
        return make_evaluation(
            action=DecisionAction.ENTER,
            candle=candles_5m[-1],
            campaign=campaign,
        )

    monkeypatch.setattr(
        shadow_trading,
        "evaluate_brawler_strategy",
        fake_evaluate_brawler_strategy,
    )

    runner = ShadowTradingRunner(
        mode=TradingMode.SHADOW,
        session=session,
        candle_source=source,
        checkpoint_store=checkpoint_store,
    )

    result = runner.run_once()

    assert result.processed_candle_count == 1
    assert result.steps[0].result.outcome is SimulationAdapterOutcome.EXECUTED
    assert session.ledger_state.cash_balance == Decimal("900")
    assert session.ledger_state.position_quantities == {"BTC": Decimal("1")}
    assert session.brawler_campaign.is_active is True


def test_shadow_runner_exposes_only_hourly_candles_closed_by_current_five_minute_candle(
    monkeypatch: pytest.MonkeyPatch,
    session: SimulationSession,
) -> None:
    candles_5m = (
        make_candle("2026-10-16T14:00:00+00:00"),
        make_candle("2026-10-16T14:05:00+00:00"),
        make_candle("2026-10-16T15:00:00+00:00"),
    )
    candles_1h = (
        make_candle("2026-10-16T14:00:00+00:00"),
        make_candle("2026-10-16T15:00:00+00:00"),
    )

    source = StaticClosedCandleSource(candles_5m, candles_1h)
    checkpoint_store = InMemoryShadowCheckpointStore()
    observed_hourly_windows: list[tuple[str, tuple[str, ...]]] = []

    def fake_evaluate_brawler_strategy(
        *,
        candles_5m: tuple[Candle, ...],
        candles_1h: tuple[Candle, ...],
        campaign: BrawlerCampaignState,
        settings: object,
    ) -> BrawlerEvaluation:
        current_candle = candles_5m[-1]

        observed_hourly_windows.append(
            (
                current_candle.timestamp,
                tuple(candle.timestamp for candle in candles_1h),
            )
        )

        return make_evaluation(
            action=DecisionAction.HOLD,
            candle=current_candle,
            campaign=campaign,
        )

    monkeypatch.setattr(
        shadow_trading,
        "evaluate_brawler_strategy",
        fake_evaluate_brawler_strategy,
    )

    ShadowTradingRunner(
        mode=TradingMode.SHADOW,
        session=session,
        candle_source=source,
        checkpoint_store=checkpoint_store,
    ).run_once()

    assert observed_hourly_windows == [
        (
            "2026-10-16T14:00:00+00:00",
            ("2026-10-16T14:00:00+00:00",),
        ),
        (
            "2026-10-16T14:05:00+00:00",
            ("2026-10-16T14:00:00+00:00",),
        ),
        (
            "2026-10-16T15:00:00+00:00",
            (
                "2026-10-16T14:00:00+00:00",
                "2026-10-16T15:00:00+00:00",
            ),
        ),
    ]


def test_shadow_runner_rejects_duplicate_or_reverse_ordered_candles(
    session: SimulationSession,
) -> None:
    source = StaticClosedCandleSource(
        candles_5m=(
            make_candle("2026-10-16T14:00:00+00:00"),
            make_candle("2026-10-16T14:00:00+00:00"),
        ),
        candles_1h=(),
    )

    runner = ShadowTradingRunner(
        mode=TradingMode.SHADOW,
        session=session,
        candle_source=source,
        checkpoint_store=InMemoryShadowCheckpointStore(),
    )

    with pytest.raises(ValueError, match="strictly increasing timestamps"):
        runner.run_once()
