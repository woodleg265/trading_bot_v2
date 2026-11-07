from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

import src.simulation.historical_replay as historical_replay
from src.simulation.fill_model import SimulationFillConfig
from src.simulation.risk_engine import SimulationRiskLimits
from src.simulation.session import SimulationSession, SimulationSessionConfig
from src.simulation.session_snapshot import SimulationSessionSnapshot
from src.simulation.strategy_adapter import SimulationStrategyAdapterConfig
from src.strategies.brawler_strategy import (
    BrawlerCampaignState,
    BrawlerDecision,
    BrawlerEvaluation,
    BrawlerSettings,
)
from src.strategies.ema_regime_strategy import Candle, DecisionAction


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
def session_config() -> SimulationSessionConfig:
    return SimulationSessionConfig(
        run_id="historical-replay-001",
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


@pytest.fixture
def session(session_config: SimulationSessionConfig) -> SimulationSession:
    return SimulationSession(session_config)


@pytest.fixture
def replay_candles_5m() -> tuple[Candle, ...]:
    return tuple(
        make_candle(
            f"2026-01-01T{hour:02d}:{minute:02d}:00+00:00",
            close=100.0 + index,
        )
        for index, (hour, minute) in enumerate(
            (
                (0, 0),
                (0, 5),
                (0, 10),
                (0, 15),
                (0, 20),
                (0, 25),
                (0, 30),
                (0, 35),
                (0, 40),
                (0, 45),
                (0, 50),
                (0, 55),
                (1, 0),
                (1, 5),
                (1, 10),
                (1, 15),
            )
        )
    )


@pytest.fixture
def replay_candles_1h() -> tuple[Candle, ...]:
    return (
        make_candle("2026-01-01T00:00:00+00:00"),
        make_candle("2026-01-01T01:00:00+00:00"),
    )


def restore_session(session: SimulationSession) -> SimulationSession:
    snapshot = SimulationSessionSnapshot.capture(session)
    return SimulationSessionSnapshot.from_json(snapshot.to_json()).restore()


def hold_evaluation(
    candle: Candle,
    campaign: BrawlerCampaignState,
) -> BrawlerEvaluation:
    return BrawlerEvaluation(
        decision=BrawlerDecision(
            action=DecisionAction.HOLD,
            reason="test hold",
            candle_timestamp=candle.timestamp,
            close=candle.close,
            rsi_5m=50.0,
            rsi_1h=50.0,
            volume_ratio=1.0,
            entry_price=campaign.entry_price,
            rsi_peak=campaign.rsi_peak,
        ),
        next_campaign=campaign,
    )


def test_replay_exposes_only_closed_hourly_candles(
    monkeypatch: pytest.MonkeyPatch,
    session: SimulationSession,
) -> None:
    observed_hourly_windows: list[tuple[str, tuple[str, ...]]] = []

    def fake_evaluate_brawler_strategy(
        *,
        candles_5m: tuple[Candle, ...],
        candles_1h: tuple[Candle, ...],
        campaign: BrawlerCampaignState,
        settings: BrawlerSettings,
    ) -> BrawlerEvaluation:
        current_candle = candles_5m[-1]

        observed_hourly_windows.append(
            (
                current_candle.timestamp,
                tuple(candle.timestamp for candle in candles_1h),
            )
        )

        return hold_evaluation(current_candle, campaign)

    monkeypatch.setattr(
        historical_replay,
        "evaluate_brawler_strategy",
        fake_evaluate_brawler_strategy,
    )

    candles_5m = (
        make_candle("2026-01-01T00:00:00+00:00"),
        make_candle("2026-01-01T00:05:00+00:00"),
        make_candle("2026-01-01T00:55:00+00:00"),
        make_candle("2026-01-01T01:00:00+00:00"),
        make_candle("2026-01-01T01:05:00+00:00"),
    )
    candles_1h = (
        make_candle("2026-01-01T00:00:00+00:00"),
        make_candle("2026-01-01T01:00:00+00:00"),
    )

    results = historical_replay.run_brawler_historical_replay(
        session=session,
        candles_5m=candles_5m,
        candles_1h=candles_1h,
    )

    assert len(results) == 5

    assert observed_hourly_windows == [
        (
            "2026-01-01T00:00:00+00:00",
            ("2026-01-01T00:00:00+00:00",),
        ),
        (
            "2026-01-01T00:05:00+00:00",
            ("2026-01-01T00:00:00+00:00",),
        ),
        (
            "2026-01-01T00:55:00+00:00",
            ("2026-01-01T00:00:00+00:00",),
        ),
        (
            "2026-01-01T01:00:00+00:00",
            (
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T01:00:00+00:00",
            ),
        ),
        (
            "2026-01-01T01:05:00+00:00",
            (
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T01:00:00+00:00",
            ),
        ),
    ]


def test_replay_skips_non_executable_warmup_evaluations(
    monkeypatch: pytest.MonkeyPatch,
    session: SimulationSession,
) -> None:
    evaluated_timestamps: list[str] = []
    processed_timestamps: list[str | None] = []

    def fake_evaluate_brawler_strategy(
        *,
        candles_5m: tuple[Candle, ...],
        candles_1h: tuple[Candle, ...],
        campaign: BrawlerCampaignState,
        settings: BrawlerSettings,
    ) -> BrawlerEvaluation:
        current_candle = candles_5m[-1]
        evaluated_timestamps.append(current_candle.timestamp)

        evaluation = hold_evaluation(current_candle, campaign)

        if len(candles_5m) < 3:
            return BrawlerEvaluation(
                decision=BrawlerDecision(
                    action=DecisionAction.HOLD,
                    reason="insufficient history",
                    candle_timestamp=None,
                    close=current_candle.close,
                    rsi_5m=None,
                    rsi_1h=None,
                    volume_ratio=None,
                    entry_price=campaign.entry_price,
                    rsi_peak=campaign.rsi_peak,
                ),
                next_campaign=campaign,
            )

        return evaluation

    original_process = session.process_brawler_evaluation

    def spy_process(evaluation: BrawlerEvaluation):
        processed_timestamps.append(evaluation.decision.candle_timestamp)
        return original_process(evaluation)

    monkeypatch.setattr(
        historical_replay,
        "evaluate_brawler_strategy",
        fake_evaluate_brawler_strategy,
    )
    monkeypatch.setattr(
        session,
        "process_brawler_evaluation",
        spy_process,
    )

    candles_5m = (
        make_candle("2026-01-01T00:00:00+00:00"),
        make_candle("2026-01-01T00:05:00+00:00"),
        make_candle("2026-01-01T00:10:00+00:00"),
        make_candle("2026-01-01T00:15:00+00:00"),
    )
    candles_1h = (make_candle("2026-01-01T00:00:00+00:00"),)

    historical_replay.run_brawler_historical_replay(
        session=session,
        candles_5m=candles_5m,
        candles_1h=candles_1h,
    )

    assert evaluated_timestamps == [
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:05:00+00:00",
        "2026-01-01T00:10:00+00:00",
        "2026-01-01T00:15:00+00:00",
    ]
    assert processed_timestamps == [
        "2026-01-01T00:10:00+00:00",
        "2026-01-01T00:15:00+00:00",
    ]
    assert None not in processed_timestamps


@pytest.mark.parametrize(
    ("candles_5m", "candles_1h"),
    [
        (
            (
                make_candle("2026-01-01T00:00:00+00:00"),
                make_candle("2026-01-01T00:05:00+00:00"),
                make_candle("2026-01-01T00:05:00+00:00"),
            ),
            (make_candle("2026-01-01T00:00:00+00:00"),),
        ),
        (
            (
                make_candle("2026-01-01T00:05:00+00:00"),
                make_candle("2026-01-01T00:00:00+00:00"),
            ),
            (make_candle("2026-01-01T00:00:00+00:00"),),
        ),
        (
            (
                make_candle("2026-01-01T00:00:00+00:00"),
                make_candle("2026-01-01T00:05:00+00:00"),
            ),
            (
                make_candle("2026-01-01T01:00:00+00:00"),
                make_candle("2026-01-01T00:00:00+00:00"),
            ),
        ),
        (
            (
                make_candle("2026-01-01T00:00:00"),
                make_candle("2026-01-01T00:05:00"),
            ),
            (make_candle("2026-01-01T00:00:00+00:00"),),
        ),
    ],
    ids=[
        "duplicate_5m_timestamp",
        "descending_5m_timestamp",
        "descending_1h_timestamp",
        "naive_5m_timestamp",
    ],
)
def test_replay_rejects_non_monotonic_or_naive_candle_timestamps(
    session: SimulationSession,
    candles_5m: tuple[Candle, ...],
    candles_1h: tuple[Candle, ...],
) -> None:
    with pytest.raises(ValueError):
        historical_replay.run_brawler_historical_replay(
            session=session,
            candles_5m=candles_5m,
            candles_1h=candles_1h,
        )


@pytest.mark.parametrize("start_index", [-1, 999])
def test_replay_rejects_out_of_range_start_index(
    session: SimulationSession,
    replay_candles_5m: tuple[Candle, ...],
    replay_candles_1h: tuple[Candle, ...],
    start_index: int,
) -> None:
    with pytest.raises(ValueError, match="start_index"):
        historical_replay.run_brawler_historical_replay(
            session=session,
            candles_5m=replay_candles_5m,
            candles_1h=replay_candles_1h,
            start_index=start_index,
        )


def test_replay_snapshot_restore_matches_uninterrupted_replay(
    session_config: SimulationSessionConfig,
    replay_candles_5m: tuple[Candle, ...],
    replay_candles_1h: tuple[Candle, ...],
) -> None:
    split_index = len(replay_candles_5m) // 2

    uninterrupted = SimulationSession(session_config)
    interrupted = SimulationSession(session_config)

    uninterrupted_results = historical_replay.run_brawler_historical_replay(
        session=uninterrupted,
        candles_5m=replay_candles_5m,
        candles_1h=replay_candles_1h,
    )

    prefix_results = historical_replay.run_brawler_historical_replay(
        session=interrupted,
        candles_5m=replay_candles_5m[:split_index],
        candles_1h=replay_candles_1h,
    )

    restored = restore_session(interrupted)

    suffix_results = historical_replay.run_brawler_historical_replay(
        session=restored,
        candles_5m=replay_candles_5m,
        candles_1h=replay_candles_1h,
        start_index=split_index,
    )

    uninterrupted_snapshot = SimulationSessionSnapshot.capture(uninterrupted)
    restored_snapshot = SimulationSessionSnapshot.capture(restored)

    assert len(prefix_results) + len(suffix_results) == len(uninterrupted_results)
    assert restored_snapshot.to_dict() == uninterrupted_snapshot.to_dict()
    assert restored_snapshot.to_json() == uninterrupted_snapshot.to_json()
