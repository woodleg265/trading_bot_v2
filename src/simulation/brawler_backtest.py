"""End-to-end orchestration for deterministic Brawler backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data.historical_candles import load_candles_csv
from src.simulation.backtest_events import BacktestEvent
from src.simulation.historical_replay import run_brawler_historical_replay
from src.simulation.replay_report import ReplayReport, build_replay_report
from src.simulation.session import SimulationSession
from src.simulation.session_snapshot import SimulationSessionSnapshot
from src.strategies.brawler_strategy import BrawlerSettings


@dataclass(frozen=True, slots=True)
class BrawlerBacktestResult:
    """Durable final state, report, and audit ledger from a Brawler backtest."""

    report: ReplayReport
    final_snapshot: SimulationSessionSnapshot
    events: tuple[BacktestEvent, ...]


def run_brawler_backtest(
    *,
    candles_5m_path: str | Path,
    candles_1h_path: str | Path,
    session: SimulationSession,
    settings: BrawlerSettings | None = None,
    start_index: int = 0,
) -> BrawlerBacktestResult:
    """Load validated candles, replay Brawler, and return durable results.

    Historical CSV validation is delegated to the data boundary. Temporal
    correctness, warm-up handling, and anti-lookahead hourly-candle exposure
    are delegated to the historical replay runner.
    """

    if not isinstance(session, SimulationSession):
        raise TypeError("session must be a SimulationSession")

    candles_5m = load_candles_csv(Path(candles_5m_path))
    candles_1h = load_candles_csv(Path(candles_1h_path))
    events: list[BacktestEvent] = []

    run_brawler_historical_replay(
        session=session,
        candles_5m=candles_5m,
        candles_1h=candles_1h,
        settings=settings,
        start_index=start_index,
        event_sink=events,
    )

    return BrawlerBacktestResult(
        report=build_replay_report(session),
        final_snapshot=SimulationSessionSnapshot.capture(session),
        events=tuple(events),
    )
