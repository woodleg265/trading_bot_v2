from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from src.simulation.brawler_backtest import (
    BrawlerBacktestResult,
    run_brawler_backtest,
)
from src.simulation.fill_model import SimulationFillConfig
from src.simulation.risk_engine import SimulationRiskLimits
from src.simulation.session import SimulationSession, SimulationSessionConfig
from src.simulation.session_snapshot import SimulationSessionSnapshot
from src.simulation.strategy_adapter import SimulationStrategyAdapterConfig
from src.strategies.brawler_strategy import BrawlerSettings


FIXTURES_DIRECTORY = Path(__file__).resolve().parents[1] / "fixtures"
CANDLES_5M_PATH = FIXTURES_DIRECTORY / "btc_usd_5m_short.csv"
CANDLES_1H_PATH = FIXTURES_DIRECTORY / "btc_usd_1h_short.csv"


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


def _run_backtest(
    session_config: SimulationSessionConfig,
) -> tuple[SimulationSession, BrawlerBacktestResult]:
    session = SimulationSession(config=session_config)

    result = run_brawler_backtest(
        candles_5m_path=CANDLES_5M_PATH,
        candles_1h_path=CANDLES_1H_PATH,
        session=session,
        settings=BrawlerSettings(),
    )

    return session, result


def _report_values(report: object) -> dict[str, object]:
    assert is_dataclass(report)
    return asdict(report)


def test_backtest_returns_report_final_snapshot_and_performance(
    session_config: SimulationSessionConfig,
) -> None:
    session, result = _run_backtest(session_config)

    assert isinstance(result, BrawlerBacktestResult)
    assert result.final_snapshot == SimulationSessionSnapshot.capture(session)

    assert isinstance(result.events, tuple)
    assert result.performance.total_events == len(result.events)

    report_values = _report_values(result.report)

    # Confirm that the report is a real durable projection of the completed
    # session rather than an empty placeholder.
    assert any(
        value == Decimal("1000")
        for value in report_values.values()
    )
    assert any(
        value in (Decimal("0"), 0, (), [], None)
        for value in report_values.values()
    )


def test_backtest_report_projects_initial_cash_when_fixture_has_no_trade(
    session_config: SimulationSessionConfig,
) -> None:
    _, result = _run_backtest(session_config)
    report_values = _report_values(result.report)

    assert result.performance.total_events == len(result.events)

    cash_values = {
        field_name: value
        for field_name, value in report_values.items()
        if "cash" in field_name
    }

    assert cash_values
    assert all(value == Decimal("1000") for value in cash_values.values())

    quantity_values = {
        field_name: value
        for field_name, value in report_values.items()
        if "quantity" in field_name or "position" in field_name
    }

    assert quantity_values
    assert all(
        value in (Decimal("0"), 0, None, (), [])
        for value in quantity_values.values()
    )


def test_backtest_rejects_missing_five_minute_candle_file(
    session_config: SimulationSessionConfig,
    tmp_path: Path,
) -> None:
    session = SimulationSession(config=session_config)
    missing_path = tmp_path / "does_not_exist_5m.csv"

    with pytest.raises(
        ValueError,
        match=r"Candle CSV file does not exist",
    ):
        run_brawler_backtest(
            candles_5m_path=missing_path,
            candles_1h_path=CANDLES_1H_PATH,
            session=session,
            settings=BrawlerSettings(),
        )


def test_backtest_rejects_missing_hourly_candle_file(
    session_config: SimulationSessionConfig,
    tmp_path: Path,
) -> None:
    session = SimulationSession(config=session_config)
    missing_path = tmp_path / "does_not_exist_1h.csv"

    with pytest.raises(
        ValueError,
        match=r"Candle CSV file does not exist",
    ):
        run_brawler_backtest(
            candles_5m_path=CANDLES_5M_PATH,
            candles_1h_path=missing_path,
            session=session,
            settings=BrawlerSettings(),
        )


@pytest.mark.parametrize(
    ("filename", "contents"),
    [
        (
            "missing_required_column.csv",
            "timestamp,open,high,low,close\n"
            "2024-01-01T00:00:00+00:00,100,101,99,100\n",
        ),
        (
            "invalid_ohlcv.csv",
            "timestamp,open,high,low,close,volume\n"
            "2024-01-01T00:00:00+00:00,100,99,101,100,10\n",
        ),
    ],
)
def test_backtest_rejects_malformed_candle_csv(
    session_config: SimulationSessionConfig,
    tmp_path: Path,
    filename: str,
    contents: str,
) -> None:
    malformed_path = tmp_path / filename
    malformed_path.write_text(contents, encoding="utf-8")

    session = SimulationSession(config=session_config)

    with pytest.raises(ValueError):
        run_brawler_backtest(
            candles_5m_path=malformed_path,
            candles_1h_path=CANDLES_1H_PATH,
            session=session,
            settings=BrawlerSettings(),
        )


def test_backtest_is_deterministic_across_independent_sessions(
    session_config: SimulationSessionConfig,
) -> None:
    first_session, first_result = _run_backtest(session_config)
    second_session, second_result = _run_backtest(session_config)

    assert first_result.report == second_result.report
    assert first_result.final_snapshot == second_result.final_snapshot
    assert first_result.events == second_result.events
    assert first_result.performance == second_result.performance

    assert first_result.performance.total_events == len(first_result.events)
    assert second_result.performance.total_events == len(second_result.events)

    assert first_result.final_snapshot == SimulationSessionSnapshot.capture(
        first_session
    )
    assert second_result.final_snapshot == SimulationSessionSnapshot.capture(
        second_session
    )
