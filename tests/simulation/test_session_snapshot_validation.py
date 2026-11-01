from __future__ import annotations

import json
from datetime import timedelta
from decimal import Decimal

import pytest

from src.simulation.fill_model import SimulationFillConfig
from src.simulation.risk_engine import SimulationRiskLimits
from src.simulation.session import SimulationSession, SimulationSessionConfig
from src.simulation.session_snapshot import SimulationSessionSnapshot
from src.simulation.strategy_adapter import SimulationStrategyAdapterConfig
from src.strategies.ema_regime_strategy import DecisionAction, StrategyDecision


@pytest.fixture
def config() -> SimulationSessionConfig:
    return SimulationSessionConfig(
        run_id="snapshot-validation-001",
        strategy_adapter=SimulationStrategyAdapterConfig(
            symbol="BTC/USD",
            entry_quantity=Decimal("2"),
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


def enter_decision() -> StrategyDecision:
    return StrategyDecision(
        action=DecisionAction.ENTER,
        reason="snapshot validation test",
        candle_timestamp="2026-10-16T14:00:00+00:00",
        close=100.0,
        rsi=None,
        ema_fast=None,
        ema_slow=None,
        ema_200=None,
        adx=None,
        plus_di=None,
        minus_di=None,
    )


def valid_snapshot_dict(config: SimulationSessionConfig) -> dict[str, object]:
    session = SimulationSession(config)
    session.process_decision(enter_decision())
    return SimulationSessionSnapshot.capture(session).to_dict()


def test_rejects_unsupported_snapshot_version(
    config: SimulationSessionConfig,
) -> None:
    payload = valid_snapshot_dict(config)
    payload["version"] = 999

    with pytest.raises(ValueError, match="version|unsupported|Unsupported"):
        SimulationSessionSnapshot.from_dict(payload)


def test_rejects_open_position_without_active_entry_price(
    config: SimulationSessionConfig,
) -> None:
    payload = valid_snapshot_dict(config)
    payload["active_entry_price"] = None

    with pytest.raises(ValueError, match="active_entry_price|active entry"):
        SimulationSessionSnapshot.from_dict(payload)


def test_rejects_open_position_without_active_entry_timestamp(
    config: SimulationSessionConfig,
) -> None:
    payload = valid_snapshot_dict(config)
    payload["active_entry_timestamp"] = None

    with pytest.raises(ValueError, match="active_entry_timestamp|active entry"):
        SimulationSessionSnapshot.from_dict(payload)


def test_rejects_malformed_snapshot_json() -> None:
    with pytest.raises(ValueError):
        SimulationSessionSnapshot.from_json("{not valid json")


def test_rejects_missing_required_snapshot_field(
    config: SimulationSessionConfig,
) -> None:
    payload = valid_snapshot_dict(config)
    del payload["ledger_state"]

    with pytest.raises((KeyError, TypeError, ValueError), match="ledger|field|missing"):
        SimulationSessionSnapshot.from_dict(payload)


def test_rejects_inconsistent_ledger_and_risk_positions(
    config: SimulationSessionConfig,
) -> None:
    payload = valid_snapshot_dict(config)

    # The exact nested representation may use strings for Decimal values.
    payload["risk_context"]["position_quantities"] = {"BTC": "0"}

    with pytest.raises(ValueError, match="position|ledger|risk|inconsistent"):
        SimulationSessionSnapshot.from_dict(payload)
