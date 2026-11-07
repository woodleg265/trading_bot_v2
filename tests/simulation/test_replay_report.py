from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from src.simulation.fill_model import SimulationFillConfig
from src.simulation.replay_report import build_replay_report
from src.simulation.risk_engine import SimulationRiskLimits
from src.simulation.session import SimulationSession, SimulationSessionConfig
from src.simulation.session_snapshot import SimulationSessionSnapshot
from src.simulation.strategy_adapter import (
    SimulationAdapterOutcome,
    SimulationStrategyAdapterConfig,
)
from src.strategies.brawler_strategy import (
    BrawlerCampaignState,
    BrawlerDecision,
    BrawlerEvaluation,
)
from src.strategies.ema_regime_strategy import DecisionAction


def brawler_evaluation(
    action: DecisionAction,
    *,
    timestamp: str = "2026-10-16T14:00:00+00:00",
    close: float = 100.0,
    next_campaign: BrawlerCampaignState,
) -> BrawlerEvaluation:
    return BrawlerEvaluation(
        decision=BrawlerDecision(
            action=action,
            reason="replay report test",
            candle_timestamp=timestamp,
            close=close,
            rsi_5m=60.0,
            rsi_1h=55.0,
            volume_ratio=1.5,
            entry_price=None,
            rsi_peak=None,
        ),
        next_campaign=next_campaign,
    )


def restore(session: SimulationSession) -> SimulationSession:
    snapshot = SimulationSessionSnapshot.capture(session)
    return SimulationSessionSnapshot.from_json(snapshot.to_json()).restore()


@pytest.fixture
def session_config() -> SimulationSessionConfig:
    return SimulationSessionConfig(
        run_id="replay-report-001",
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


def test_empty_session_report_projects_durable_initial_state(
    session: SimulationSession,
) -> None:
    before = SimulationSessionSnapshot.capture(session).to_dict()

    report = build_replay_report(session)

    assert report.initial_cash_balance == Decimal("1000")
    assert report.cash_balance == Decimal("1000")
    assert report.active_quantity == Decimal("0")
    assert report.vaulted_lot_count == 0
    assert report.vaulted_quantity == Decimal("0")
    assert report.campaign_active is False
    assert report.cooldown_until is None

    assert SimulationSessionSnapshot.capture(session).to_dict() == before


def test_report_is_deterministic_and_does_not_mutate_session(
    session: SimulationSession,
) -> None:
    before = SimulationSessionSnapshot.capture(session).to_dict()

    first = build_replay_report(session)
    second = build_replay_report(session)

    assert first == second
    assert SimulationSessionSnapshot.capture(session).to_dict() == before


def test_risk_rejected_enter_does_not_create_active_report_state(
    session: SimulationSession,
) -> None:
    proposed_campaign = BrawlerCampaignState(
        entry_price=1001.0,
        entry_timestamp="2026-10-16T14:00:00Z",
        rsi_peak=60.0,
    )

    result = session.process_brawler_evaluation(
        brawler_evaluation(
            DecisionAction.ENTER,
            close=1001.0,
            next_campaign=proposed_campaign,
        )
    )

    assert result.outcome is SimulationAdapterOutcome.REJECTED

    report = build_replay_report(session)

    assert report.initial_cash_balance == Decimal("1000")
    assert report.cash_balance == Decimal("1000")
    assert report.active_quantity == Decimal("0")
    assert report.vaulted_lot_count == 0
    assert report.vaulted_quantity == Decimal("0")
    assert report.campaign_active is False
    assert report.cooldown_until is None


def test_executed_enter_and_exit_report_active_inventory_and_final_cash(
    session: SimulationSession,
) -> None:
    active_campaign = BrawlerCampaignState(
        entry_price=100.0,
        entry_timestamp="2026-10-16T14:00:00Z",
        rsi_peak=60.0,
    )

    enter_result = session.process_brawler_evaluation(
        brawler_evaluation(
            DecisionAction.ENTER,
            close=100.0,
            next_campaign=active_campaign,
        )
    )

    assert enter_result.outcome is SimulationAdapterOutcome.EXECUTED

    active_report = build_replay_report(session)

    assert active_report.initial_cash_balance == Decimal("1000")
    assert active_report.cash_balance == Decimal("900")
    assert active_report.active_quantity == Decimal("1")
    assert active_report.vaulted_lot_count == 0
    assert active_report.vaulted_quantity == Decimal("0")
    assert active_report.campaign_active is True
    assert active_report.cooldown_until is None

    exit_result = session.process_brawler_evaluation(
        brawler_evaluation(
            DecisionAction.EXIT,
            timestamp="2026-10-16T14:05:00+00:00",
            close=110.0,
            next_campaign=BrawlerCampaignState(),
        )
    )

    assert exit_result.outcome is SimulationAdapterOutcome.EXECUTED

    final_report = build_replay_report(session)

    assert final_report.initial_cash_balance == Decimal("1000")
    assert final_report.cash_balance == Decimal("1010")
    assert final_report.active_quantity == Decimal("0")
    assert final_report.vaulted_lot_count == 0
    assert final_report.vaulted_quantity == Decimal("0")
    assert final_report.campaign_active is False
    assert final_report.cooldown_until is None


def test_executed_vault_report_distinguishes_archived_inventory(
    session: SimulationSession,
) -> None:
    active_campaign = BrawlerCampaignState(
        entry_price=100.0,
        entry_timestamp="2026-10-16T14:00:00Z",
        rsi_peak=60.0,
    )
    cooldown_campaign = BrawlerCampaignState(
        cooldown_until="2026-10-16T14:15:00Z",
    )

    enter_result = session.process_brawler_evaluation(
        brawler_evaluation(
            DecisionAction.ENTER,
            close=100.0,
            next_campaign=active_campaign,
        )
    )

    assert enter_result.outcome is SimulationAdapterOutcome.EXECUTED

    vault_result = session.process_brawler_evaluation(
        brawler_evaluation(
            DecisionAction.VAULT,
            timestamp="2026-10-16T14:05:00+00:00",
            close=98.0,
            next_campaign=cooldown_campaign,
        )
    )

    assert vault_result.outcome is SimulationAdapterOutcome.EXECUTED

    report = build_replay_report(session)

    assert report.initial_cash_balance == Decimal("1000")
    assert report.cash_balance == Decimal("900")
    assert report.active_quantity == Decimal("0")
    assert report.vaulted_lot_count == 1
    assert report.vaulted_quantity == Decimal("1")
    assert report.campaign_active is False
    assert report.cooldown_until == "2026-10-16T14:15:00Z"


def test_report_projects_cooldown_only_campaign(
    session_config: SimulationSessionConfig,
) -> None:
    session = SimulationSession(
        session_config,
        initial_brawler_campaign=BrawlerCampaignState(
            cooldown_until="2026-10-16T14:15:00Z",
        ),
    )

    report = build_replay_report(session)

    assert report.cash_balance == Decimal("1000")
    assert report.active_quantity == Decimal("0")
    assert report.vaulted_lot_count == 0
    assert report.vaulted_quantity == Decimal("0")
    assert report.campaign_active is False
    assert report.cooldown_until == "2026-10-16T14:15:00Z"


def test_report_matches_after_snapshot_json_restore(
    session: SimulationSession,
) -> None:
    active_campaign = BrawlerCampaignState(
        entry_price=100.0,
        entry_timestamp="2026-10-16T14:00:00Z",
        rsi_peak=60.0,
    )
    cooldown_campaign = BrawlerCampaignState(
        cooldown_until="2026-10-16T14:15:00Z",
    )

    assert (
        session.process_brawler_evaluation(
            brawler_evaluation(
                DecisionAction.ENTER,
                close=100.0,
                next_campaign=active_campaign,
            )
        ).outcome
        is SimulationAdapterOutcome.EXECUTED
    )

    assert (
        session.process_brawler_evaluation(
            brawler_evaluation(
                DecisionAction.VAULT,
                timestamp="2026-10-16T14:05:00+00:00",
                close=98.0,
                next_campaign=cooldown_campaign,
            )
        ).outcome
        is SimulationAdapterOutcome.EXECUTED
    )

    original_report = build_replay_report(session)
    restored = restore(session)
    restored_report = build_replay_report(restored)

    assert restored_report == original_report
    assert (
        SimulationSessionSnapshot.capture(restored).to_dict()
        == SimulationSessionSnapshot.capture(session).to_dict()
    )


def test_build_replay_report_rejects_non_session_input() -> None:
    with pytest.raises(TypeError, match="session must be a SimulationSession"):
        build_replay_report(object())  # type: ignore[arg-type]
