from __future__ import annotations
from datetime import timedelta

from dataclasses import replace
from decimal import Decimal

import pytest

from src.simulation.backtest_events import (
    BacktestEventState,
    build_backtest_event,
    capture_backtest_event_state,
)
from src.simulation.fill_model import SimulationFillConfig
from src.simulation.risk_engine import SimulationRiskLimits
from src.simulation.session import SimulationSession, SimulationSessionConfig
from src.simulation.strategy_adapter import SimulationStrategyAdapterConfig
from src.strategies.brawler_strategy import (
    BrawlerCampaignState,
    BrawlerDecision,
    BrawlerEvaluation,
)
from src.strategies.ema_regime_strategy import DecisionAction


TIMESTAMP = "2026-11-08T12:00:00+00:00"
NEXT_TIMESTAMP = "2026-11-08T12:05:00+00:00"


@pytest.fixture
def session_config() -> SimulationSessionConfig:
    return SimulationSessionConfig(
        run_id="backtest-events-001",
        strategy_adapter=SimulationStrategyAdapterConfig(
            symbol="BTC/USD",
            entry_quantity=Decimal("2"),
            max_exit_quantity=Decimal("2"),
        ),
        risk_limits=SimulationRiskLimits(
            allowed_symbols=frozenset({"BTC/USD"}),
            max_order_notional=Decimal("1000"),
            max_position_notional=Decimal("5000"),
            max_total_exposure=Decimal("5000"),
            max_market_data_age=timedelta(minutes=5),
            estimated_fee_rate=Decimal("0"),
        ),
        fill_config=SimulationFillConfig(
            fee_rate=Decimal("0"),
            slippage_rate=Decimal("0"),
        ),
        initial_cash_balance=Decimal("10000"),
        cash_currency="USD",
    )


def brawler_evaluation(
    action: DecisionAction,
    *,
    timestamp: str = TIMESTAMP,
    reason: str = "TEST_DECISION",
    campaign: BrawlerCampaignState | None = None,
    next_campaign: BrawlerCampaignState | None = None,
) -> BrawlerEvaluation:
    current_campaign = campaign or BrawlerCampaignState()

    if next_campaign is None:
        next_campaign = current_campaign

    return BrawlerEvaluation(
        decision=BrawlerDecision(
            action=action,
            reason=reason,
            candle_timestamp=timestamp,
            close=100.0,
            rsi_5m=50.0,
            rsi_1h=50.0,
            volume_ratio=1.0,
            entry_price=current_campaign.entry_price,
            rsi_peak=current_campaign.rsi_peak,
        ),
        next_campaign=next_campaign,
    )


def test_capture_backtest_event_state_reads_session_without_mutation(
    session_config: SimulationSessionConfig,
) -> None:
    session = SimulationSession(session_config)

    before = capture_backtest_event_state(session)
    after = capture_backtest_event_state(session)

    assert before == after
    assert before == BacktestEventState(
        cash_balance=Decimal("10000"),
        active_quantity=Decimal("0"),
        vaulted_quantity=Decimal("0"),
        campaign_state="IDLE",
    )
    assert session.ledger_state.cash_balance == Decimal("10000")
    assert session.ledger_state.position_quantities == {}
    assert session.vaulted_lots == ()
    assert session.brawler_campaign == BrawlerCampaignState()


def test_risk_rejected_enter_event_preserves_financial_and_campaign_state(
    session_config: SimulationSessionConfig,
) -> None:
    rejecting_config = replace(
        session_config,
        risk_limits=replace(
            session_config.risk_limits,
            max_order_notional=Decimal("100"),
        ),
    )
    session = SimulationSession(rejecting_config)

    evaluation = brawler_evaluation(
        DecisionAction.ENTER,
        reason="ENTER_TEST_RISK_REJECTION",
        next_campaign=BrawlerCampaignState(
            entry_price=100.0,
            entry_timestamp=TIMESTAMP,
            rsi_peak=50.0,
        ),
    )

    state_before = capture_backtest_event_state(session)
    result = session.process_brawler_evaluation(evaluation)
    state_after = capture_backtest_event_state(session)

    event = build_backtest_event(
        sequence=1,
        evaluation=evaluation,
        result=result,
        state_before=state_before,
        state_after=state_after,
    )

    assert event.sequence == 1
    assert event.candle_timestamp == TIMESTAMP
    assert event.action == "ENTER"
    assert event.reason == "ENTER_TEST_RISK_REJECTION"
    assert event.outcome == "REJECTED"

    assert event.risk_approved is False
    assert event.risk_code == "MAX_ORDER_NOTIONAL_EXCEEDED"

    assert event.cash_before == Decimal("10000")
    assert event.cash_after == Decimal("10000")
    assert event.active_quantity_before == Decimal("0")
    assert event.active_quantity_after == Decimal("0")
    assert event.vaulted_quantity_before == Decimal("0")
    assert event.vaulted_quantity_after == Decimal("0")
    assert event.campaign_before == "IDLE"
    assert event.campaign_after == "IDLE"

    assert session.brawler_campaign == BrawlerCampaignState()


def test_executed_enter_event_records_durable_position_and_campaign_transition(
    session_config: SimulationSessionConfig,
) -> None:
    session = SimulationSession(session_config)

    next_campaign = BrawlerCampaignState(
        entry_price=100.0,
        entry_timestamp=TIMESTAMP,
        rsi_peak=50.0,
    )
    evaluation = brawler_evaluation(
        DecisionAction.ENTER,
        reason="ENTER_TEST_EXECUTED",
        next_campaign=next_campaign,
    )

    state_before = capture_backtest_event_state(session)
    result = session.process_brawler_evaluation(evaluation)
    state_after = capture_backtest_event_state(session)

    event = build_backtest_event(
        sequence=1,
        evaluation=evaluation,
        result=result,
        state_before=state_before,
        state_after=state_after,
    )

    assert event.action == "ENTER"
    assert event.outcome == "EXECUTED"
    assert event.risk_approved is True
    assert event.risk_code is not None

    assert event.cash_before == Decimal("10000")
    assert event.cash_after == Decimal("9800")
    assert event.active_quantity_before == Decimal("0")
    assert event.active_quantity_after == Decimal("2")
    assert event.vaulted_quantity_before == Decimal("0")
    assert event.vaulted_quantity_after == Decimal("0")
    assert event.campaign_before == "IDLE"
    assert event.campaign_after == "ACTIVE"

    assert session.brawler_campaign == next_campaign


def test_executed_vault_event_moves_inventory_to_vault_and_starts_cooldown(
    session_config: SimulationSessionConfig,
) -> None:
    session = SimulationSession(session_config)

    entry_campaign = BrawlerCampaignState(
        entry_price=100.0,
        entry_timestamp=TIMESTAMP,
        rsi_peak=50.0,
    )
    enter_evaluation = brawler_evaluation(
        DecisionAction.ENTER,
        next_campaign=entry_campaign,
    )
    session.process_brawler_evaluation(enter_evaluation)

    cooldown_campaign = BrawlerCampaignState(
        cooldown_until="2026-11-08T12:20:00+00:00",
    )
    vault_evaluation = brawler_evaluation(
        DecisionAction.VAULT,
        timestamp=NEXT_TIMESTAMP,
        reason="VAULT_TEST_EXECUTED",
        campaign=entry_campaign,
        next_campaign=cooldown_campaign,
    )

    state_before = capture_backtest_event_state(session)
    result = session.process_brawler_evaluation(vault_evaluation)
    state_after = capture_backtest_event_state(session)

    event = build_backtest_event(
        sequence=2,
        evaluation=vault_evaluation,
        result=result,
        state_before=state_before,
        state_after=state_after,
    )

    assert event.sequence == 2
    assert event.action == "VAULT"
    assert event.reason == "VAULT_TEST_EXECUTED"
    assert event.outcome == "EXECUTED"

    # VAULT is not a sale: the virtual cash balance must not change.
    assert event.cash_before == Decimal("9800")
    assert event.cash_after == Decimal("9800")

    assert event.active_quantity_before == Decimal("2")
    assert event.active_quantity_after == Decimal("0")
    assert event.vaulted_quantity_before == Decimal("0")
    assert event.vaulted_quantity_after == Decimal("2")

    assert event.campaign_before == "ACTIVE"
    assert event.campaign_after == "COOLDOWN"
    assert session.brawler_campaign == cooldown_campaign
    assert len(session.vaulted_lots) == 1


def test_build_backtest_event_rejects_invalid_sequence(
    session_config: SimulationSessionConfig,
) -> None:
    session = SimulationSession(session_config)
    evaluation = brawler_evaluation(DecisionAction.HOLD)

    state_before = capture_backtest_event_state(session)
    result = session.process_brawler_evaluation(evaluation)
    state_after = capture_backtest_event_state(session)

    with pytest.raises(ValueError, match="sequence"):
        build_backtest_event(
            sequence=0,
            evaluation=evaluation,
            result=result,
            state_before=state_before,
            state_after=state_after,
        )
