from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from src.simulation.fill_model import SimulationFillConfig
from src.simulation.risk_engine import SimulationRiskLimits
from src.simulation.session import SimulationSession, SimulationSessionConfig
from src.simulation.strategy_adapter import (
    SimulationAdapterOutcome,
    SimulationStrategyAdapterConfig,
)
from src.strategies.brawler_strategy import (
    BrawlerCampaignState,
    BrawlerDecision,
    BrawlerEvaluation,
)
from src.strategies.ema_regime_strategy import DecisionAction, StrategyDecision


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
            reason="test",
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


def execution_decision(
    action: DecisionAction,
    *,
    timestamp: str = "2026-10-16T14:00:00+00:00",
    close: float = 100.0,
) -> StrategyDecision:
    return StrategyDecision(
        action=action,
        reason="test setup",
        candle_timestamp=timestamp,
        close=close,
        rsi=None,
        ema_fast=None,
        ema_slow=None,
        ema_200=None,
        adx=None,
        plus_di=None,
        minus_di=None,
    )


@pytest.fixture
def config() -> SimulationSessionConfig:
    return SimulationSessionConfig(
        run_id="brawler-campaign-commit-001",
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


def test_risk_rejected_brawler_enter_does_not_commit_campaign(
    config: SimulationSessionConfig,
) -> None:
    initial_campaign = BrawlerCampaignState()
    proposed_campaign = BrawlerCampaignState(
        entry_price=100.0,
        entry_timestamp="2026-10-16T14:00:00Z",
        rsi_peak=60.0,
    )

    session = SimulationSession(config)

    # Force an order-notional rejection: 20 * 100 = 2000 > 1000.
    session._adapter.config = SimulationStrategyAdapterConfig(
        symbol="BTC/USD",
        entry_quantity=Decimal("20"),
        max_exit_quantity=Decimal("1"),
    )

    result = session.process_brawler_evaluation(
        brawler_evaluation(
            DecisionAction.ENTER,
            next_campaign=proposed_campaign,
        )
    )

    assert result.outcome is SimulationAdapterOutcome.REJECTED
    assert result.run_results[0].risk_decision.approved is False
    assert result.run_results[0].risk_decision.code == "MAX_ORDER_NOTIONAL_EXCEEDED"

    # Most importantly: no false active campaign is persisted.
    assert session.brawler_campaign == initial_campaign
    assert session.brawler_campaign.is_active is False
    assert session.ledger_state.position_quantities == {}
    assert session.active_entry_price is None
    assert session.active_entry_timestamp is None


def test_successful_brawler_enter_commits_proposed_campaign(
    config: SimulationSessionConfig,
) -> None:
    proposed_campaign = BrawlerCampaignState(
        entry_price=100.0,
        entry_timestamp="2026-10-16T14:00:00Z",
        rsi_peak=60.0,
    )
    session = SimulationSession(config)

    result = session.process_brawler_evaluation(
        brawler_evaluation(
            DecisionAction.ENTER,
            next_campaign=proposed_campaign,
        )
    )

    assert result.outcome is SimulationAdapterOutcome.EXECUTED
    assert session.brawler_campaign == proposed_campaign
    assert session.brawler_campaign.is_active is True

    assert session.ledger_state.position_quantities == {"BTC": Decimal("2")}
    assert session.active_entry_price == Decimal("100")
    assert session.active_entry_timestamp is not None


def test_successful_brawler_vault_commits_cooldown_campaign(
    config: SimulationSessionConfig,
) -> None:
    active_campaign = BrawlerCampaignState(
        entry_price=100.0,
        entry_timestamp="2026-10-16T14:00:00Z",
        rsi_peak=62.0,
    )
    cooldown_campaign = BrawlerCampaignState(
        cooldown_until="2026-10-16T14:15:00Z",
    )

    session = SimulationSession(
        config,
        initial_brawler_campaign=active_campaign,
    )

    enter_result = session.process_decision(
        execution_decision(DecisionAction.ENTER)
    )
    assert enter_result.outcome is SimulationAdapterOutcome.EXECUTED

    vault_result = session.process_brawler_evaluation(
        brawler_evaluation(
            DecisionAction.VAULT,
            timestamp="2026-10-16T14:00:30+00:00",
            close=98.0,
            next_campaign=cooldown_campaign,
        )
    )

    assert vault_result.outcome is SimulationAdapterOutcome.EXECUTED

    # VAULT clears active management and persists Brawler's cooldown proposal.
    assert session.brawler_campaign == cooldown_campaign
    assert session.brawler_campaign.is_active is False
    assert session.brawler_campaign.cooldown_until == "2026-10-16T14:15:00Z"

    assert session.ledger_state.position_quantities == {}
    assert len(session.vaulted_lots) == 1
    assert session.active_entry_price is None
    assert session.active_entry_timestamp is None


def test_rejected_brawler_vault_does_not_commit_campaign(
    config: SimulationSessionConfig,
) -> None:
    active_campaign = BrawlerCampaignState(
        entry_price=100.0,
        entry_timestamp="2026-10-16T14:00:00Z",
        rsi_peak=62.0,
    )
    proposed_cooldown = BrawlerCampaignState(
        cooldown_until="2026-10-16T14:15:00Z",
    )

    session = SimulationSession(
        config,
        initial_brawler_campaign=active_campaign,
    )

    enter_result = session.process_decision(
        execution_decision(DecisionAction.ENTER)
    )
    assert enter_result.outcome is SimulationAdapterOutcome.EXECUTED

    # Simulates a legacy/corrupted runtime state where inventory lacks
    # acquisition metadata, causing adapter-side VAULT rejection.
    session._adapter._active_entry_price = None
    session._adapter._active_entry_timestamp = None

    result = session.process_brawler_evaluation(
        brawler_evaluation(
            DecisionAction.VAULT,
            timestamp="2026-10-16T14:00:30+00:00",
            close=98.0,
            next_campaign=proposed_cooldown,
        )
    )

    assert result.outcome is SimulationAdapterOutcome.REJECTED

    # A failed vault must not end the active Brawler campaign.
    assert session.brawler_campaign == active_campaign
    assert session.ledger_state.position_quantities == {"BTC": Decimal("2")}
    assert session.vaulted_lots == ()
