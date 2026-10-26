from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from src.simulation.fill_model import SimulationFillConfig
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
from src.strategies.ema_regime_strategy import DecisionAction, StrategyDecision


def decision(
    action: DecisionAction,
    *,
    timestamp: str,
    close: float,
) -> StrategyDecision:
    return StrategyDecision(
        action=action,
        reason="restore continuation test",
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


def brawler_evaluation(
    action: DecisionAction,
    *,
    timestamp: str,
    close: float,
    next_campaign: BrawlerCampaignState,
) -> BrawlerEvaluation:
    return BrawlerEvaluation(
        decision=BrawlerDecision(
            action=action,
            reason="restore continuation test",
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


@pytest.fixture
def config() -> SimulationSessionConfig:
    return SimulationSessionConfig(
        run_id="restore-continue-001",
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


def restore(session: SimulationSession) -> SimulationSession:
    snapshot = SimulationSessionSnapshot.capture(session)
    return SimulationSessionSnapshot.from_json(snapshot.to_json()).restore()


def assert_continuations_match(
    uninterrupted: SimulationSession,
    restored: SimulationSession,
) -> None:
    uninterrupted_snapshot = SimulationSessionSnapshot.capture(uninterrupted)
    restored_snapshot = SimulationSessionSnapshot.capture(restored)

    assert restored.config == uninterrupted.config
    assert restored.ledger_state == uninterrupted.ledger_state
    assert restored.risk_context == uninterrupted.risk_context
    assert restored.processed_decision_keys == uninterrupted.processed_decision_keys
    assert restored.vaulted_lots == uninterrupted.vaulted_lots
    assert restored.active_entry_price == uninterrupted.active_entry_price
    assert restored.active_entry_timestamp == uninterrupted.active_entry_timestamp
    assert restored.brawler_campaign == uninterrupted.brawler_campaign

    # Includes the restored event and runner counters, along with every
    # resumable state value. Historic in-memory event lists are intentional
    # exclusions from snapshot restoration.
    assert restored_snapshot.to_dict() == uninterrupted_snapshot.to_dict()
    assert restored_snapshot.to_json() == uninterrupted_snapshot.to_json()


def test_enter_snapshot_restore_then_vault_matches_uninterrupted_execution(
    config: SimulationSessionConfig,
) -> None:
    enter = decision(
        DecisionAction.ENTER,
        timestamp="2026-10-16T14:00:00+00:00",
        close=100.0,
    )
    vault = decision(
        DecisionAction.VAULT,
        timestamp="2026-10-16T14:00:30+00:00",
        close=98.0,
    )

    uninterrupted = SimulationSession(config)
    assert uninterrupted.process_decision(enter).outcome is SimulationAdapterOutcome.EXECUTED

    restored = restore(uninterrupted)

    uninterrupted_result = uninterrupted.process_decision(vault)
    restored_result = restored.process_decision(vault)

    assert uninterrupted_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert restored_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert restored_result == uninterrupted_result

    assert len(restored.vaulted_lots) == 1
    assert restored.ledger_state.position_quantities == {}
    assert restored.active_entry_price is None
    assert restored.active_entry_timestamp is None
    assert_continuations_match(uninterrupted, restored)


def test_enter_snapshot_restore_then_exit_matches_uninterrupted_execution(
    config: SimulationSessionConfig,
) -> None:
    enter = decision(
        DecisionAction.ENTER,
        timestamp="2026-10-16T14:00:00+00:00",
        close=100.0,
    )
    exit_decision = decision(
        DecisionAction.EXIT,
        timestamp="2026-10-16T14:00:30+00:00",
        close=110.0,
    )

    uninterrupted = SimulationSession(config)
    assert uninterrupted.process_decision(enter).outcome is SimulationAdapterOutcome.EXECUTED

    restored = restore(uninterrupted)

    uninterrupted_result = uninterrupted.process_decision(exit_decision)
    restored_result = restored.process_decision(exit_decision)

    assert uninterrupted_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert restored_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert restored_result == uninterrupted_result

    assert restored.ledger_state.position_quantities == {}
    assert restored.risk_context.position_quantities == {}
    assert restored.active_entry_price is None
    assert restored.active_entry_timestamp is None
    assert_continuations_match(uninterrupted, restored)


def test_duplicate_decision_remains_duplicate_after_restore(
    config: SimulationSessionConfig,
) -> None:
    enter = decision(
        DecisionAction.ENTER,
        timestamp="2026-10-16T14:00:00+00:00",
        close=100.0,
    )

    uninterrupted = SimulationSession(config)
    assert uninterrupted.process_decision(enter).outcome is SimulationAdapterOutcome.EXECUTED

    restored = restore(uninterrupted)

    uninterrupted_duplicate = uninterrupted.process_decision(enter)
    restored_duplicate = restored.process_decision(enter)

    assert uninterrupted_duplicate.outcome is SimulationAdapterOutcome.DUPLICATE
    assert restored_duplicate.outcome is SimulationAdapterOutcome.DUPLICATE
    assert restored_duplicate == uninterrupted_duplicate
    assert restored_duplicate.run_results == ()

    # The restored session has no historic events, but its next event preserves
    # the durable sequence value from the snapshot.
    assert restored.events[-1].sequence == 2
    assert_continuations_match(uninterrupted, restored)


def test_post_restore_execution_preserves_event_and_runner_sequences(
    config: SimulationSessionConfig,
) -> None:
    enter = decision(
        DecisionAction.ENTER,
        timestamp="2026-10-16T14:00:00+00:00",
        close=100.0,
    )
    exit_decision = decision(
        DecisionAction.EXIT,
        timestamp="2026-10-16T14:00:30+00:00",
        close=110.0,
    )

    uninterrupted = SimulationSession(config)
    assert uninterrupted.process_decision(enter).outcome is SimulationAdapterOutcome.EXECUTED

    pre_restore_snapshot = SimulationSessionSnapshot.capture(uninterrupted)
    assert pre_restore_snapshot.event_sequence == 1
    assert pre_restore_snapshot.runner_sequence == 4

    restored = pre_restore_snapshot.restore()

    uninterrupted_result = uninterrupted.process_decision(exit_decision)
    restored_result = restored.process_decision(exit_decision)

    assert uninterrupted_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert restored_result.outcome is SimulationAdapterOutcome.EXECUTED

    # EXIT of quantity two is split into two sell intents, consuming three
    # runner sequence values per intent: 4 -> 10.
    uninterrupted_snapshot = SimulationSessionSnapshot.capture(uninterrupted)
    restored_snapshot = SimulationSessionSnapshot.capture(restored)

    assert uninterrupted_snapshot.event_sequence == 2
    assert restored_snapshot.event_sequence == 2
    assert uninterrupted_snapshot.runner_sequence == 10
    assert restored_snapshot.runner_sequence == 10
    assert restored.events[-1].sequence == 2
    assert_continuations_match(uninterrupted, restored)


def test_brawler_campaign_continues_identically_after_restore(
    config: SimulationSessionConfig,
) -> None:
    active_campaign = BrawlerCampaignState(
        entry_price=100.0,
        entry_timestamp="2026-10-16T14:00:00Z",
        rsi_peak=60.0,
    )
    cooldown_campaign = BrawlerCampaignState(
        cooldown_until="2026-10-16T14:15:00Z",
    )

    enter_evaluation = brawler_evaluation(
        DecisionAction.ENTER,
        timestamp="2026-10-16T14:00:00+00:00",
        close=100.0,
        next_campaign=active_campaign,
    )
    vault_evaluation = brawler_evaluation(
        DecisionAction.VAULT,
        timestamp="2026-10-16T14:00:30+00:00",
        close=98.0,
        next_campaign=cooldown_campaign,
    )

    uninterrupted = SimulationSession(config)
    enter_result = uninterrupted.process_brawler_evaluation(enter_evaluation)
    assert enter_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert uninterrupted.brawler_campaign == active_campaign

    restored = restore(uninterrupted)
    assert restored.brawler_campaign == active_campaign

    uninterrupted_result = uninterrupted.process_brawler_evaluation(vault_evaluation)
    restored_result = restored.process_brawler_evaluation(vault_evaluation)

    assert uninterrupted_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert restored_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert restored_result == uninterrupted_result

    assert restored.brawler_campaign == cooldown_campaign
    assert restored.brawler_campaign.is_active is False
    assert len(restored.vaulted_lots) == 1
    assert_continuations_match(uninterrupted, restored)
