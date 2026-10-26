from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from decimal import Decimal

import pytest

from src.simulation.fill_model import SimulationFillConfig
from src.simulation.risk_engine import SimulationRiskLimits
from src.simulation.session import SimulationSession, SimulationSessionConfig
from src.simulation.session_snapshot import (
    SNAPSHOT_VERSION,
    SimulationSessionSnapshot,
)
from src.simulation.strategy_adapter import (
    SimulationAdapterOutcome,
    SimulationStrategyAdapterConfig,
)
from src.strategies.brawler_strategy import BrawlerCampaignState
from src.strategies.ema_regime_strategy import DecisionAction, StrategyDecision


def decision(
    action: DecisionAction,
    *,
    timestamp: str = "2026-10-16T14:00:00+00:00",
    close: float | None = 100.0,
) -> StrategyDecision:
    return StrategyDecision(
        action=action,
        reason="test",
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
        run_id="snapshot-session-001",
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


def assert_restored_state_matches(
    original: SimulationSession,
    restored: SimulationSession,
) -> None:
    assert restored.config == original.config
    assert restored.ledger_state == original.ledger_state
    assert restored.risk_context == original.risk_context
    assert restored.processed_decision_keys == original.processed_decision_keys
    assert restored.vaulted_lots == original.vaulted_lots
    assert restored.active_entry_price == original.active_entry_price
    assert restored.active_entry_timestamp == original.active_entry_timestamp
    assert restored.brawler_campaign == original.brawler_campaign

    original_snapshot = SimulationSessionSnapshot.capture(original)
    restored_snapshot = SimulationSessionSnapshot.capture(restored)

    assert restored_snapshot.event_sequence == original_snapshot.event_sequence
    assert restored_snapshot.runner_sequence == original_snapshot.runner_sequence
    assert restored_snapshot.to_dict() == original_snapshot.to_dict()
    assert restored_snapshot.to_json() == original_snapshot.to_json()


def test_default_session_snapshot_round_trip(
    config: SimulationSessionConfig,
) -> None:
    session = SimulationSession(config)

    snapshot = SimulationSessionSnapshot.capture(session)
    restored = snapshot.restore()

    assert snapshot.version == SNAPSHOT_VERSION
    assert_restored_state_matches(session, restored)
    assert restored.events == ()


def test_active_position_snapshot_round_trip_preserves_entry_metadata_and_idempotency(
    config: SimulationSessionConfig,
) -> None:
    session = SimulationSession(
        config,
        initial_brawler_campaign=BrawlerCampaignState(
            entry_price=100.0,
            entry_timestamp="2026-10-16T14:00:00Z",
            rsi_peak=61.0,
        ),
    )

    enter = decision(DecisionAction.ENTER)
    result = session.process_decision(enter)

    assert result.outcome is SimulationAdapterOutcome.EXECUTED
    assert session.active_entry_price == Decimal("100")
    assert session.active_entry_timestamp is not None

    snapshot = SimulationSessionSnapshot.capture(session)
    restored = SimulationSessionSnapshot.from_json(snapshot.to_json()).restore()

    assert_restored_state_matches(session, restored)
    assert restored.brawler_campaign == BrawlerCampaignState(
        entry_price=100.0,
        entry_timestamp="2026-10-16T14:00:00Z",
        rsi_peak=61.0,
    )

    original_duplicate = session.process_decision(enter)
    restored_duplicate = restored.process_decision(enter)

    assert original_duplicate.outcome is SimulationAdapterOutcome.DUPLICATE
    assert restored_duplicate.outcome is SimulationAdapterOutcome.DUPLICATE
    assert_restored_state_matches(session, restored)


def test_cooldown_campaign_snapshot_round_trip(
    config: SimulationSessionConfig,
) -> None:
    cooldown_campaign = BrawlerCampaignState(
        cooldown_until="2026-10-16T14:15:00Z",
    )
    session = SimulationSession(
        config,
        initial_brawler_campaign=cooldown_campaign,
    )

    snapshot = SimulationSessionSnapshot.capture(session)
    restored = SimulationSessionSnapshot.from_json(snapshot.to_json()).restore()

    assert_restored_state_matches(session, restored)
    assert restored.brawler_campaign == cooldown_campaign
    assert restored.brawler_campaign.is_active is False


def test_vaulted_inventory_snapshot_round_trip(
    config: SimulationSessionConfig,
) -> None:
    session = SimulationSession(config)

    enter_result = session.process_decision(
        decision(
            DecisionAction.ENTER,
            timestamp="2026-10-16T14:00:00+00:00",
            close=100.0,
        )
    )
    vault_result = session.process_decision(
        decision(
            DecisionAction.VAULT,
            timestamp="2026-10-16T14:00:30+00:00",
            close=98.5,
        )
    )

    assert enter_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert vault_result.outcome is SimulationAdapterOutcome.EXECUTED
    assert len(session.vaulted_lots) == 1
    assert session.active_entry_price is None
    assert session.active_entry_timestamp is None

    snapshot = SimulationSessionSnapshot.capture(session)
    restored = SimulationSessionSnapshot.from_json(snapshot.to_json()).restore()

    assert_restored_state_matches(session, restored)

    vaulted_lot = restored.vaulted_lots[0]
    assert vaulted_lot.symbol == "BTC/USD"
    assert vaulted_lot.quantity == Decimal("2")
    assert vaulted_lot.acquisition_price == Decimal("100")
    assert restored.ledger_state.position_quantities == {}
    assert restored.risk_context.position_quantities == {}


def test_snapshot_json_is_deterministic_and_round_trips(
    config: SimulationSessionConfig,
) -> None:
    session = SimulationSession(config)
    session.process_decision(decision(DecisionAction.ENTER))

    snapshot = SimulationSessionSnapshot.capture(session)
    first_json = snapshot.to_json()
    second_json = snapshot.to_json()

    restored_snapshot = SimulationSessionSnapshot.from_json(first_json)

    assert first_json == second_json
    assert restored_snapshot == snapshot
    assert restored_snapshot.to_json() == first_json


def test_snapshot_rejects_legacy_v1_payload(
    config: SimulationSessionConfig,
) -> None:
    payload = SimulationSessionSnapshot.capture(
        SimulationSession(config)
    ).to_dict()
    payload["version"] = 1

    with pytest.raises(
        ValueError,
        match="unsupported simulation session snapshot version: 1",
    ):
        SimulationSessionSnapshot.from_dict(payload)


def test_snapshot_rejects_open_position_without_active_entry_metadata(
    config: SimulationSessionConfig,
) -> None:
    session = SimulationSession(config)
    session.process_decision(decision(DecisionAction.ENTER))

    payload = SimulationSessionSnapshot.capture(session).to_dict()
    payload["active_entry_price"] = None
    payload["active_entry_timestamp"] = None

    with pytest.raises(
        ValueError,
        match="require active entry metadata for an open configured-symbol position",
    ):
        SimulationSessionSnapshot.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "active_entry_price",
            None,
            "must either both be present or both be absent",
        ),
        (
            "active_entry_timestamp",
            "2026-10-16T14:00:00",
            "must include timezone information",
        ),
    ],
)
def test_snapshot_rejects_invalid_active_entry_metadata(
    config: SimulationSessionConfig,
    field: str,
    value: object,
    message: str,
) -> None:
    session = SimulationSession(config)
    session.process_decision(decision(DecisionAction.ENTER))

    payload = deepcopy(SimulationSessionSnapshot.capture(session).to_dict())
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        SimulationSessionSnapshot.from_dict(payload)
