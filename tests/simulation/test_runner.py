"""Integration tests for the deterministic simulation orchestration pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from src.simulation.audit_writer import SimulationAuditWriter
from src.simulation.fill_model import SimulationFillConfig, SimulationFillModel
from src.simulation.ledger import SimulationLedger, SimulationLedgerState
from src.simulation.models import OrderIntent, Side
from src.simulation.risk_engine import (
    SimulationRiskContext,
    SimulationRiskEngine,
    SimulationRiskLimits,
)
from src.simulation.runner import SimulationRunner


NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
RUN_ID = "sim-test-runner"


@pytest.fixture
def audit_path() -> Path:
    path = Path(f"data/simulation/{RUN_ID}.jsonl").resolve()
    path.unlink(missing_ok=True)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def runner() -> SimulationRunner:
    limits = SimulationRiskLimits(
        allowed_symbols=frozenset({"BTC/USD"}),
        max_order_notional=Decimal("1000"),
        max_position_notional=Decimal("2000"),
        max_total_exposure=Decimal("2000"),
        max_market_data_age=timedelta(minutes=5),
        estimated_fee_rate=Decimal("0"),
    )
    return SimulationRunner(
        risk_engine=SimulationRiskEngine(limits),
        fill_model=SimulationFillModel(
            SimulationFillConfig(
                fee_rate=Decimal("0"),
                slippage_rate=Decimal("0"),
            )
        ),
        ledger=SimulationLedger(),
        audit_writer=SimulationAuditWriter(),
    )


def make_intent(
    *,
    intent_id: str = "intent-001",
    side: Side = Side.BUY,
    quantity: Decimal = Decimal("2"),
    reference_price: Decimal = Decimal("100"),
) -> OrderIntent:
    return OrderIntent(
        intent_id=intent_id,
        symbol="BTC/USD",
        side=side,
        quantity=quantity,
        reference_price=reference_price,
        market_data_timestamp=NOW,
        created_at=NOW,
    )


def make_state(
    *,
    cash: Decimal = Decimal("1000"),
    positions: dict[str, Decimal] | None = None,
) -> SimulationLedgerState:
    return SimulationLedgerState(
        cash_balance=cash,
        cash_currency="USD",
        position_quantities=positions or {},
    )


def make_context(
    *,
    cash: Decimal = Decimal("1000"),
    positions: dict[str, Decimal] | None = None,
    exposure: Decimal = Decimal("0"),
    processed: frozenset[str] = frozenset(),
) -> SimulationRiskContext:
    return SimulationRiskContext(
        available_cash=cash,
        position_quantities=positions or {},
        current_total_exposure=exposure,
        processed_intent_ids=processed,
    )


def read_records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_approved_buy_runs_full_pipeline_and_writes_audit_events(
    runner: SimulationRunner,
    audit_path: Path,
) -> None:
    original_state = make_state()
    original_context = make_context()

    result = runner.process_intent(
        make_intent(),
        state=original_state,
        risk_context=original_context,
        run_id=RUN_ID,
        sequence=10,
        evaluated_at=NOW,
    )

    assert result.risk_decision.approved is True
    assert result.fill is not None
    assert result.ledger_entry is not None
    assert result.fill.fill_id == f"{RUN_ID}:fill:000011"
    assert result.ledger_entry.entry_id == f"{RUN_ID}:ledger:12"

    assert result.state.cash_balance == Decimal("800")
    assert result.state.position_quantities["BTC"] == Decimal("2")
    assert result.risk_context.available_cash == Decimal("800")
    assert result.risk_context.position_quantities == {"BTC/USD": Decimal("2")}
    assert result.risk_context.current_total_exposure == Decimal("200")
    assert result.risk_context.processed_intent_ids == frozenset({"intent-001"})

    assert original_state.cash_balance == Decimal("1000")
    assert dict(original_state.position_quantities) == {}
    assert original_context.available_cash == Decimal("1000")
    assert dict(original_context.position_quantities) == {}

    records = read_records(audit_path)
    assert [record["event_id"] for record in records] == [
        f"{RUN_ID}:audit:000010",
        f"{RUN_ID}:audit:000011",
        f"{RUN_ID}:audit:000012",
    ]
    assert [record["event_type"] for record in records] == [
        "RISK_DECISION",
        "SIMULATED_FILL",
        "LEDGER_TRANSITION",
    ]


def test_approved_sell_reduces_position_and_exposure(
    runner: SimulationRunner,
    audit_path: Path,
) -> None:
    result = runner.process_intent(
        make_intent(
            intent_id="intent-sell-001",
            side=Side.SELL,
            quantity=Decimal("1"),
            reference_price=Decimal("100"),
        ),
        state=make_state(cash=Decimal("500"), positions={"BTC": Decimal("2")}),
        risk_context=make_context(
            cash=Decimal("500"),
            positions={"BTC/USD": Decimal("2")},
            exposure=Decimal("200"),
        ),
        run_id=RUN_ID,
        sequence=1,
        evaluated_at=NOW,
    )

    assert result.risk_decision.approved is True
    assert result.state.cash_balance == Decimal("600")
    assert result.state.position_quantities["BTC"] == Decimal("1")
    assert result.risk_context.position_quantities == {"BTC/USD": Decimal("1")}
    assert result.risk_context.current_total_exposure == Decimal("100")
    assert result.ledger_entry is not None
    assert result.ledger_entry.position_delta == Decimal("-1")


def test_risk_rejection_preserves_state_and_writes_only_risk_event(
    runner: SimulationRunner,
    audit_path: Path,
) -> None:
    state = make_state()
    context = make_context()

    result = runner.process_intent(
        make_intent(quantity=Decimal("20")),
        state=state,
        risk_context=context,
        run_id=RUN_ID,
        sequence=1,
        evaluated_at=NOW,
    )

    assert result.risk_decision.approved is False
    assert result.risk_decision.code == "MAX_ORDER_NOTIONAL_EXCEEDED"
    assert result.fill is None
    assert result.ledger_entry is None
    assert result.state is state
    assert result.state.cash_balance == Decimal("1000")
    assert result.risk_context.available_cash == Decimal("1000")
    assert result.risk_context.processed_intent_ids == frozenset({"intent-001"})

    records = read_records(audit_path)
    assert len(records) == 1
    assert records[0]["event_type"] == "RISK_DECISION"
    assert records[0]["sequence"] == 1


def test_insufficient_cash_is_rejected_before_fill_or_ledger(
    runner: SimulationRunner,
    audit_path: Path,
) -> None:
    result = runner.process_intent(
        make_intent(quantity=Decimal("2"), reference_price=Decimal("100")),
        state=make_state(cash=Decimal("100")),
        risk_context=make_context(cash=Decimal("100")),
        run_id=RUN_ID,
        sequence=1,
        evaluated_at=NOW,
    )

    assert result.risk_decision.approved is False
    assert result.risk_decision.code == "INSUFFICIENT_VIRTUAL_CASH"
    assert result.fill is None
    assert result.ledger_entry is None
    assert result.state.cash_balance == Decimal("100")
    assert len(read_records(audit_path)) == 1


def test_insufficient_position_is_rejected_before_fill_or_ledger(
    runner: SimulationRunner,
    audit_path: Path,
) -> None:
    result = runner.process_intent(
        make_intent(side=Side.SELL, quantity=Decimal("2")),
        state=make_state(cash=Decimal("500"), positions={"BTC": Decimal("1")}),
        risk_context=make_context(
            cash=Decimal("500"),
            positions={"BTC/USD": Decimal("1")},
            exposure=Decimal("100"),
        ),
        run_id=RUN_ID,
        sequence=1,
        evaluated_at=NOW,
    )

    assert result.risk_decision.approved is False
    assert result.risk_decision.code == "INSUFFICIENT_SIMULATED_POSITION"
    assert result.fill is None
    assert result.ledger_entry is None
    assert result.state.position_quantities["BTC"] == Decimal("1")
    assert len(read_records(audit_path)) == 1


def test_processed_intent_cannot_be_run_twice(
    runner: SimulationRunner,
    audit_path: Path,
) -> None:
    first = runner.process_intent(
        make_intent(),
        state=make_state(),
        risk_context=make_context(),
        run_id=RUN_ID,
        sequence=1,
        evaluated_at=NOW,
    )

    second = runner.process_intent(
        make_intent(),
        state=first.state,
        risk_context=first.risk_context,
        run_id=RUN_ID,
        sequence=4,
        evaluated_at=NOW,
    )

    assert first.risk_decision.approved is True
    assert second.risk_decision.approved is False
    assert second.risk_decision.code == "DUPLICATE_INTENT"
    assert second.fill is None
    assert second.ledger_entry is None
    assert second.state is first.state

    records = read_records(audit_path)
    assert [record["sequence"] for record in records] == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("run_id", "sequence", "evaluated_at", "exception_type"),
    [
        ("bad run", 1, NOW, ValueError),
        ("bad:run", 1, NOW, ValueError),
        (RUN_ID, 0, NOW, ValueError),
        (RUN_ID, True, NOW, ValueError),
        (RUN_ID, 1, datetime(2026, 10, 14, 12, 0), ValueError),
    ],
)
def test_rejects_invalid_runner_inputs(
    runner: SimulationRunner,
    audit_path: Path,
    run_id: str,
    sequence: int,
    evaluated_at: datetime,
    exception_type: type[Exception],
) -> None:
    with pytest.raises(exception_type):
        runner.process_intent(
            make_intent(),
            state=make_state(),
            risk_context=make_context(),
            run_id=run_id,
            sequence=sequence,
            evaluated_at=evaluated_at,
        )

    assert not audit_path.exists()
