from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.execution.models import (
    ExchangeOrderRecord,
    ExecutionSide,
    ExecutionStatus,
    IdempotencyClaim,
    StrategyIntentEvent,
)
from src.execution.service import LiveExecutionService
from src.live_risk.models import (
    LiveAccountSnapshot,
    LiveRiskLimits,
    LiveTradingControlState,
    MarketPriceSnapshot,
    RiskDecision,
    RiskDecisionStatus,
)
from src.live_risk.service import (
    ControlledLiveExecutionService,
    LiveRiskControlService,
)


NOW = datetime(2026, 11, 24, 16, 0, tzinfo=timezone.utc)


class InMemoryControlStore:
    def __init__(
        self,
        state: LiveTradingControlState | None = None,
    ) -> None:
        self.state = state or LiveTradingControlState(
            live_trading_enabled=True,
            kill_switch_engaged=False,
            reconciliation_required=False,
            operationally_healthy=True,
        )
        self.decisions: list[RiskDecision] = []
        self.saved_states: list[LiveTradingControlState] = []
        self.raise_on_save_decision: Exception | None = None

    def get_control_state(self) -> LiveTradingControlState:
        return self.state

    def save_control_state(self, state: LiveTradingControlState) -> None:
        self.state = state
        self.saved_states.append(state)

    def save_risk_decision(self, decision: RiskDecision) -> None:
        if self.raise_on_save_decision is not None:
            raise self.raise_on_save_decision
        self.decisions.append(decision)

    def now(self) -> datetime:
        return NOW


class StaticAccountStateProvider:
    def __init__(self, snapshot: LiveAccountSnapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0
        self.raise_on_get: Exception | None = None

    def get_account_snapshot(self) -> LiveAccountSnapshot:
        self.calls += 1
        if self.raise_on_get is not None:
            raise self.raise_on_get
        return self.snapshot


class StaticMarketPriceProvider:
    def __init__(self, snapshot: MarketPriceSnapshot) -> None:
        self.snapshot = snapshot
        self.calls: list[str] = []
        self.raise_on_get: Exception | None = None

    def get_price_snapshot(self, symbol: str) -> MarketPriceSnapshot:
        self.calls.append(symbol)
        if self.raise_on_get is not None:
            raise self.raise_on_get
        return self.snapshot


class InMemoryExecutionStore:
    def __init__(self) -> None:
        self.claims: dict[str, IdempotencyClaim] = {}
        self.orders: dict[str, ExchangeOrderRecord] = {}
        self.reports = []

    def create_idempotency_claim(
        self,
        claim: IdempotencyClaim,
    ) -> tuple[IdempotencyClaim, bool]:
        existing = self.claims.get(claim.idempotency_key)
        if existing is not None:
            return existing, False
        self.claims[claim.idempotency_key] = claim
        return claim, True

    def get_exchange_order(self, idempotency_key: str) -> ExchangeOrderRecord | None:
        return self.orders.get(idempotency_key)

    def save_exchange_order(self, record: ExchangeOrderRecord) -> None:
        self.orders[record.idempotency_key] = record

    def save_execution_report(self, report) -> None:
        self.reports.append(report)

    def update_idempotency_claim(self, claim: IdempotencyClaim) -> None:
        self.claims[claim.idempotency_key] = claim

    def now(self) -> datetime:
        return NOW


class RecordingGateway:
    def __init__(self) -> None:
        self.requests = []
        self.raise_on_submit: Exception | None = None

    def submit_order(self, request):
        self.requests.append(request)
        if self.raise_on_submit is not None:
            raise self.raise_on_submit
        return ExchangeOrderRecord(
            idempotency_key=request.idempotency_key,
            intent_id="intent-001",
            exchange_order_id="exchange-order-001",
            accepted_at=NOW,
        )

    def cancel_order(self, exchange_order_id: str):
        raise AssertionError("not used")

    def fetch_order_status(self, exchange_order_id: str):
        raise AssertionError("not used")


@pytest.fixture
def intent() -> StrategyIntentEvent:
    return StrategyIntentEvent(
        intent_id="intent-001",
        strategy_name="brawler",
        symbol="BTC/USD",
        side=ExecutionSide.BUY,
        quantity=Decimal("1"),
        created_at=NOW,
    )


@pytest.fixture
def limits() -> LiveRiskLimits:
    return LiveRiskLimits(
        allowed_symbols=frozenset({"BTC/USD"}),
        max_order_notional=Decimal("500"),
        max_symbol_exposure=Decimal("1000"),
        max_total_exposure=Decimal("2000"),
        max_open_orders=5,
        max_market_data_age_seconds=60,
        max_account_snapshot_age_seconds=60,
    )


@pytest.fixture
def account_provider() -> StaticAccountStateProvider:
    return StaticAccountStateProvider(
        LiveAccountSnapshot(
            captured_at=NOW,
            symbol_quantities={},
            symbol_exposures={},
            total_exposure=Decimal("0"),
            open_order_count=0,
        )
    )


@pytest.fixture
def price_provider() -> StaticMarketPriceProvider:
    return StaticMarketPriceProvider(
        MarketPriceSnapshot(
            symbol="BTC/USD",
            price=Decimal("100"),
            captured_at=NOW,
        )
    )


def make_risk_service(
    *,
    limits: LiveRiskLimits,
    control_store: InMemoryControlStore,
    account_provider: StaticAccountStateProvider,
    price_provider: StaticMarketPriceProvider,
) -> LiveRiskControlService:
    return LiveRiskControlService(
        limits=limits,
        control_store=control_store,
        account_state_provider=account_provider,
        market_price_provider=price_provider,
    )


def make_controlled_service(
    *,
    risk_service: LiveRiskControlService,
    control_store: InMemoryControlStore,
    gateway: RecordingGateway,
) -> ControlledLiveExecutionService:
    return ControlledLiveExecutionService(
        risk_control_service=risk_service,
        live_execution_service=LiveExecutionService(
            gateway=gateway,
            store=InMemoryExecutionStore(),
        ),
        control_store=control_store,
    )


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (
            LiveTradingControlState(
                live_trading_enabled=False,
                kill_switch_engaged=False,
                reconciliation_required=False,
                operationally_healthy=True,
            ),
            "LIVE_TRADING_DISABLED",
        ),
        (
            LiveTradingControlState(
                live_trading_enabled=True,
                kill_switch_engaged=True,
                reconciliation_required=False,
                operationally_healthy=True,
            ),
            "KILL_SWITCH_ENGAGED",
        ),
        (
            LiveTradingControlState(
                live_trading_enabled=True,
                kill_switch_engaged=False,
                reconciliation_required=True,
                operationally_healthy=True,
            ),
            "RECONCILIATION_REQUIRED",
        ),
        (
            LiveTradingControlState(
                live_trading_enabled=True,
                kill_switch_engaged=False,
                reconciliation_required=False,
                operationally_healthy=False,
            ),
            "OPERATIONAL_HEALTH_UNAVAILABLE",
        ),
    ],
)
def test_control_state_rejection_blocks_gateway(
    intent: StrategyIntentEvent,
    limits: LiveRiskLimits,
    account_provider: StaticAccountStateProvider,
    price_provider: StaticMarketPriceProvider,
    state: LiveTradingControlState,
    reason: str,
) -> None:
    control_store = InMemoryControlStore(state)
    risk_service = make_risk_service(
        limits=limits,
        control_store=control_store,
        account_provider=account_provider,
        price_provider=price_provider,
    )
    gateway = RecordingGateway()
    service = make_controlled_service(
        risk_service=risk_service,
        control_store=control_store,
        gateway=gateway,
    )

    result = service.submit_intent(intent)

    assert result.submitted is False
    assert result.risk_decision.status is RiskDecisionStatus.REJECTED
    assert result.risk_decision.reason == reason
    assert gateway.requests == []
    assert len(control_store.decisions) == 1
    assert account_provider.calls == 0
    assert price_provider.calls == []


def test_approved_decision_is_persisted_before_gateway_submission(
    intent: StrategyIntentEvent,
    limits: LiveRiskLimits,
    account_provider: StaticAccountStateProvider,
    price_provider: StaticMarketPriceProvider,
) -> None:
    control_store = InMemoryControlStore()
    risk_service = make_risk_service(
        limits=limits,
        control_store=control_store,
        account_provider=account_provider,
        price_provider=price_provider,
    )

    class InspectingGateway(RecordingGateway):
        def submit_order(self, request):
            assert len(control_store.decisions) == 1
            assert control_store.decisions[0].status is RiskDecisionStatus.APPROVED
            return super().submit_order(request)

    gateway = InspectingGateway()
    service = make_controlled_service(
        risk_service=risk_service,
        control_store=control_store,
        gateway=gateway,
    )

    result = service.submit_intent(intent)

    assert result.risk_decision.status is RiskDecisionStatus.APPROVED
    assert result.risk_decision.order_notional == Decimal("100")
    assert result.execution_result is not None
    assert result.execution_result.report.status is ExecutionStatus.ACCEPTED
    assert len(gateway.requests) == 1


def test_symbol_not_allowed_blocks_gateway(
    intent: StrategyIntentEvent,
    limits: LiveRiskLimits,
    account_provider: StaticAccountStateProvider,
    price_provider: StaticMarketPriceProvider,
) -> None:
    control_store = InMemoryControlStore()
    other_intent = StrategyIntentEvent(
        intent_id=intent.intent_id,
        strategy_name=intent.strategy_name,
        symbol="ETH/USD",
        side=intent.side,
        quantity=intent.quantity,
        created_at=intent.created_at,
    )
    risk_service = make_risk_service(
        limits=limits,
        control_store=control_store,
        account_provider=account_provider,
        price_provider=price_provider,
    )
    gateway = RecordingGateway()
    service = make_controlled_service(
        risk_service=risk_service,
        control_store=control_store,
        gateway=gateway,
    )

    result = service.submit_intent(other_intent)

    assert result.risk_decision.reason == "SYMBOL_NOT_ALLOWED"
    assert gateway.requests == []
    assert account_provider.calls == 0


@pytest.mark.parametrize(
    ("account_time", "price_time", "reason"),
    [
        (NOW - timedelta(seconds=61), NOW, "ACCOUNT_SNAPSHOT_STALE"),
        (NOW, NOW - timedelta(seconds=61), "MARKET_PRICE_STALE"),
        (NOW + timedelta(seconds=1), NOW, "FUTURE_DATED_SNAPSHOT"),
        (NOW, NOW + timedelta(seconds=1), "FUTURE_DATED_SNAPSHOT"),
    ],
)
def test_stale_or_future_snapshots_block_gateway(
    intent: StrategyIntentEvent,
    limits: LiveRiskLimits,
    account_time: datetime,
    price_time: datetime,
    reason: str,
) -> None:
    control_store = InMemoryControlStore()
    account_provider = StaticAccountStateProvider(
        LiveAccountSnapshot(
            captured_at=account_time,
            symbol_quantities={},
            symbol_exposures={},
            total_exposure=Decimal("0"),
            open_order_count=0,
        )
    )
    price_provider = StaticMarketPriceProvider(
        MarketPriceSnapshot(
            symbol="BTC/USD",
            price=Decimal("100"),
            captured_at=price_time,
        )
    )
    risk_service = make_risk_service(
        limits=limits,
        control_store=control_store,
        account_provider=account_provider,
        price_provider=price_provider,
    )
    gateway = RecordingGateway()
    service = make_controlled_service(
        risk_service=risk_service,
        control_store=control_store,
        gateway=gateway,
    )

    result = service.submit_intent(intent)

    assert result.risk_decision.reason == reason
    assert gateway.requests == []


@pytest.mark.parametrize(
    ("snapshot", "price", "reason"),
    [
        (
            LiveAccountSnapshot(
                captured_at=NOW,
                symbol_quantities={},
                symbol_exposures={},
                total_exposure=Decimal("0"),
                open_order_count=5,
            ),
            Decimal("100"),
            "OPEN_ORDER_LIMIT_REACHED",
        ),
        (
            LiveAccountSnapshot(
                captured_at=NOW,
                symbol_quantities={},
                symbol_exposures={},
                total_exposure=Decimal("0"),
                open_order_count=0,
            ),
            Decimal("501"),
            "ORDER_NOTIONAL_LIMIT_EXCEEDED",
        ),
        (
            LiveAccountSnapshot(
                captured_at=NOW,
                symbol_quantities={},
                symbol_exposures={"BTC/USD": Decimal("950")},
                total_exposure=Decimal("950"),
                open_order_count=0,
            ),
            Decimal("100"),
            "SYMBOL_EXPOSURE_LIMIT_EXCEEDED",
        ),
        (
            LiveAccountSnapshot(
                captured_at=NOW,
                symbol_quantities={},
                symbol_exposures={"BTC/USD": Decimal("100")},
                total_exposure=Decimal("1950"),
                open_order_count=0,
            ),
            Decimal("100"),
            "TOTAL_EXPOSURE_LIMIT_EXCEEDED",
        ),
    ],
)
def test_quantitative_limits_block_gateway(
    intent: StrategyIntentEvent,
    limits: LiveRiskLimits,
    snapshot: LiveAccountSnapshot,
    price: Decimal,
    reason: str,
) -> None:
    control_store = InMemoryControlStore()
    account_provider = StaticAccountStateProvider(snapshot)
    price_provider = StaticMarketPriceProvider(
        MarketPriceSnapshot(
            symbol="BTC/USD",
            price=price,
            captured_at=NOW,
        )
    )
    risk_service = make_risk_service(
        limits=limits,
        control_store=control_store,
        account_provider=account_provider,
        price_provider=price_provider,
    )
    gateway = RecordingGateway()
    service = make_controlled_service(
        risk_service=risk_service,
        control_store=control_store,
        gateway=gateway,
    )

    result = service.submit_intent(intent)

    assert result.risk_decision.status is RiskDecisionStatus.REJECTED
    assert result.risk_decision.reason == reason
    assert gateway.requests == []


def test_dependency_failure_is_rejected_and_persisted(
    intent: StrategyIntentEvent,
    limits: LiveRiskLimits,
    account_provider: StaticAccountStateProvider,
    price_provider: StaticMarketPriceProvider,
) -> None:
    control_store = InMemoryControlStore()
    price_provider.raise_on_get = ConnectionError("market data unavailable")
    risk_service = make_risk_service(
        limits=limits,
        control_store=control_store,
        account_provider=account_provider,
        price_provider=price_provider,
    )
    gateway = RecordingGateway()
    service = make_controlled_service(
        risk_service=risk_service,
        control_store=control_store,
        gateway=gateway,
    )

    result = service.submit_intent(intent)

    assert result.risk_decision.reason == "DEPENDENCY_UNAVAILABLE:ConnectionError"
    assert len(control_store.decisions) == 1
    assert gateway.requests == []


def test_risk_decision_persistence_failure_blocks_submission(
    intent: StrategyIntentEvent,
    limits: LiveRiskLimits,
    account_provider: StaticAccountStateProvider,
    price_provider: StaticMarketPriceProvider,
) -> None:
    control_store = InMemoryControlStore()
    control_store.raise_on_save_decision = OSError("storage unavailable")
    risk_service = make_risk_service(
        limits=limits,
        control_store=control_store,
        account_provider=account_provider,
        price_provider=price_provider,
    )
    gateway = RecordingGateway()
    service = make_controlled_service(
        risk_service=risk_service,
        control_store=control_store,
        gateway=gateway,
    )

    with pytest.raises(RuntimeError, match="persistence failed"):
        service.submit_intent(intent)

    assert gateway.requests == []


def test_unknown_execution_requires_reconciliation(
    intent: StrategyIntentEvent,
    limits: LiveRiskLimits,
    account_provider: StaticAccountStateProvider,
    price_provider: StaticMarketPriceProvider,
) -> None:
    control_store = InMemoryControlStore()
    risk_service = make_risk_service(
        limits=limits,
        control_store=control_store,
        account_provider=account_provider,
        price_provider=price_provider,
    )
    gateway = RecordingGateway()
    gateway.raise_on_submit = TimeoutError("ambiguous acknowledgement")
    service = make_controlled_service(
        risk_service=risk_service,
        control_store=control_store,
        gateway=gateway,
    )

    result = service.submit_intent(intent)

    assert result.risk_decision.status is RiskDecisionStatus.APPROVED
    assert result.execution_result is not None
    assert result.execution_result.report.status is ExecutionStatus.UNKNOWN
    assert control_store.state.reconciliation_required is True
    assert len(control_store.saved_states) == 1
    assert len(gateway.requests) == 1
