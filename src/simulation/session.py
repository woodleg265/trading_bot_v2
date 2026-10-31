"""Application-level lifecycle owner for one isolated simulation run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from src.simulation.audit_writer import SimulationAuditWriter
from src.simulation.fill_model import SimulationFillConfig, SimulationFillModel
from src.simulation.ledger import SimulationLedger, SimulationLedgerState
from src.simulation.models import VaultedLot
from src.simulation.risk_engine import (
    SimulationRiskContext,
    SimulationRiskEngine,
    SimulationRiskLimits,
)
from src.simulation.runner import SimulationRunner
from src.simulation.staged_audit_writer import StagedSimulationAuditWriter
from src.simulation.strategy_adapter import (
    SimulationAdapterOutcome,
    SimulationStrategyAdapter,
    SimulationStrategyAdapterConfig,
    SimulationStrategyAdapterEvent,
    SimulationStrategyAdapterResult,
)
from src.strategies.brawler_strategy import (
    BrawlerCampaignState,
    BrawlerEvaluation,
)
from src.strategies.ema_regime_strategy import DecisionAction, StrategyDecision


ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class SimulationSessionConfig:
    """Configuration required to initialize one isolated simulation session."""

    run_id: str
    strategy_adapter: SimulationStrategyAdapterConfig
    risk_limits: SimulationRiskLimits
    fill_config: SimulationFillConfig
    initial_cash_balance: Decimal
    cash_currency: str

    def __post_init__(self) -> None:
        normalized_run_id = self.run_id.strip()
        normalized_currency = self.cash_currency.strip().upper()

        if not normalized_run_id:
            raise ValueError("run_id is required")
        if ":" in normalized_run_id or any(char.isspace() for char in normalized_run_id):
            raise ValueError("run_id must not contain whitespace or ':'")
        if self.initial_cash_balance < ZERO:
            raise ValueError("initial_cash_balance cannot be negative")
        if not normalized_currency:
            raise ValueError("cash_currency is required")
        if self.strategy_adapter.symbol not in self.risk_limits.allowed_symbols:
            raise ValueError(
                "strategy adapter symbol must be included in risk allowed_symbols"
            )

        object.__setattr__(self, "run_id", normalized_run_id)
        object.__setattr__(self, "cash_currency", normalized_currency)


class SimulationSession:
    """Own services and in-memory lifecycle for one virtual trading run.

    This class is intentionally simulation-only. It constructs the deterministic
    runner pipeline and routes strategy decisions through SimulationStrategyAdapter.
    It has no broker, exchange, production-order, or legacy ledger dependency.

    Brawler campaign state belongs to the session rather than the strategy
    adapter. The adapter owns execution facts; BrawlerCampaignState owns the
    strategy's independent decision-making state.
    """

    def __init__(
        self,
        config: SimulationSessionConfig,
        *,
        initial_ledger_state: SimulationLedgerState | None = None,
        initial_risk_context: SimulationRiskContext | None = None,
        initial_vaulted_lots: tuple[VaultedLot, ...] = (),
        initial_active_entry_price: Decimal | None = None,
        initial_active_entry_timestamp: datetime | None = None,
        initial_brawler_campaign: BrawlerCampaignState | None = None,
    ) -> None:
        self._config = config

        state = initial_ledger_state or SimulationLedgerState(
            cash_balance=config.initial_cash_balance,
            cash_currency=config.cash_currency,
        )

        if initial_risk_context is None:
            if state.position_quantities:
                raise ValueError(
                    "initial_risk_context is required when restoring "
                    "a ledger state with open positions"
                )

            risk_context = SimulationRiskContext(
                available_cash=state.cash_balance,
                position_quantities={},
                current_total_exposure=ZERO,
                processed_intent_ids=frozenset(),
            )
        else:
            risk_context = initial_risk_context

        if (initial_active_entry_price is None) != (
            initial_active_entry_timestamp is None
        ):
            raise ValueError(
                "initial_active_entry_price and initial_active_entry_timestamp "
                "must either both be present or both be absent"
            )

        if initial_active_entry_price is not None:
            if (
                not initial_active_entry_price.is_finite()
                or initial_active_entry_price <= ZERO
            ):
                raise ValueError(
                    "initial_active_entry_price must be finite and greater than zero"
                )

        if (
            initial_active_entry_timestamp is not None
            and (
                initial_active_entry_timestamp.tzinfo is None
                or initial_active_entry_timestamp.utcoffset() is None
            )
        ):
            raise ValueError(
                "initial_active_entry_timestamp must be timezone-aware"
            )

        if not all(isinstance(lot, VaultedLot) for lot in initial_vaulted_lots):
            raise TypeError("initial_vaulted_lots must contain VaultedLot values")

        self._audit_writer = StagedSimulationAuditWriter(
            SimulationAuditWriter()
        )
        self._runner = SimulationRunner(
            risk_engine=SimulationRiskEngine(config.risk_limits),
            fill_model=SimulationFillModel(config.fill_config),
            ledger=SimulationLedger(),
            audit_writer=self._audit_writer,
        )
        self._adapter = SimulationStrategyAdapter(
            runner=self._runner,
            audit_writer=self._audit_writer,
            config=config.strategy_adapter,
            state=state,
            risk_context=risk_context,
            run_id=config.run_id,
        )

        self._validate_active_entry_restore_state(
            state=state,
            active_entry_price=initial_active_entry_price,
            active_entry_timestamp=initial_active_entry_timestamp,
        )

        # These are restoration-only state values. Normal runtime changes remain
        # encapsulated inside SimulationStrategyAdapter.
        self._adapter._vaulted_lots = list(initial_vaulted_lots)
        self._adapter._active_entry_price = initial_active_entry_price
        self._adapter._active_entry_timestamp = initial_active_entry_timestamp

        self._brawler_campaign = initial_brawler_campaign or BrawlerCampaignState()

    @property
    def config(self) -> SimulationSessionConfig:
        """Return immutable configuration for this simulation run."""

        return self._config

    @property
    def run_id(self) -> str:
        """Return the stable identifier for this simulation run."""

        return self._config.run_id

    @property
    def ledger_state(self) -> SimulationLedgerState:
        """Return the current immutable virtual-ledger snapshot."""

        return self._adapter.state

    @property
    def risk_context(self) -> SimulationRiskContext:
        """Return the current immutable virtual-risk snapshot."""

        return self._adapter.risk_context

    @property
    def events(self) -> tuple[SimulationStrategyAdapterEvent, ...]:
        """Return an immutable snapshot of session-level adapter events."""

        return self._adapter.events

    @property
    def processed_decision_keys(self) -> frozenset[str]:
        """Return strategy decisions already handled in this session."""

        return self._adapter.processed_decision_keys

    @property
    def vaulted_lots(self) -> tuple[VaultedLot, ...]:
        """Return inventory removed from active virtual management."""

        return self._adapter.vaulted_lots

    @property
    def active_entry_price(self) -> Decimal | None:
        """Return cost basis for the active virtual position, if any."""

        return self._adapter.active_entry_price

    @property
    def active_entry_timestamp(self) -> datetime | None:
        """Return acquisition time for the active virtual position, if any."""

        return self._adapter.active_entry_timestamp

    @property
    def brawler_campaign(self) -> BrawlerCampaignState:
        """Return persistent Brawler strategy state for this session."""

        return self._brawler_campaign

    def process_decision(
        self,
        decision: StrategyDecision,
    ) -> SimulationStrategyAdapterResult:
        """Process one EMA-compatible decision through virtual simulation."""

        return self._adapter.apply(decision)

    def process_brawler_evaluation(
        self,
        evaluation: BrawlerEvaluation,
    ) -> SimulationStrategyAdapterResult:
        """Apply a Brawler evaluation and conditionally persist campaign state.

        The campaign proposal is committed for HOLD and BLOCKED evaluations,
        because they require no executable trade. For ENTER, EXIT, and VAULT,
        it is committed only when the adapter successfully executes the
        transition. In particular, an ENTER rejected by risk controls leaves
        Brawler campaign state unchanged, preventing a false active campaign.
        """

        if not isinstance(evaluation, BrawlerEvaluation):
            raise TypeError("evaluation must be a BrawlerEvaluation")

        result = self._adapter.apply(evaluation.decision)

        if self._should_commit_brawler_campaign(
            action=evaluation.decision.action,
            outcome=result.outcome,
        ):
            self._brawler_campaign = evaluation.next_campaign

        return result

    @staticmethod
    def _should_commit_brawler_campaign(
        *,
        action: DecisionAction,
        outcome: SimulationAdapterOutcome,
    ) -> bool:
        """Return whether Brawler's proposed next campaign is durable."""

        if action in (DecisionAction.HOLD, DecisionAction.BLOCKED):
            return outcome is SimulationAdapterOutcome.NO_ACTION

        if action in (
            DecisionAction.ENTER,
            DecisionAction.EXIT,
            DecisionAction.VAULT,
        ):
            return outcome is SimulationAdapterOutcome.EXECUTED

        return False

    def _validate_active_entry_restore_state(
        self,
        *,
        state: SimulationLedgerState,
        active_entry_price: Decimal | None,
        active_entry_timestamp: datetime | None,
    ) -> None:
        """Validate supplied active-entry metadata against open inventory.

        Direct callers may restore legacy execution state that predates active
        entry tracking. Therefore, an open position with no supplied metadata
        remains valid here. The v2 snapshot deserializer is responsible for
        requiring the metadata fields during snapshot restoration.
        """

        has_active_metadata = (
            active_entry_price is not None
            and active_entry_timestamp is not None
        )

        # Constructor validation already rejects a partially supplied pair.
        # No metadata permits compatibility with legacy direct restores.
        if not has_active_metadata:
            return

        base_asset = self._config.strategy_adapter.symbol.split("/", maxsplit=1)[0]
        position_quantity = state.position_quantities.get(base_asset, ZERO)

        if position_quantity <= ZERO:
            raise ValueError(
                "active entry metadata cannot be present when the configured "
                "symbol has no open virtual position"
            )
