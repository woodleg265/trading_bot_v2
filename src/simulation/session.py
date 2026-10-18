"""Application-level lifecycle owner for one isolated simulation run."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.simulation.audit_writer import SimulationAuditWriter
from src.simulation.fill_model import SimulationFillConfig, SimulationFillModel
from src.simulation.ledger import SimulationLedger, SimulationLedgerState
from src.simulation.risk_engine import (
    SimulationRiskContext,
    SimulationRiskEngine,
    SimulationRiskLimits,
)
from src.simulation.runner import SimulationRunner
from src.simulation.strategy_adapter import (
    SimulationStrategyAdapter,
    SimulationStrategyAdapterConfig,
    SimulationStrategyAdapterEvent,
    SimulationStrategyAdapterResult,
)
from src.strategies.ema_regime_strategy import StrategyDecision


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
    """Own the services and in-memory lifecycle for one virtual trading run.

    This class is intentionally simulation-only. It constructs the deterministic
    runner pipeline and routes strategy decisions through SimulationStrategyAdapter.
    It has no broker, exchange, production-order, or legacy ledger dependency.
    """

    def __init__(
        self,
        config: SimulationSessionConfig,
        *,
        initial_ledger_state: SimulationLedgerState | None = None,
        initial_risk_context: SimulationRiskContext | None = None,
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

        self._audit_writer = SimulationAuditWriter()
        self._runner = SimulationRunner(
            risk_engine=SimulationRiskEngine(config.risk_limits),
            fill_model=SimulationFillModel(config.fill_config),
            ledger=SimulationLedger(),
            audit_writer=self._audit_writer,
        )
        self._adapter = SimulationStrategyAdapter(
            runner=self._runner,
            config=config.strategy_adapter,
            state=state,
            risk_context=risk_context,
            run_id=config.run_id,
        )

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

    def process_decision(
        self,
        decision: StrategyDecision,
    ) -> SimulationStrategyAdapterResult:
        """Process one strategy decision through the virtual simulation pipeline."""

        return self._adapter.apply(decision)
