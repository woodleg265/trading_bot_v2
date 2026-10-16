"""Versioned JSON persistence for SimulationSession state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from src.simulation.fill_model import SimulationFillConfig
from src.simulation.ledger import SimulationLedgerState
from src.simulation.risk_engine import SimulationRiskContext, SimulationRiskLimits
from src.simulation.session import SimulationSession, SimulationSessionConfig
from src.simulation.strategy_adapter import SimulationStrategyAdapterConfig


SNAPSHOT_VERSION = 1


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _decimal(value: object, name: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a decimal string")

    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be a valid decimal string") from error

    if not result.is_finite():
        raise ValueError(f"{name} must be finite")

    return result


def _decimal_map(value: object, name: str) -> dict[str, Decimal]:
    raw_values = _mapping(value, name)
    result: dict[str, Decimal] = {}

    for key, decimal_value in raw_values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings")
        result[key] = _decimal(decimal_value, f"{name}.{key}")

    return result


def _decimal_map_json(values: Mapping[str, Decimal]) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in sorted(values.items())
    }


def _exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    missing = expected - set(payload)
    unexpected = set(payload) - expected

    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"{name} has unexpected fields: {sorted(unexpected)}")


@dataclass(frozen=True, slots=True)
class SimulationSessionSnapshot:
    """All deterministic state needed to resume one simulation session."""

    version: int
    config: SimulationSessionConfig
    ledger_state: SimulationLedgerState
    risk_context: SimulationRiskContext
    processed_decision_keys: frozenset[str]
    event_sequence: int
    runner_sequence: int

    def __post_init__(self) -> None:
        if self.version != SNAPSHOT_VERSION:
            raise ValueError(
                f"unsupported simulation session snapshot version: {self.version}"
            )

        if self.ledger_state.cash_currency != self.config.cash_currency:
            raise ValueError(
                "ledger_state.cash_currency must match config.cash_currency"
            )

        if self.risk_context.available_cash != self.ledger_state.cash_balance:
            raise ValueError(
                "risk_context.available_cash must match ledger_state.cash_balance"
            )

        if self.event_sequence < 0:
            raise ValueError("event_sequence cannot be negative")

        if self.runner_sequence <= 0:
            raise ValueError("runner_sequence must be greater than zero")

        for key in self.processed_decision_keys:
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "processed_decision_keys must contain non-empty strings"
                )

    @classmethod
    def from_session(
        cls,
        session: SimulationSession,
    ) -> "SimulationSessionSnapshot":
        """Capture the current resumable state of a simulation session."""

        if not isinstance(session, SimulationSession):
            raise TypeError("session must be a SimulationSession")

        adapter = session._adapter

        return cls(
            version=SNAPSHOT_VERSION,
            config=session.config,
            ledger_state=session.ledger_state,
            risk_context=session.risk_context,
            processed_decision_keys=session.processed_decision_keys,
            event_sequence=adapter._event_sequence,
            runner_sequence=adapter._runner_sequence,
        )

    @classmethod
    def capture(
        cls,
        session: SimulationSession,
    ) -> "SimulationSessionSnapshot":
        """Alias for from_session."""

        return cls.from_session(session)

    def restore_session(self) -> SimulationSession:
        """Create a new SimulationSession initialized with this snapshot state."""

        session = SimulationSession(
            self.config,
            initial_ledger_state=self.ledger_state,
            initial_risk_context=self.risk_context,
        )

        adapter = session._adapter
        adapter._processed_decision_keys = set(self.processed_decision_keys)
        adapter._event_sequence = self.event_sequence
        adapter._runner_sequence = self.runner_sequence

        return session

    def restore(self) -> SimulationSession:
        """Alias for restore_session."""

        return self.restore_session()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this snapshot."""

        config = self.config
        limits = config.risk_limits

        return {
            "version": self.version,
            "config": {
                "run_id": config.run_id,
                "strategy_adapter": {
                    "symbol": config.strategy_adapter.symbol,
                    "entry_quantity": str(config.strategy_adapter.entry_quantity),
                    "max_exit_quantity": str(
                        config.strategy_adapter.max_exit_quantity
                    ),
                },
                "risk_limits": {
                    "allowed_symbols": sorted(limits.allowed_symbols),
                    "max_order_notional": str(limits.max_order_notional),
                    "max_position_notional": str(limits.max_position_notional),
                    "max_total_exposure": str(limits.max_total_exposure),
                    "max_market_data_age_microseconds": (
                        limits.max_market_data_age.days * 86_400_000_000
                        + limits.max_market_data_age.seconds * 1_000_000
                        + limits.max_market_data_age.microseconds
                    ),
                    "estimated_fee_rate": str(limits.estimated_fee_rate),
                },
                "fill_config": {
                    "fee_rate": str(config.fill_config.fee_rate),
                    "slippage_rate": str(config.fill_config.slippage_rate),
                },
                "initial_cash_balance": str(config.initial_cash_balance),
                "cash_currency": config.cash_currency,
            },
            "ledger_state": {
                "cash_balance": str(self.ledger_state.cash_balance),
                "cash_currency": self.ledger_state.cash_currency,
                "position_quantities": _decimal_map_json(
                    self.ledger_state.position_quantities
                ),
            },
            "risk_context": {
                "available_cash": str(self.risk_context.available_cash),
                "position_quantities": _decimal_map_json(
                    self.risk_context.position_quantities
                ),
                "current_total_exposure": str(
                    self.risk_context.current_total_exposure
                ),
                "processed_intent_ids": sorted(
                    self.risk_context.processed_intent_ids
                ),
            },
            "processed_decision_keys": sorted(self.processed_decision_keys),
            "event_sequence": self.event_sequence,
            "runner_sequence": self.runner_sequence,
        }

    def to_json(self) -> str:
        """Serialize deterministically for storage."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, value: object) -> "SimulationSessionSnapshot":
        """Validate and deserialize a decoded snapshot object."""

        payload = _mapping(value, "snapshot")
        _exact_keys(
            payload,
            {
                "version",
                "config",
                "ledger_state",
                "risk_context",
                "processed_decision_keys",
                "event_sequence",
                "runner_sequence",
            },
            "snapshot",
        )

        version = _integer(payload["version"], "version", minimum=1)
        if version != SNAPSHOT_VERSION:
            raise ValueError(
                f"unsupported simulation session snapshot version: {version}"
            )

        raw_config = _mapping(payload["config"], "config")
        _exact_keys(
            raw_config,
            {
                "run_id",
                "strategy_adapter",
                "risk_limits",
                "fill_config",
                "initial_cash_balance",
                "cash_currency",
            },
            "config",
        )

        raw_adapter = _mapping(
            raw_config["strategy_adapter"],
            "config.strategy_adapter",
        )
        _exact_keys(
            raw_adapter,
            {"symbol", "entry_quantity", "max_exit_quantity"},
            "config.strategy_adapter",
        )

        raw_limits = _mapping(raw_config["risk_limits"], "config.risk_limits")
        _exact_keys(
            raw_limits,
            {
                "allowed_symbols",
                "max_order_notional",
                "max_position_notional",
                "max_total_exposure",
                "max_market_data_age_microseconds",
                "estimated_fee_rate",
            },
            "config.risk_limits",
        )

        allowed_symbols = raw_limits["allowed_symbols"]
        if (
            not isinstance(allowed_symbols, list)
            or not allowed_symbols
            or not all(
                isinstance(symbol, str) and symbol.strip()
                for symbol in allowed_symbols
            )
        ):
            raise ValueError(
                "config.risk_limits.allowed_symbols must be a non-empty string list"
            )

        raw_fill = _mapping(raw_config["fill_config"], "config.fill_config")
        _exact_keys(
            raw_fill,
            {"fee_rate", "slippage_rate"},
            "config.fill_config",
        )

        config = SimulationSessionConfig(
            run_id=_string(raw_config["run_id"], "config.run_id"),
            strategy_adapter=SimulationStrategyAdapterConfig(
                symbol=_string(
                    raw_adapter["symbol"],
                    "config.strategy_adapter.symbol",
                ),
                entry_quantity=_decimal(
                    raw_adapter["entry_quantity"],
                    "config.strategy_adapter.entry_quantity",
                ),
                max_exit_quantity=_decimal(
                    raw_adapter["max_exit_quantity"],
                    "config.strategy_adapter.max_exit_quantity",
                ),
            ),
            risk_limits=SimulationRiskLimits(
                allowed_symbols=frozenset(allowed_symbols),
                max_order_notional=_decimal(
                    raw_limits["max_order_notional"],
                    "config.risk_limits.max_order_notional",
                ),
                max_position_notional=_decimal(
                    raw_limits["max_position_notional"],
                    "config.risk_limits.max_position_notional",
                ),
                max_total_exposure=_decimal(
                    raw_limits["max_total_exposure"],
                    "config.risk_limits.max_total_exposure",
                ),
                max_market_data_age=timedelta(
                    microseconds=_integer(
                        raw_limits["max_market_data_age_microseconds"],
                        "config.risk_limits.max_market_data_age_microseconds",
                    )
                ),
                estimated_fee_rate=_decimal(
                    raw_limits["estimated_fee_rate"],
                    "config.risk_limits.estimated_fee_rate",
                ),
            ),
            fill_config=SimulationFillConfig(
                fee_rate=_decimal(
                    raw_fill["fee_rate"],
                    "config.fill_config.fee_rate",
                ),
                slippage_rate=_decimal(
                    raw_fill["slippage_rate"],
                    "config.fill_config.slippage_rate",
                ),
            ),
            initial_cash_balance=_decimal(
                raw_config["initial_cash_balance"],
                "config.initial_cash_balance",
            ),
            cash_currency=_string(
                raw_config["cash_currency"],
                "config.cash_currency",
            ),
        )

        raw_ledger = _mapping(payload["ledger_state"], "ledger_state")
        _exact_keys(
            raw_ledger,
            {"cash_balance", "cash_currency", "position_quantities"},
            "ledger_state",
        )
        ledger_state = SimulationLedgerState(
            cash_balance=_decimal(
                raw_ledger["cash_balance"],
                "ledger_state.cash_balance",
            ),
            cash_currency=_string(
                raw_ledger["cash_currency"],
                "ledger_state.cash_currency",
            ),
            position_quantities=_decimal_map(
                raw_ledger["position_quantities"],
                "ledger_state.position_quantities",
            ),
        )

        raw_risk = _mapping(payload["risk_context"], "risk_context")
        _exact_keys(
            raw_risk,
            {
                "available_cash",
                "position_quantities",
                "current_total_exposure",
                "processed_intent_ids",
            },
            "risk_context",
        )

        intent_ids = raw_risk["processed_intent_ids"]
        if not isinstance(intent_ids, list) or not all(
            isinstance(intent_id, str) and intent_id
            for intent_id in intent_ids
        ):
            raise ValueError(
                "risk_context.processed_intent_ids must be a list of non-empty strings"
            )

        risk_context = SimulationRiskContext(
            available_cash=_decimal(
                raw_risk["available_cash"],
                "risk_context.available_cash",
            ),
            position_quantities=_decimal_map(
                raw_risk["position_quantities"],
                "risk_context.position_quantities",
            ),
            current_total_exposure=_decimal(
                raw_risk["current_total_exposure"],
                "risk_context.current_total_exposure",
            ),
            processed_intent_ids=frozenset(intent_ids),
        )

        decision_keys = payload["processed_decision_keys"]
        if not isinstance(decision_keys, list) or not all(
            isinstance(key, str) and key for key in decision_keys
        ):
            raise ValueError(
                "processed_decision_keys must be a list of non-empty strings"
            )

        return cls(
            version=version,
            config=config,
            ledger_state=ledger_state,
            risk_context=risk_context,
            processed_decision_keys=frozenset(decision_keys),
            event_sequence=_integer(payload["event_sequence"], "event_sequence"),
            runner_sequence=_integer(
                payload["runner_sequence"],
                "runner_sequence",
                minimum=1,
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> "SimulationSessionSnapshot":
        """Validate and deserialize snapshot JSON."""

        if not isinstance(value, str):
            raise TypeError("snapshot JSON must be a string")

        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("snapshot JSON is invalid") from error

        return cls.from_dict(decoded)
