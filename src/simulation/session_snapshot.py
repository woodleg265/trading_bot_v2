"""Versioned JSON persistence for SimulationSession state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any, Mapping

from src.simulation.fill_model import SimulationFillConfig
from src.simulation.ledger import SimulationLedgerState
from src.simulation.models import VaultedLot
from src.simulation.risk_engine import SimulationRiskContext, SimulationRiskLimits
from src.simulation.session import SimulationSession, SimulationSessionConfig
from src.simulation.strategy_adapter import SimulationStrategyAdapterConfig
from src.strategies.brawler_strategy import BrawlerCampaignState


SNAPSHOT_VERSION = 2
ZERO = Decimal("0")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")

    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")

    return result


def _optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _float(value, name)


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


def _optional_decimal(value: object, name: str) -> Decimal | None:
    if value is None:
        return None
    return _decimal(value, name)


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


def _timestamp(value: object, name: str) -> datetime:
    raw_timestamp = _string(value, name)

    try:
        parsed = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be a valid ISO-8601 timestamp") from error

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include timezone information")

    return parsed.astimezone(timezone.utc)


def _optional_timestamp(value: object, name: str) -> datetime | None:
    if value is None:
        return None
    return _timestamp(value, name)


def _timestamp_json(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include timezone information")

    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _vaulted_lot_to_dict(lot: VaultedLot) -> dict[str, str]:
    return {
        "symbol": lot.symbol,
        "quantity": str(lot.quantity),
        "acquisition_price": str(lot.acquisition_price),
        "acquired_at": _timestamp_json(lot.acquired_at),
        "vaulted_at": _timestamp_json(lot.vaulted_at),
    }


def _vaulted_lot_from_dict(value: object, name: str) -> VaultedLot:
    payload = _mapping(value, name)
    _exact_keys(
        payload,
        {
            "symbol",
            "quantity",
            "acquisition_price",
            "acquired_at",
            "vaulted_at",
        },
        name,
    )

    return VaultedLot(
        symbol=_string(payload["symbol"], f"{name}.symbol"),
        quantity=_decimal(payload["quantity"], f"{name}.quantity"),
        acquisition_price=_decimal(
            payload["acquisition_price"],
            f"{name}.acquisition_price",
        ),
        acquired_at=_timestamp(payload["acquired_at"], f"{name}.acquired_at"),
        vaulted_at=_timestamp(payload["vaulted_at"], f"{name}.vaulted_at"),
    )


def _vaulted_lots_from_json(value: object) -> tuple[VaultedLot, ...]:
    if not isinstance(value, list):
        raise ValueError("vaulted_lots must be a list")

    return tuple(
        _vaulted_lot_from_dict(raw_lot, f"vaulted_lots[{index}]")
        for index, raw_lot in enumerate(value)
    )


def _brawler_campaign_to_dict(
    campaign: BrawlerCampaignState,
) -> dict[str, float | str | None]:
    return {
        "entry_price": campaign.entry_price,
        "entry_timestamp": campaign.entry_timestamp,
        "rsi_peak": campaign.rsi_peak,
        "cooldown_until": campaign.cooldown_until,
    }


def _brawler_campaign_from_dict(value: object) -> BrawlerCampaignState:
    payload = _mapping(value, "brawler_campaign")
    _exact_keys(
        payload,
        {
            "entry_price",
            "entry_timestamp",
            "rsi_peak",
            "cooldown_until",
        },
        "brawler_campaign",
    )

    entry_price = _optional_float(
        payload["entry_price"],
        "brawler_campaign.entry_price",
    )
    entry_timestamp = _optional_string(
        payload["entry_timestamp"],
        "brawler_campaign.entry_timestamp",
    )
    rsi_peak = _optional_float(
        payload["rsi_peak"],
        "brawler_campaign.rsi_peak",
    )
    cooldown_until = _optional_string(
        payload["cooldown_until"],
        "brawler_campaign.cooldown_until",
    )

    try:
        return BrawlerCampaignState(
            entry_price=entry_price,
            entry_timestamp=entry_timestamp,
            rsi_peak=rsi_peak,
            cooldown_until=cooldown_until,
        )
    except ValueError as error:
        raise ValueError(f"brawler_campaign is invalid: {error}") from error


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
    vaulted_lots: tuple[VaultedLot, ...]
    active_entry_price: Decimal | None
    active_entry_timestamp: datetime | None
    brawler_campaign: BrawlerCampaignState

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

        if not all(isinstance(lot, VaultedLot) for lot in self.vaulted_lots):
            raise TypeError("vaulted_lots must contain VaultedLot values")

        if (self.active_entry_price is None) != (
            self.active_entry_timestamp is None
        ):
            raise ValueError(
                "active_entry_price and active_entry_timestamp must either "
                "both be present or both be absent"
            )

        if self.active_entry_price is not None:
            if (
                not self.active_entry_price.is_finite()
                or self.active_entry_price <= ZERO
            ):
                raise ValueError(
                    "active_entry_price must be finite and greater than zero"
                )

        if self.active_entry_timestamp is not None:
            if (
                self.active_entry_timestamp.tzinfo is None
                or self.active_entry_timestamp.utcoffset() is None
            ):
                raise ValueError(
                    "active_entry_timestamp must be timezone-aware"
                )

        if not isinstance(self.brawler_campaign, BrawlerCampaignState):
            raise TypeError("brawler_campaign must be a BrawlerCampaignState")

        configured_symbol = self.config.strategy_adapter.symbol
        base_asset = configured_symbol.split("/", maxsplit=1)[0].strip().upper()

        ledger_position_quantity = self.ledger_state.position_quantities.get(
            base_asset,
            ZERO,
        )
        risk_position_quantity = self.risk_context.position_quantities.get(
            configured_symbol,
            ZERO,
        )

        if ledger_position_quantity != risk_position_quantity:
            raise ValueError(
                "ledger_state.position_quantities and "
                "risk_context.position_quantities are inconsistent for "
                f"{configured_symbol}: "
                f"ledger={ledger_position_quantity}, "
                f"risk={risk_position_quantity}"
            )

        has_open_position = ledger_position_quantity > ZERO
        has_active_entry = self.active_entry_price is not None

        if has_open_position and not has_active_entry:
            raise ValueError(
                "v2 snapshots require active entry metadata for an open "
                "configured-symbol position"
            )

        if not has_open_position and has_active_entry:
            raise ValueError(
                "active entry metadata cannot be present when the configured "
                "symbol has no open virtual position"
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
            vaulted_lots=session.vaulted_lots,
            active_entry_price=session.active_entry_price,
            active_entry_timestamp=session.active_entry_timestamp,
            brawler_campaign=session.brawler_campaign,
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
            initial_vaulted_lots=self.vaulted_lots,
            initial_active_entry_price=self.active_entry_price,
            initial_active_entry_timestamp=self.active_entry_timestamp,
            initial_brawler_campaign=self.brawler_campaign,
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
            "vaulted_lots": [
                _vaulted_lot_to_dict(lot)
                for lot in self.vaulted_lots
            ],
            "active_entry_price": (
                str(self.active_entry_price)
                if self.active_entry_price is not None
                else None
            ),
            "active_entry_timestamp": (
                _timestamp_json(self.active_entry_timestamp)
                if self.active_entry_timestamp is not None
                else None
            ),
            "brawler_campaign": _brawler_campaign_to_dict(
                self.brawler_campaign
            ),
        }

    def to_json(self) -> str:
        """Serialize deterministically for storage."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: object) -> "SimulationSessionSnapshot":
        """Validate and deserialize a decoded v2 snapshot object."""

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
                "vaulted_lots",
                "active_entry_price",
                "active_entry_timestamp",
                "brawler_campaign",
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
            isinstance(key, str) and key
            for key in decision_keys
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
            vaulted_lots=_vaulted_lots_from_json(payload["vaulted_lots"]),
            active_entry_price=_optional_decimal(
                payload["active_entry_price"],
                "active_entry_price",
            ),
            active_entry_timestamp=_optional_timestamp(
                payload["active_entry_timestamp"],
                "active_entry_timestamp",
            ),
            brawler_campaign=_brawler_campaign_from_dict(
                payload["brawler_campaign"]
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
