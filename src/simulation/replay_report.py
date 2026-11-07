"""Read-only reporting for deterministic simulation replays."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.simulation.session import SimulationSession


ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """Immutable summary derived from durable SimulationSession state."""

    initial_cash_balance: Decimal
    cash_balance: Decimal
    active_quantity: Decimal
    vaulted_lot_count: int
    vaulted_quantity: Decimal
    campaign_active: bool
    cooldown_until: str | None


def build_replay_report(session: SimulationSession) -> ReplayReport:
    """Build a read-only replay summary without mutating session state."""
    if not isinstance(session, SimulationSession):
        raise TypeError("session must be a SimulationSession")

    base_asset = session.config.strategy_adapter.symbol.split("/", 1)[0]

    active_quantity = session.ledger_state.position_quantities.get(
        base_asset,
        ZERO,
    )

    vaulted_quantity = sum(
        (lot.quantity for lot in session.vaulted_lots),
        start=ZERO,
    )

    campaign = session.brawler_campaign

    return ReplayReport(
        initial_cash_balance=session.config.initial_cash_balance,
        cash_balance=session.ledger_state.cash_balance,
        active_quantity=active_quantity,
        vaulted_lot_count=len(session.vaulted_lots),
        vaulted_quantity=vaulted_quantity,
        campaign_active=campaign.is_active,
        cooldown_until=campaign.cooldown_until,
    )
