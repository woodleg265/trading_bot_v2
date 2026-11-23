from __future__ import annotations

from hashlib import sha256

from src.execution.models import StrategyIntentEvent


def derive_idempotency_key(intent: StrategyIntentEvent) -> str:
    """Derive a stable execution key from an immutable intent identity.

    The namespace avoids collisions with unrelated systems that may happen to
    use the same upstream identifier.
    """

    material = f"live-execution:v1:{intent.intent_id}".encode("utf-8")
    return sha256(material).hexdigest()
