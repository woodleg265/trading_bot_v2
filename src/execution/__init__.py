"""Exchange-agnostic live execution domain boundary.

This package intentionally contains no exchange-client construction, credential
loading, strategy evaluation, or simulation imports.
"""

from src.execution.models import (
    ExchangeOrderRecord,
    ExecutionReport,
    ExecutionSide,
    ExecutionStatus,
    IdempotencyClaim,
    IdempotencyStatus,
    OrderSubmissionRequest,
    StrategyIntentEvent,
)
from src.execution.service import LiveExecutionService

__all__ = [
    "ExchangeOrderRecord",
    "ExecutionReport",
    "ExecutionSide",
    "ExecutionStatus",
    "IdempotencyClaim",
    "IdempotencyStatus",
    "LiveExecutionService",
    "OrderSubmissionRequest",
    "StrategyIntentEvent",
]
