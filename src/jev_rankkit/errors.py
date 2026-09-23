"""Package-specific exception hierarchy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ErrorDetails:
    stage: str | None = None
    retryable: bool = False
    status_code: int | None = None
    safe_context: Mapping[str, Any] | None = None


class RerankError(Exception):
    """Base error with payload-safe diagnostics."""

    def __init__(self, message: str, *, details: ErrorDetails | None = None) -> None:
        super().__init__(message)
        self.details = details or ErrorDetails()


class ConfigurationError(RerankError):
    pass


class ProjectionError(RerankError):
    pass


class CapabilityError(RerankError):
    pass


class ContextLimitError(RerankError):
    pass


class BackendError(RerankError):
    pass


class AuthenticationError(BackendError):
    pass


class RateLimitError(BackendError):
    pass


class OutputValidationError(BackendError):
    pass


class BudgetExceededError(RerankError):
    pass


class BudgetUnverifiableError(BudgetExceededError):
    pass


class DeadlineExceededError(RerankError):
    pass


class ConstraintError(RerankError):
    pass


class CacheError(RerankError):
    pass


class EvaluationError(RerankError):
    pass
