"""Per-request ranking context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

from .config import Budget
from .errors import ConfigurationError

QualityMode = Literal["fast", "balanced", "quality", "offline"]


@dataclass(frozen=True, slots=True)
class RerankContext:
    request_id: str = field(default_factory=lambda: uuid4().hex)
    tenant: str = "default"
    authorization_revision: str = "default"
    as_of: datetime = field(default_factory=lambda: datetime.now(UTC))
    quality_mode: QualityMode = "balanced"
    budget: Budget = field(default_factory=Budget)
    tags: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ConfigurationError("as_of must be timezone-aware")
        if not self.request_id or not self.tenant:
            raise ConfigurationError("request_id and tenant cannot be empty")
        object.__setattr__(self, "as_of", self.as_of.astimezone(UTC))
        object.__setattr__(self, "tags", MappingProxyType(dict(self.tags)))
