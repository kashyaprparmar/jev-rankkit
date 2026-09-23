"""Metric contracts and weighted composition."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar, runtime_checkable

from ..candidates import CandidateView
from ..context import RerankContext
from ..errors import ConfigurationError, OutputValidationError

T = TypeVar("T")


@runtime_checkable
class Metric(Protocol[T]):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def cacheable(self) -> bool: ...

    @property
    def cache_identity(self) -> str: ...

    async def score(
        self, query: str, candidate: CandidateView[T], context: RerankContext
    ) -> float: ...


@runtime_checkable
class BatchMetric(Metric[T], Protocol[T]):
    async def score_many(
        self,
        query: str,
        candidates: Sequence[CandidateView[T]],
        context: RerankContext,
    ) -> Mapping[str, float]: ...


def validate_utility(value: float, *, metric_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OutputValidationError(f"metric {metric_name!r} did not return a number")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise OutputValidationError(f"metric {metric_name!r} must return a value in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class WeightedMetrics(Generic[T]):
    metrics: tuple[Metric[T], ...]
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ConfigurationError("at least one metric is required")
        names = [metric.name for metric in self.metrics]
        if len(set(names)) != len(names):
            raise ConfigurationError("metric names must be unique")
        if set(self.weights) != set(names):
            raise ConfigurationError("weights must exactly match metric names")
        if any(not math.isfinite(weight) or weight < 0 for weight in self.weights.values()):
            raise ConfigurationError("weights must be finite and non-negative")
        total = sum(self.weights.values())
        if total <= 0:
            raise ConfigurationError("at least one metric weight must be positive")
        object.__setattr__(
            self,
            "weights",
            MappingProxyType({name: self.weights[name] / total for name in names}),
        )

    @classmethod
    def equal(cls, metrics: Sequence[Metric[T]]) -> WeightedMetrics[T]:
        items = tuple(metrics)
        return cls(items, {metric.name: 1.0 for metric in items})

    def combine(self, scores: Mapping[str, float]) -> float:
        active = {name for name, weight in self.weights.items() if weight > 0}
        if not active <= set(scores) or not set(scores) <= set(self.weights):
            raise OutputValidationError("metric score coverage does not match configured weights")
        return sum(
            validate_utility(scores[name], metric_name=name) * weight
            for name, weight in self.weights.items()
            if weight > 0
        )

    @property
    def fingerprint(self) -> str:
        parts = [
            f"{metric.cache_identity}:{self.weights[metric.name]:.17g}" for metric in self.metrics
        ]
        return "|".join(parts)
