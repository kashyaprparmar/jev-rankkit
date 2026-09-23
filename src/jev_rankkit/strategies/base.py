"""Strategy protocol and execution-service contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

from ..candidates import CandidateView
from ..context import RerankContext
from ..metrics import WeightedMetrics
from ..types import RankingOutcome

T = TypeVar("T")


@runtime_checkable
class EvaluationServices(Protocol):
    async def model_scores(
        self,
        *,
        query: str,
        candidates: Sequence[CandidateView[object]],
        mode: str,
        allow_partial: bool = False,
    ) -> RankingOutcome: ...

    async def metric_scores(
        self,
        *,
        query: str,
        candidates: Sequence[CandidateView[object]],
        metrics: WeightedMetrics[object],
    ) -> RankingOutcome: ...


@runtime_checkable
class RankingStrategy(Protocol[T]):
    @property
    def name(self) -> str: ...

    async def rank(
        self,
        *,
        query: str,
        candidates: Sequence[CandidateView[T]],
        context: RerankContext,
        services: EvaluationServices,
        top_k: int | None,
    ) -> RankingOutcome: ...
