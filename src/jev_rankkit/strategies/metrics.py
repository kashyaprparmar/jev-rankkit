from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar, cast

from ..candidates import CandidateView
from ..context import RerankContext
from ..metrics import WeightedMetrics
from ..types import ExecutionStage, RankingOutcome, ScoreKind
from .base import EvaluationServices

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class MetricStrategy:
    metrics: WeightedMetrics[object]
    name: str = "weighted_metrics"

    async def rank(
        self,
        *,
        query: str,
        candidates: Sequence[CandidateView[T]],
        context: RerankContext,
        services: EvaluationServices,
        top_k: int | None,
    ) -> RankingOutcome:
        del context
        outcome = await services.metric_scores(
            query=query,
            candidates=cast(Sequence[CandidateView[object]], candidates),
            metrics=self.metrics,
        )
        indices = {candidate.occurrence_id: candidate.input_index for candidate in candidates}
        ordered = sorted(
            outcome.entries,
            key=lambda entry: (-(entry.score or 0.0), indices[entry.occurrence_id]),
        )
        if top_k is not None:
            ordered = ordered[:top_k]
        stage = ExecutionStage(self.name, self.name, len(candidates), len(ordered))
        return RankingOutcome(
            tuple(ordered), ScoreKind.UTILITY, outcome.approximate, outcome.warnings, (stage,)
        )
