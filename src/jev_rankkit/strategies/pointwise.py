from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar, cast

from ..candidates import CandidateView
from ..context import RerankContext
from ..types import ExecutionStage, RankingEntry, RankingOutcome, ScoreKind
from .base import EvaluationServices

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PointwiseStrategy:
    name: str = "pointwise"

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
        outcome = await services.model_scores(
            query=query,
            candidates=cast(Sequence[CandidateView[object]], candidates),
            mode="pointwise",
        )
        index = {candidate.occurrence_id: candidate.input_index for candidate in candidates}
        ordered = sorted(
            outcome.entries,
            key=lambda entry: (
                -(entry.score if entry.score is not None else float("-inf")),
                index[entry.occurrence_id],
            ),
        )
        if top_k is not None:
            ordered = ordered[:top_k]
        stage = ExecutionStage(self.name, self.name, len(candidates), len(ordered))
        return RankingOutcome(
            tuple(ordered), ScoreKind.UTILITY, outcome.approximate, outcome.warnings, (stage,)
        )


@dataclass(frozen=True, slots=True)
class LexicalStrategy:
    name: str = "lexical"

    async def rank(
        self,
        *,
        query: str,
        candidates: Sequence[CandidateView[T]],
        context: RerankContext,
        services: EvaluationServices,
        top_k: int | None,
    ) -> RankingOutcome:
        del services
        from ..metrics import LexicalRelevance

        metric: LexicalRelevance[T] = LexicalRelevance()
        scores = await metric.score_many(query, candidates, context)
        entries = [
            RankingEntry(candidate.occurrence_id, scores[candidate.occurrence_id])
            for candidate in candidates
        ]
        index = {candidate.occurrence_id: candidate.input_index for candidate in candidates}
        entries.sort(key=lambda entry: (-(entry.score or 0.0), index[entry.occurrence_id]))
        if top_k is not None:
            entries = entries[:top_k]
        stage = ExecutionStage(self.name, self.name, len(candidates), len(entries))
        return RankingOutcome(tuple(entries), ScoreKind.UTILITY, stages=(stage,))
