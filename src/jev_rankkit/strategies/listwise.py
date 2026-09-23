from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar, cast

from ..candidates import CandidateView
from ..context import RerankContext
from ..errors import CapabilityError
from ..types import ExecutionStage, RankingEntry, RankingOutcome, ScoreKind
from .base import EvaluationServices

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ListwiseStrategy:
    batch_size: int | None = None
    allow_partial: bool = False
    name: str = "listwise"

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
        if self.batch_size is not None and self.batch_size < 1:
            raise CapabilityError("listwise batch_size must be positive")
        size = self.batch_size or len(candidates) or 1
        if len(candidates) > size:
            raise CapabilityError(
                "independent listwise chunks cannot be merged without score calibration; "
                "use pointwise ranking or a single listwise group"
            )
        entries: list[RankingEntry] = []
        warnings: list[str] = []
        for start in range(0, len(candidates), size):
            batch = candidates[start : start + size]
            outcome = await services.model_scores(
                query=query,
                candidates=cast(Sequence[CandidateView[object]], batch),
                mode="listwise",
                allow_partial=self.allow_partial,
            )
            entries.extend(outcome.entries)
            warnings.extend(outcome.warnings)
        index = {candidate.occurrence_id: candidate.input_index for candidate in candidates}
        entries.sort(
            key=lambda entry: (
                -(entry.score if entry.score is not None else float("-inf")),
                index[entry.occurrence_id],
            )
        )
        if top_k is not None:
            entries = entries[:top_k]
        stage = ExecutionStage(
            self.name,
            self.name,
            len(candidates),
            len(entries),
        )
        return RankingOutcome(tuple(entries), ScoreKind.UTILITY, False, tuple(warnings), (stage,))
