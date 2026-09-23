"""Shallow, ordered ranking pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast

from .candidates import CandidateView
from .context import RerankContext
from .errors import ConfigurationError, RerankError
from .metrics import BM25Metric, EmbeddingSimilarity, Metric, WeightedMetrics
from .selection import mmr_select
from .strategies import EvaluationServices, ListwiseStrategy, MetricStrategy, PointwiseStrategy
from .types import ExecutionStage, RankingOutcome

T = TypeVar("T")


class PipelineStageError(RerankError):
    def __init__(self, stage: str, checkpoint: RankingOutcome, cause: RerankError) -> None:
        super().__init__(f"pipeline stage {stage!r} failed; previous checkpoint is available")
        self.checkpoint = checkpoint
        self.cause = cause


class PipelineStage(Protocol[T]):
    @property
    def name(self) -> str: ...

    async def run(
        self,
        *,
        query: str,
        candidates: Sequence[CandidateView[T]],
        context: RerankContext,
        services: EvaluationServices,
        previous: RankingOutcome | None,
    ) -> RankingOutcome: ...


def _selected_views(
    candidates: Sequence[CandidateView[T]], outcome: RankingOutcome
) -> tuple[CandidateView[T], ...]:
    by_id = {candidate.occurrence_id: candidate for candidate in candidates}
    return tuple(by_id[entry.occurrence_id] for entry in outcome.entries)


@dataclass(frozen=True, slots=True)
class CandidateFilter(Generic[T]):
    limit: int
    metric: Metric[T] | None = None
    name: str = "CandidateFilter"

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ConfigurationError("candidate filter limit must be non-negative")

    async def run(
        self,
        *,
        query: str,
        candidates: Sequence[CandidateView[T]],
        context: RerankContext,
        services: EvaluationServices,
        previous: RankingOutcome | None,
    ) -> RankingOutcome:
        del previous
        metric = self.metric or cast(Metric[T], BM25Metric())
        strategy = MetricStrategy(cast(WeightedMetrics[object], WeightedMetrics.equal((metric,))))
        outcome = await strategy.rank(
            query=query,
            candidates=candidates,
            context=context,
            services=services,
            top_k=self.limit,
        )
        stage = ExecutionStage(self.name, "metric_filter", len(candidates), len(outcome.entries))
        return RankingOutcome(
            outcome.entries,
            outcome.score_kind,
            len(outcome.entries) < len(candidates),
            outcome.warnings,
            (stage,),
        )


@dataclass(frozen=True, slots=True)
class EmbeddingReranker(Generic[T]):
    metric: EmbeddingSimilarity[T]
    limit: int
    name: str = "EmbeddingReranker"

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ConfigurationError("embedding stage limit must be non-negative")

    async def run(
        self,
        *,
        query: str,
        candidates: Sequence[CandidateView[T]],
        context: RerankContext,
        services: EvaluationServices,
        previous: RankingOutcome | None,
    ) -> RankingOutcome:
        del previous
        strategy = MetricStrategy(
            cast(WeightedMetrics[object], WeightedMetrics.equal((self.metric,)))
        )
        outcome = await strategy.rank(
            query=query, candidates=candidates, context=context, services=services, top_k=self.limit
        )
        stage = ExecutionStage(
            self.name,
            "embedding",
            len(candidates),
            len(outcome.entries),
            self.metric.backend.backend_id,
        )
        return RankingOutcome(
            outcome.entries,
            outcome.score_kind,
            len(outcome.entries) < len(candidates),
            outcome.warnings,
            (stage,),
        )


class BM25Filter(CandidateFilter[T]):
    def __init__(self, limit: int, metric: BM25Metric[T] | None = None) -> None:
        super().__init__(limit=limit, metric=metric or BM25Metric(), name="BM25Filter")


@dataclass(frozen=True, slots=True)
class JevReranker(Generic[T]):
    limit: int
    mode: str = "listwise"
    batch_size: int | None = None
    name: str = "JevReranker"

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ConfigurationError("model stage limit must be non-negative")
        if self.mode not in {"listwise", "pointwise"}:
            raise ConfigurationError("model stage mode must be pointwise or listwise")

    async def run(
        self,
        *,
        query: str,
        candidates: Sequence[CandidateView[T]],
        context: RerankContext,
        services: EvaluationServices,
        previous: RankingOutcome | None,
    ) -> RankingOutcome:
        del previous
        strategy = (
            ListwiseStrategy(self.batch_size) if self.mode == "listwise" else PointwiseStrategy()
        )
        outcome = await strategy.rank(
            query=query, candidates=candidates, context=context, services=services, top_k=self.limit
        )
        stage = ExecutionStage(
            self.name,
            self.mode,
            len(candidates),
            len(outcome.entries),
            approximate=outcome.approximate,
        )
        return RankingOutcome(
            outcome.entries, outcome.score_kind, outcome.approximate, outcome.warnings, (stage,)
        )


@dataclass(frozen=True, slots=True)
class DiversityReranker(Generic[T]):
    limit: int
    lambda_: float = 0.7
    name: str = "DiversityReranker"

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ConfigurationError("diversity stage limit must be non-negative")

    async def run(
        self,
        *,
        query: str,
        candidates: Sequence[CandidateView[T]],
        context: RerankContext,
        services: EvaluationServices,
        previous: RankingOutcome | None,
    ) -> RankingOutcome:
        del query, context, services
        if previous is None:
            raise ValueError("diversity stage requires a previous ranking")
        outcome = await mmr_select(previous, candidates, top_k=self.limit, lambda_=self.lambda_)
        stage = ExecutionStage(self.name, "mmr", len(candidates), len(outcome.entries))
        return RankingOutcome(
            outcome.entries, outcome.score_kind, outcome.approximate, outcome.warnings, (stage,)
        )


@dataclass(frozen=True, slots=True)
class RerankPipeline(Generic[T]):
    stages: tuple[PipelineStage[T], ...]
    name: str = "pipeline"

    def __init__(self, stages: Sequence[PipelineStage[T]]) -> None:
        if not stages:
            raise ValueError("pipeline requires at least one stage")
        object.__setattr__(self, "stages", tuple(stages))
        object.__setattr__(self, "name", "pipeline")

    async def rank(
        self,
        *,
        query: str,
        candidates: Sequence[CandidateView[T]],
        context: RerankContext,
        services: EvaluationServices,
        top_k: int | None,
    ) -> RankingOutcome:
        current = tuple(candidates)
        outcome: RankingOutcome | None = None
        executed: list[ExecutionStage] = []
        for stage in self.stages:
            if top_k is None and getattr(stage, "limit", len(current)) < len(current):
                raise ConfigurationError("full ordering cannot use a pruning pipeline stage")
            try:
                outcome = await stage.run(
                    query=query,
                    candidates=current,
                    context=context,
                    services=services,
                    previous=outcome,
                )
            except RerankError as exc:
                if outcome is None:
                    raise
                checkpoint = RankingOutcome(
                    outcome.entries,
                    outcome.score_kind,
                    outcome.approximate,
                    outcome.warnings,
                    tuple(executed),
                )
                raise PipelineStageError(stage.name, checkpoint, exc) from exc
            executed.extend(outcome.stages)
            current = _selected_views(current, outcome)
        assert outcome is not None
        entries = outcome.entries if top_k is None else outcome.entries[:top_k]
        return RankingOutcome(
            entries, outcome.score_kind, outcome.approximate, outcome.warnings, tuple(executed)
        )
