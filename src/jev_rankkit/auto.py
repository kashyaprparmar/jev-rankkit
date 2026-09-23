"""Inspectable, deterministic automatic strategy selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from ._runtime.executor import effective_budget, estimate_request_tokens
from .backends import BackendCandidate, EmbeddingBackend, ModelBackend, ModelRequest
from .backends.base import BackendMode
from .candidates import CandidateView
from .config import RerankerConfig
from .context import RerankContext
from .metrics import EmbeddingSimilarity
from .pipeline import (
    BM25Filter,
    EmbeddingReranker,
    JevReranker,
    PipelineStage,
    RerankPipeline,
)
from .prompts import DEFAULT_PROMPT, RerankPrompt
from .strategies import (
    EvaluationServices,
    LexicalStrategy,
    ListwiseStrategy,
    PointwiseStrategy,
    RankingStrategy,
)
from .types import ExecutionPlan, ExecutionStage, RankingOutcome

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class AutoDecision:
    strategy: RankingStrategy[Any]
    plan: ExecutionPlan


@dataclass(frozen=True, slots=True)
class AutoStrategy:
    backend: ModelBackend | None
    embedding_backend: EmbeddingBackend | None
    config: RerankerConfig
    name: str = "auto"

    def decide(
        self,
        candidates: Sequence[CandidateView[T]],
        *,
        query: str,
        top_k: int | None,
        context: RerankContext,
        prompt: RerankPrompt = DEFAULT_PROMPT,
    ) -> AutoDecision:
        count = len(candidates)
        average_chars = sum(len(candidate.text) for candidate in candidates) / max(count, 1)
        requested = count if top_k is None else min(top_k, count)
        rationale = [
            f"candidate_count={count}",
            f"average_candidate_chars={average_chars:.1f}",
            f"top_k={requested}",
            f"quality_mode={context.quality_mode}",
        ]
        budget = effective_budget(self.config.budget, context.budget)
        attempts = max(1, getattr(getattr(self.backend, "retry", None), "max_attempts", 1))
        available_calls = (
            None if budget.max_model_calls is None else budget.max_model_calls // attempts
        )
        no_model_budget = (
            available_calls == 0
            or budget.max_tokens == 0
            or budget.max_cost_usd == 0
            or budget.max_latency_ms == 0
        )
        unverifiable_strict_cost = (
            budget.strict_cost
            and budget.max_cost_usd is not None
            and not callable(getattr(self.backend, "chargeable_cost_upper_bound", None))
        )
        if (
            context.quality_mode in {"offline", "fast"}
            or self.backend is None
            or no_model_budget
            or unverifiable_strict_cost
        ):
            strategy: RankingStrategy[Any] = LexicalStrategy()
            if context.quality_mode == "fast":
                rationale.append("fast mode selected deterministic lexical ranking")
            elif no_model_budget:
                rationale.append("model budget prohibits model calls")
            elif unverifiable_strict_cost:
                rationale.append("strict cost budget lacks a chargeable-cost upper bound")
            else:
                rationale.append(
                    "selected deterministic lexical ranking because no remote model is allowed"
                )
            return AutoDecision(
                strategy,
                ExecutionPlan("lexical", rationale=tuple(rationale), estimated_model_calls=0),
            )

        def estimate(pool: Sequence[CandidateView[T]], mode: BackendMode) -> int:
            assert self.backend is not None
            return estimate_request_tokens(
                self.backend,
                ModelRequest(
                    query,
                    tuple(BackendCandidate(item.occurrence_id, item.text) for item in pool),
                    prompt,
                    mode,
                ),
            )

        max_listwise = min(
            self.config.batch_size,
            self.backend.capabilities.max_batch_size or self.config.batch_size,
        )
        estimated_tokens = estimate(candidates, "listwise") if count <= max_listwise else None
        listwise_fits = (
            self.backend.capabilities.listwise
            and estimated_tokens is not None
            and (
                self.backend.capabilities.max_context_tokens is None
                or estimated_tokens <= self.backend.capabilities.max_context_tokens
            )
            and (budget.max_tokens is None or estimated_tokens * attempts <= budget.max_tokens)
            and (available_calls is None or available_calls >= 1)
        )
        if count <= 10 and count <= max_listwise and listwise_fits:
            strategy = ListwiseStrategy(batch_size=10)
            rationale.append("small candidate set fits one listwise judgment group")
            return AutoDecision(
                strategy,
                ExecutionPlan(
                    "listwise",
                    rationale=tuple(rationale),
                    estimated_model_calls=1,
                    estimated_tokens=estimated_tokens,
                ),
            )
        if count <= 100:
            pointwise_estimates = [estimate((candidate,), "pointwise") for candidate in candidates]
            pointwise_tokens = sum(pointwise_estimates)
            pointwise_fits = (
                self.backend.capabilities.pointwise
                and (available_calls is None or available_calls >= count)
                and (
                    self.backend.capabilities.max_context_tokens is None
                    or max(pointwise_estimates, default=0)
                    <= self.backend.capabilities.max_context_tokens
                )
                and (budget.max_tokens is None or pointwise_tokens * attempts <= budget.max_tokens)
            )
            one_group_fits = listwise_fits and count <= max_listwise
            if context.quality_mode == "quality" and one_group_fits:
                strategy = ListwiseStrategy(batch_size=count)
                calls = 1
                rationale.append("quality mode uses one bounded listwise group")
                strategy_name = "listwise"
            elif pointwise_fits:
                strategy = PointwiseStrategy()
                calls = count
                rationale.append("medium candidate set uses bounded concurrent pointwise scoring")
                strategy_name = "pointwise"
            elif one_group_fits:
                strategy = ListwiseStrategy(batch_size=count)
                calls = 1
                rationale.append("model-call budget favors one bounded listwise group")
                strategy_name = "listwise"
            else:
                strategy = LexicalStrategy()
                calls = 0
                rationale.append("configured budgets make model reranking infeasible")
                strategy_name = "lexical"
            return AutoDecision(
                strategy,
                ExecutionPlan(
                    strategy_name,
                    rationale=tuple(rationale),
                    estimated_model_calls=calls,
                    estimated_tokens=(
                        estimated_tokens if strategy_name == "listwise" else pointwise_tokens
                    )
                    if calls
                    else 0,
                ),
            )

        if top_k is None:
            rationale.append("full ordering cannot use a pruning cascade")
            return AutoDecision(
                LexicalStrategy(),
                ExecutionPlan("lexical", rationale=tuple(rationale), estimated_model_calls=0),
            )

        retention = 40 if context.quality_mode == "quality" else 20
        lexical_limit = min(count, max(100, requested * retention))
        stages: list[PipelineStage[Any]] = [BM25Filter(limit=lexical_limit)]
        planned = [
            ExecutionStage(
                "BM25Filter",
                "lexical",
                count,
                lexical_limit,
                approximate=True,
                reason="large pools require a cheap bounded prefilter",
            )
        ]
        remaining = lexical_limit
        if self.embedding_backend is not None:
            embedding_factor = 10 if context.quality_mode == "quality" else 6
            embedding_limit = min(remaining, max(30, requested * embedding_factor))
            stages.append(
                EmbeddingReranker(EmbeddingSimilarity(self.embedding_backend), embedding_limit)
            )
            planned.append(
                ExecutionStage(
                    "EmbeddingReranker",
                    "embedding",
                    remaining,
                    embedding_limit,
                    self.embedding_backend.backend_id,
                    approximate=True,
                    reason="reduce the semantic shortlist before model judgment",
                )
            )
            remaining = embedding_limit
        mode = (
            "listwise"
            if self.backend.capabilities.listwise and remaining <= max_listwise
            else "pointwise"
        )
        if mode == "pointwise" and not self.backend.capabilities.pointwise:
            rationale.append("available model primitives cannot score the shortlist")
            return AutoDecision(
                LexicalStrategy(),
                ExecutionPlan("lexical", rationale=tuple(rationale), estimated_model_calls=0),
            )
        expected_calls = 1 if mode == "listwise" else remaining
        worst_pool = sorted(candidates, key=lambda item: len(item.text), reverse=True)[:remaining]
        request_estimates = (
            [estimate(worst_pool, "listwise")]
            if mode == "listwise"
            else [estimate((item,), "pointwise") for item in worst_pool]
        )
        estimated_shortlist_tokens = sum(request_estimates)
        context_limit = self.backend.capabilities.max_context_tokens
        if (
            (available_calls is not None and expected_calls > available_calls)
            or (context_limit is not None and max(request_estimates) > context_limit)
            or (
                budget.max_tokens is not None
                and estimated_shortlist_tokens * attempts > budget.max_tokens
            )
        ):
            rationale.append("model limits or budgets only permit deterministic ranking")
            return AutoDecision(
                LexicalStrategy(),
                ExecutionPlan("lexical", rationale=tuple(rationale), estimated_model_calls=0),
            )
        stages.append(JevReranker(limit=requested, mode=mode, batch_size=max_listwise))
        planned.append(
            ExecutionStage(
                "ModelReranker",
                mode,
                remaining,
                requested,
                self.backend.backend_id,
                approximate=False,
                reason="apply the strongest configured backend to the shortlist",
            )
        )
        rationale.append("large candidate set uses a transparent coarse-to-strong cascade")
        pipeline: RerankPipeline[Any] = RerankPipeline(stages)
        return AutoDecision(
            pipeline,
            ExecutionPlan(
                "auto",
                stages=tuple(planned),
                approximate=True,
                rationale=tuple(rationale),
                estimated_model_calls=expected_calls,
                estimated_tokens=estimated_shortlist_tokens,
            ),
        )

    async def rank(
        self,
        *,
        query: str,
        candidates: Sequence[CandidateView[T]],
        context: RerankContext,
        services: EvaluationServices,
        top_k: int | None,
    ) -> RankingOutcome:
        decision = self.decide(candidates, query=query, top_k=top_k, context=context)
        strategy = cast(RankingStrategy[T], decision.strategy)
        return await strategy.rank(
            query=query,
            candidates=candidates,
            context=context,
            services=services,
            top_k=top_k,
        )
