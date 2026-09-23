"""Primary async and synchronous reranking facade."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar, cast

from ._runtime.executor import ExecutionServices, effective_budget
from .auto import AutoStrategy
from .backends import EmbeddingBackend, JevBackend, ModelBackend
from .cache import CacheBackend, MemoryCache
from .candidates import CandidateAdapter, MetadataValue, prepare_candidates
from .config import FallbackPolicy, RerankerConfig
from .context import RerankContext
from .convenience import (
    rerank_documents as _rerank_documents,
)
from .convenience import (
    rerank_entities as _rerank_entities,
)
from .convenience import (
    rerank_memories as _rerank_memories,
)
from .convenience import (
    rerank_tools as _rerank_tools,
)
from .errors import (
    CapabilityError,
    ConfigurationError,
    ConstraintError,
    DeadlineExceededError,
    RerankError,
)
from .metrics import (
    EmbeddingSimilarity,
    LexicalRelevance,
    LLMRelevance,
    MetadataNumericMetric,
    Metric,
    RecencyMetric,
    WeightedMetrics,
)
from .observability import ObserverDispatcher, TraceEvent, TraceKind
from .pipeline import PipelineStageError
from .prompts import DEFAULT_PROMPT, RerankPrompt
from .strategies import (
    LexicalStrategy,
    ListwiseStrategy,
    MetricStrategy,
    PointwiseStrategy,
    RankingStrategy,
)
from .types import (
    Coverage,
    ExecutionPlan,
    ExecutionStage,
    RerankResponse,
    RerankResult,
    RerankStatistics,
    ResultStatus,
    ScoreKind,
)

T = TypeVar("T")


class Reranker:
    def __init__(
        self,
        model: str | ModelBackend | None = None,
        *,
        strategy: str | RankingStrategy[Any] | None = None,
        embedding_backend: EmbeddingBackend | None = None,
        config: RerankerConfig | None = None,
        cache: CacheBackend | None = None,
        prompt: RerankPrompt | None = None,
    ) -> None:
        self.config = config or RerankerConfig()
        self.embedding_backend = embedding_backend
        self.prompt = prompt or DEFAULT_PROMPT
        self._owns_backend = isinstance(model, str)
        self._model_spec = model if isinstance(model, str) else None
        self.backend = self._resolve_backend(model)
        self.strategy = strategy
        self.cache = cache or (
            MemoryCache(
                max_entries=self.config.cache.max_entries,
                default_ttl_seconds=self.config.cache.ttl_seconds,
            )
            if self.config.cache.enabled
            else None
        )
        self._scope_active = False
        self._closed = False
        self._bound_loop: asyncio.AbstractEventLoop | None = None
        self._observer_dispatcher: ObserverDispatcher | None = None
        self._shared_semaphore: asyncio.Semaphore | None = None
        self._active_tasks: set[asyncio.Task[Any]] = set()

    def _resolve_backend(self, model: str | ModelBackend | None) -> ModelBackend | None:
        if model is None:
            return None
        if not isinstance(model, str):
            return model
        if model.startswith("typesafe:"):
            model_id = model.partition(":")[2]
            if not model_id:
                raise ConfigurationError("typesafe model identifier cannot be empty")
            return JevBackend(
                model=model_id,
                timeout_s=self.config.timeout_s,
                retry=self.config.retry,
            )
        raise ConfigurationError(f"unknown model namespace: {model!r}")

    async def __aenter__(self) -> Reranker:
        if self._closed or self._scope_active:
            raise RuntimeError("reranker scope cannot be entered")
        self._scope_active = True
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        current = asyncio.current_task()
        pending = [task for task in self._active_tasks if task is not current and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self._observer_dispatcher is not None:
            await self._observer_dispatcher.aclose()
        if self._owns_backend and self.backend is not None:
            await self.backend.aclose()
        self._scope_active = False
        self._bound_loop = None
        self._shared_semaphore = None

    def _select_strategy(
        self,
        *,
        metrics: WeightedMetrics[object] | None,
    ) -> RankingStrategy[Any]:
        if metrics is not None:
            if self.strategy not in (None, "pointwise"):
                raise ConfigurationError(
                    "weighted metrics currently require pointwise metric aggregation"
                )
            return MetricStrategy(metrics)
        selected = self.strategy
        if selected is None:
            return PointwiseStrategy() if self.backend is not None else LexicalStrategy()
        if not isinstance(selected, str):
            return selected
        if selected == "pointwise":
            if self.backend is None:
                return LexicalStrategy()
            return PointwiseStrategy()
        if selected == "listwise":
            return ListwiseStrategy(batch_size=self.config.batch_size)
        if selected == "auto":
            return AutoStrategy(self.backend, self.embedding_backend, self.config)
        if selected == "lexical":
            return LexicalStrategy()
        raise ConfigurationError(f"unknown ranking strategy: {selected!r}")

    def _resolve_metrics(
        self,
        metrics: Sequence[str | Metric[T]] | None,
        weights: Mapping[str, float] | None,
        prompt: RerankPrompt,
    ) -> WeightedMetrics[object] | None:
        if metrics is None:
            if weights is not None:
                raise ConfigurationError("weights require explicit metrics")
            return None
        if not metrics:
            raise ConfigurationError("metrics cannot be empty")
        resolved: list[Metric[Any]] = []
        aliases_seen: set[str] = set()
        for metric in metrics:
            if not isinstance(metric, str):
                resolved.append(metric)
                continue
            canonical = (
                "semantic"
                if metric in {"semantic", "semantic_similarity", "embedding_similarity"}
                else metric
            )
            if canonical in aliases_seen:
                raise ConfigurationError(f"duplicate metric: {canonical}")
            aliases_seen.add(canonical)
            if canonical == "lexical":
                resolved.append(LexicalRelevance())
            elif canonical == "semantic":
                if self.embedding_backend is None:
                    raise ConfigurationError("semantic metric requires an embedding backend")
                resolved.append(EmbeddingSimilarity(self.embedding_backend))
            elif canonical == "llm_relevance":
                if self.backend is None:
                    raise ConfigurationError("llm_relevance requires a model backend")
                resolved.append(
                    LLMRelevance(
                        self.backend,
                        prompt=prompt,
                        include_reasoning=self.config.include_reasoning,
                        reasoning_level=self.config.reasoning_level,
                    )
                )
            elif canonical == "recency":
                resolved.append(RecencyMetric())
            elif canonical in {"authority", "popularity", "business_priority"}:
                resolved.append(MetadataNumericMetric(canonical, name=canonical))
            else:
                raise ConfigurationError(f"unknown metric: {metric!r}")
        effective_weights = (
            {metric.name: 1.0 for metric in resolved} if weights is None else dict(weights)
        )
        normalized_weights: dict[str, float] = {}
        for name, value in effective_weights.items():
            canonical = (
                "semantic" if name in {"semantic_similarity", "embedding_similarity"} else name
            )
            if canonical in normalized_weights:
                raise ConfigurationError(f"duplicate metric weight: {canonical}")
            normalized_weights[canonical] = value
        return cast(WeightedMetrics[object], WeightedMetrics(tuple(resolved), normalized_weights))

    async def plan(
        self,
        *,
        query: str,
        candidates: Sequence[T],
        top_k: int | None = None,
        text_fn: Callable[[T], str] | None = None,
        metadata_fn: Callable[[T], Mapping[str, MetadataValue]] | None = None,
        id_fn: Callable[[T], str] | None = None,
        eligible_fn: Callable[[T], bool] | None = None,
        adapter: CandidateAdapter[T] | None = None,
        metrics: Sequence[str | Metric[T]] | None = None,
        weights: Mapping[str, float] | None = None,
        prompt: str | RerankPrompt | None = None,
        strategy: str | RankingStrategy[T] | None = None,
        context: RerankContext | None = None,
    ) -> ExecutionPlan:
        self._validate_request(query, top_k)
        if top_k == 0:
            return ExecutionPlan("noop")
        views = prepare_candidates(
            candidates,
            config=self.config,
            text_fn=text_fn,
            metadata_fn=metadata_fn,
            id_fn=id_fn,
            eligible_fn=eligible_fn,
            adapter=adapter,
        )
        if not views:
            return ExecutionPlan("noop")
        request_context = context or RerankContext()
        actual_prompt = self._resolve_prompt(prompt)
        weighted = self._resolve_metrics(metrics, weights, actual_prompt)
        if weighted is not None and strategy is not None and not isinstance(strategy, str):
            raise ConfigurationError("metrics cannot be combined with a strategy object")
        selected = self._select_strategy(metrics=weighted) if strategy is None else strategy
        if isinstance(selected, str):
            selected = cast(RankingStrategy[T], self._named_strategy(selected, weighted))
        if isinstance(selected, AutoStrategy):
            return selected.decide(
                views, query=query, top_k=top_k, context=request_context, prompt=actual_prompt
            ).plan
        return ExecutionPlan(
            selected.name,
            stages=(
                ExecutionStage(
                    selected.name,
                    selected.name,
                    len(views),
                    min(len(views) if top_k is None else top_k, len(views)),
                ),
            ),
        )

    async def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[T],
        top_k: int | None = None,
        text_fn: Callable[[T], str] | None = None,
        metadata_fn: Callable[[T], Mapping[str, MetadataValue]] | None = None,
        id_fn: Callable[[T], str] | None = None,
        eligible_fn: Callable[[T], bool] | None = None,
        adapter: CandidateAdapter[T] | None = None,
        metrics: Sequence[str | Metric[T]] | None = None,
        weights: Mapping[str, float] | None = None,
        prompt: str | RerankPrompt | None = None,
        strategy: str | RankingStrategy[T] | None = None,
        context: RerankContext | None = None,
    ) -> RerankResponse[T]:
        task = asyncio.current_task()
        if task is not None:
            self._active_tasks.add(task)
        request_context = context or RerankContext()
        latency_limit = effective_budget(self.config.budget, request_context.budget).max_latency_ms
        loop = asyncio.get_running_loop()
        deadline = None if latency_limit is None else loop.time() + latency_limit / 1000
        try:
            try:
                async with asyncio.timeout_at(deadline) as scope:
                    response = await self._rerank_impl(
                        query=query,
                        candidates=candidates,
                        top_k=top_k,
                        text_fn=text_fn,
                        metadata_fn=metadata_fn,
                        id_fn=id_fn,
                        eligible_fn=eligible_fn,
                        adapter=adapter,
                        metrics=metrics,
                        weights=weights,
                        prompt=prompt,
                        strategy=strategy,
                        context=request_context,
                    )
            except TimeoutError as exc:
                if scope.expired():
                    raise DeadlineExceededError("reranking latency budget was exhausted") from exc
                raise
            if deadline is not None and loop.time() >= deadline and top_k != 0:
                raise DeadlineExceededError("reranking latency budget was exhausted")
            return response
        finally:
            if task is not None:
                self._active_tasks.discard(task)

    async def _rerank_impl(
        self,
        *,
        query: str,
        candidates: Sequence[T],
        top_k: int | None = None,
        text_fn: Callable[[T], str] | None = None,
        metadata_fn: Callable[[T], Mapping[str, MetadataValue]] | None = None,
        id_fn: Callable[[T], str] | None = None,
        eligible_fn: Callable[[T], bool] | None = None,
        adapter: CandidateAdapter[T] | None = None,
        metrics: Sequence[str | Metric[T]] | None = None,
        weights: Mapping[str, float] | None = None,
        prompt: str | RerankPrompt | None = None,
        strategy: str | RankingStrategy[T] | None = None,
        context: RerankContext | None = None,
    ) -> RerankResponse[T]:
        if self._closed:
            raise RuntimeError("reranker is closed")
        loop = asyncio.get_running_loop()
        if self._bound_loop is None:
            self._bound_loop = loop
        elif self._bound_loop is not loop:
            raise RuntimeError("reranker is bound to another event loop")
        if self._shared_semaphore is None:
            self._shared_semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self._validate_request(query, top_k)
        started = time.perf_counter()
        request_context = context or RerankContext()
        actual_prompt = self._resolve_prompt(prompt)
        if top_k == 0:
            return RerankResponse(
                (),
                stats=RerankStatistics(
                    total_latency_ms=(time.perf_counter() - started) * 1000,
                    candidate_count=len(candidates),
                ),
                execution_plan=ExecutionPlan("noop"),
                status=ResultStatus.NOOP,
                coverage=Coverage(input_count=len(candidates)),
                request_id=request_context.request_id,
            )
        views = prepare_candidates(
            candidates,
            config=self.config,
            text_fn=text_fn,
            metadata_fn=metadata_fn,
            id_fn=id_fn,
            eligible_fn=eligible_fn,
            adapter=adapter,
        )
        if not views:
            return RerankResponse(
                (),
                stats=RerankStatistics(
                    total_latency_ms=(time.perf_counter() - started) * 1000,
                    candidate_count=len(candidates),
                    eligible_count=len(views),
                ),
                execution_plan=ExecutionPlan("noop"),
                status=ResultStatus.NOOP,
                coverage=Coverage(len(candidates), len(views), 0, 0),
                request_id=request_context.request_id,
            )
        weighted = self._resolve_metrics(metrics, weights, actual_prompt)
        if self.config.observer is not None and self._observer_dispatcher is None:
            self._observer_dispatcher = ObserverDispatcher(
                self.config.observer, capacity=self.config.observer_queue_size
            )
        observer = self._observer_dispatcher
        observer_dropped_before = observer.dropped if observer is not None else 0
        observer_errors_before = observer.errors if observer is not None else 0
        selected_strategy = cast(
            RankingStrategy[T],
            self._select_strategy(metrics=weighted) if strategy is None else strategy,
        )
        if weighted is not None and strategy is not None and not isinstance(strategy, str):
            raise ConfigurationError("metrics cannot be combined with a strategy object")
        if isinstance(selected_strategy, str):
            selected_strategy = cast(
                RankingStrategy[T], self._named_strategy(selected_strategy, weighted)
            )
        automatic_plan: ExecutionPlan | None = None
        execution_strategy = selected_strategy
        if isinstance(selected_strategy, AutoStrategy):
            decision = selected_strategy.decide(
                views, query=query, top_k=top_k, context=request_context, prompt=actual_prompt
            )
            automatic_plan = decision.plan
            execution_strategy = cast(RankingStrategy[T], decision.strategy)
        services = ExecutionServices(
            backend=self.backend,
            cache=self.cache,
            config=self.config,
            context=request_context,
            prompt=actual_prompt,
            observer=observer,
            semaphore=self._shared_semaphore,
        )
        status = ResultStatus.OK
        warnings: list[str] = []
        try:
            outcome = await execution_strategy.rank(
                query=query,
                candidates=views,
                context=request_context,
                services=services,
                top_k=top_k,
            )
        except asyncio.CancelledError:
            raise
        except RerankError as primary_error:
            outcome, status = await self._fallback(
                query=query,
                candidates=views,
                context=request_context,
                services=services,
                top_k=top_k,
                selected=execution_strategy,
                primary_error=primary_error,
            )
            services.statistics.fallbacks += 1
            warnings.append("primary strategy failed; configured fallback was used")

        by_id = {view.occurrence_id: view for view in views}
        if len(outcome.entries) != len({entry.occurrence_id for entry in outcome.entries}):
            raise ConfigurationError("strategy returned duplicate occurrence IDs")
        unknown = {entry.occurrence_id for entry in outcome.entries} - set(by_id)
        if unknown:
            raise ConfigurationError("strategy returned unknown occurrence IDs")
        for entry in outcome.entries:
            if entry.score is None:
                if outcome.score_kind is not ScoreKind.RANK_ONLY:
                    raise ConfigurationError("scored strategies must return a score for every item")
                continue
            if isinstance(entry.score, bool) or not math.isfinite(entry.score):
                raise ConfigurationError("strategy returned a non-finite score")
            if outcome.score_kind is ScoreKind.UTILITY and not 0 <= entry.score <= 1:
                raise ConfigurationError("utility scores must be in [0, 1]")
        if self.config.score_threshold is not None and outcome.score_kind is not ScoreKind.UTILITY:
            raise ConfigurationError("score_threshold requires utility scores")
        if eligible_fn is not None and any(not eligible_fn(view.item) for view in views):
            raise ConstraintError("candidate eligibility changed during ranking")
        selected_entries = [
            entry
            for entry in outcome.entries
            if self.config.score_threshold is None
            or (entry.score is not None and entry.score >= self.config.score_threshold)
        ]
        if top_k is not None:
            selected_entries = selected_entries[:top_k]
        if len(outcome.entries) < len(views) and any(
            "omitted" in warning for warning in outcome.warnings
        ):
            status = ResultStatus.PARTIAL
        results = tuple(
            RerankResult(
                item=by_id[entry.occurrence_id].item,
                score=entry.score,
                rank=rank,
                metrics=entry.metrics,
                reasoning=entry.reasoning,
                metadata=by_id[entry.occurrence_id].metadata,
                input_index=by_id[entry.occurrence_id].input_index,
                occurrence_id=entry.occurrence_id,
                candidate_id=by_id[entry.occurrence_id].candidate_id,
                score_kind=outcome.score_kind,
                selection_score=entry.selection_score,
            )
            for rank, entry in enumerate(selected_entries, start=1)
        )
        mutable = services.statistics
        if observer is not None:
            for stage in outcome.stages:
                observer.emit(
                    TraceEvent(
                        TraceKind.STAGE_END,
                        request_context.request_id,
                        stage=stage.name,
                        candidate_count=stage.output_count,
                        attributes={"input_count": stage.input_count},
                    )
                )
            observer.emit(
                TraceEvent(
                    TraceKind.RERANK_END,
                    request_context.request_id,
                    candidate_count=len(results),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    attributes={"status": status.value},
                )
            )
        stats = RerankStatistics(
            total_latency_ms=(time.perf_counter() - started) * 1000,
            model_latency_ms=mutable.model_latency_ms,
            candidate_count=len(candidates),
            eligible_count=len(views),
            model_calls=mutable.model_calls,
            input_tokens=mutable.input_tokens,
            output_tokens=mutable.output_tokens,
            usage_complete=mutable.usage_complete,
            estimated_cost_usd=mutable.estimated_cost_usd,
            cost_confidence=mutable.cost_confidence,
            cache_hits=mutable.cache_hits,
            cache_misses=mutable.cache_misses,
            retries=mutable.retries,
            fallbacks=mutable.fallbacks,
            batches=mutable.batches,
            max_concurrency=mutable.max_active,
            observer_events_dropped=(
                observer.dropped - observer_dropped_before if observer is not None else 0
            ),
            observer_errors=(
                observer.errors - observer_errors_before if observer is not None else 0
            ),
        )
        stages = outcome.stages or (
            ExecutionStage(
                selected_strategy.name, selected_strategy.name, len(views), len(results)
            ),
        )
        partial_count = (
            max(0, stages[-1].input_count - len(outcome.entries))
            if any("omitted" in warning for warning in outcome.warnings)
            else 0
        )
        pruned_count = max(0, len(views) - stages[-1].input_count)
        plan = (
            ExecutionPlan(
                automatic_plan.strategy,
                stages=stages,
                approximate=automatic_plan.approximate or outcome.approximate,
                rationale=automatic_plan.rationale,
                estimated_model_calls=automatic_plan.estimated_model_calls,
                estimated_tokens=automatic_plan.estimated_tokens,
                estimated_cost_usd=automatic_plan.estimated_cost_usd,
            )
            if automatic_plan is not None
            else ExecutionPlan(
                selected_strategy.name,
                stages=stages,
                approximate=outcome.approximate,
                rationale=("explicit strategy",),
            )
        )
        if outcome.approximate and status is ResultStatus.OK:
            warnings.append("the selected strategy is approximate")
        warnings.extend(outcome.warnings)
        return RerankResponse(
            results,
            stats=stats,
            execution_plan=plan,
            status=status,
            coverage=Coverage(
                len(candidates),
                len(views),
                stages[-1].input_count - partial_count,
                len(results),
                partial_count,
                pruned_count,
            ),
            warnings=tuple(warnings),
            request_id=request_context.request_id,
        )

    def _named_strategy(
        self, name: str, metrics: WeightedMetrics[object] | None
    ) -> RankingStrategy[Any]:
        if metrics is not None:
            if name != "pointwise":
                raise ConfigurationError(
                    "weighted metrics currently require pointwise metric aggregation"
                )
            return MetricStrategy(metrics)
        if name == "pointwise":
            return PointwiseStrategy() if self.backend is not None else LexicalStrategy()
        if name == "listwise":
            return ListwiseStrategy(batch_size=self.config.batch_size)
        if name == "auto":
            return AutoStrategy(self.backend, self.embedding_backend, self.config)
        if name == "lexical":
            return LexicalStrategy()
        raise ConfigurationError(f"unknown ranking strategy: {name!r}")

    async def rerank_documents(
        self, *, query: str, candidates: Sequence[T], **kwargs: Any
    ) -> RerankResponse[T]:
        return await _rerank_documents(self, query=query, candidates=candidates, **kwargs)

    async def rerank_entities(
        self, *, query: str, candidates: Sequence[T], **kwargs: Any
    ) -> RerankResponse[T]:
        return await _rerank_entities(self, query=query, candidates=candidates, **kwargs)

    async def rerank_tools(
        self, *, query: str, candidates: Sequence[T], **kwargs: Any
    ) -> RerankResponse[T]:
        return await _rerank_tools(self, query=query, candidates=candidates, **kwargs)

    async def rerank_memories(
        self, *, query: str, candidates: Sequence[T], **kwargs: Any
    ) -> RerankResponse[T]:
        return await _rerank_memories(self, query=query, candidates=candidates, **kwargs)

    async def _fallback(
        self,
        *,
        query: str,
        candidates: Sequence[Any],
        context: RerankContext,
        services: ExecutionServices,
        top_k: int | None,
        selected: RankingStrategy[Any],
        primary_error: RerankError,
    ) -> tuple[Any, ResultStatus]:
        last_error: Exception | None = None
        for policy in self.config.fallback_chain:
            if policy is FallbackPolicy.STRICT:
                raise primary_error
            try:
                if policy in {FallbackPolicy.PREVIOUS_STAGE, FallbackPolicy.PARTIAL}:
                    if isinstance(primary_error, PipelineStageError):
                        return (
                            primary_error.checkpoint,
                            ResultStatus.PARTIAL
                            if policy is FallbackPolicy.PARTIAL
                            else ResultStatus.FALLBACK,
                        )
                    continue
                if policy is FallbackPolicy.SMALLER_BATCHES and isinstance(
                    selected, ListwiseStrategy
                ):
                    size = max(1, (selected.batch_size or self.config.batch_size) // 2)
                    return await ListwiseStrategy(size).rank(
                        query=query,
                        candidates=candidates,
                        context=context,
                        services=services,
                        top_k=top_k,
                    ), ResultStatus.FALLBACK
                if policy is FallbackPolicy.POINTWISE:
                    return await PointwiseStrategy().rank(
                        query=query,
                        candidates=candidates,
                        context=context,
                        services=services,
                        top_k=top_k,
                    ), ResultStatus.FALLBACK
                if policy is FallbackPolicy.LEXICAL:
                    return await LexicalStrategy().rank(
                        query=query,
                        candidates=candidates,
                        context=context,
                        services=services,
                        top_k=top_k,
                    ), ResultStatus.FALLBACK
                if policy is FallbackPolicy.INPUT_ORDER:
                    from .types import RankingEntry, RankingOutcome, ScoreKind

                    entries = tuple(
                        RankingEntry(candidate.occurrence_id, None)
                        for candidate in (candidates if top_k is None else candidates[:top_k])
                    )
                    return RankingOutcome(entries, ScoreKind.RANK_ONLY), ResultStatus.FALLBACK
            except RerankError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise primary_error

    def _resolve_prompt(self, prompt: str | RerankPrompt | None) -> RerankPrompt:
        if prompt is None:
            return self.prompt
        if isinstance(prompt, RerankPrompt):
            return prompt
        return RerankPrompt(criteria={"relevance": prompt})

    @staticmethod
    def _validate_request(query: str, top_k: int | None) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ConfigurationError("query must be a non-empty string")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0
        ):
            raise ConfigurationError("top_k must be a non-negative integer or None")

    def rerank_sync(self, **kwargs: Any) -> RerankResponse[Any]:
        if self._scope_active:
            raise RuntimeError("rerank_sync cannot run while an async scope is active")
        if self.backend is not None and not self._owns_backend:
            raise CapabilityError(
                "rerank_sync requires an owned model configuration; use await rerank"
            )
        try:
            asyncio.get_running_loop()
        except RuntimeError:

            async def run_once() -> RerankResponse[Any]:
                try:
                    return await self.rerank(**kwargs)
                finally:
                    if self._observer_dispatcher is not None:
                        await self._observer_dispatcher.aclose()
                        self._observer_dispatcher = None
                    if self._owns_backend and self.backend is not None:
                        await self.backend.aclose()
                        self.backend = self._resolve_backend(self._model_spec)
                    self._bound_loop = None
                    self._shared_semaphore = None

            return asyncio.run(run_once())
        raise RuntimeError(
            "rerank_sync cannot run inside an active event loop; await rerank instead"
        )


class AutoReranker(Reranker):
    def __init__(self, model: str | ModelBackend | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("strategy", "auto")
        super().__init__(model, **kwargs)
