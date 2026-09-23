from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, TypeVar, cast

from ..backends import BackendCandidate, ModelBackend, ModelRequest, ModelResponse
from ..cache import CacheBackend, CacheRecord
from ..candidates import CandidateView
from ..config import Budget, RerankerConfig
from ..context import RerankContext
from ..errors import (
    BudgetExceededError,
    BudgetUnverifiableError,
    CapabilityError,
    ConfigurationError,
    DeadlineExceededError,
    OutputValidationError,
    RerankError,
)
from ..metrics import BatchMetric, Metric, WeightedMetrics
from ..metrics.base import validate_utility
from ..metrics.builtin import LexicalRelevance, LLMRelevance, MetadataNumericMetric
from ..observability import ObserverDispatcher, TraceEvent, TraceKind
from ..prompts import RerankPrompt
from ..types import CostConfidence, MetricScore, RankingEntry, RankingOutcome, ScoreKind
from .cache_keys import cache_key

U = TypeVar("U")


def estimate_request_tokens(backend: ModelBackend, request: ModelRequest) -> int:
    """Use a backend's serialized-request estimate when one is available."""
    estimator = getattr(backend, "estimate_request_tokens", None)
    if callable(estimator):
        value = estimator(request)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CapabilityError("backend returned an invalid request token estimate")
        return value
    chars = len(request.query) + sum(len(candidate.text) for candidate in request.candidates)
    return max(1, math.ceil(chars / 4))


async def _gather_owned(*calls: Coroutine[Any, Any, U]) -> list[U]:
    """Finish or cancel every child before its caller can leave the request."""
    tasks: list[asyncio.Task[U]] = [asyncio.create_task(call) for call in calls]
    try:
        return list(await asyncio.gather(*tasks))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@dataclass(slots=True)
class MutableStatistics:
    model_latency_ms: float = 0.0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usage_complete: bool = True
    estimated_cost_usd: float | None = None
    cost_confidence: CostConfidence = CostConfidence.UNKNOWN
    cost_missing: bool = False
    cache_hits: int = 0
    cache_misses: int = 0
    retries: int = 0
    fallbacks: int = 0
    batches: int = 0
    active: int = 0
    max_active: int = 0


def _minimum_optional(
    left: int | float | Decimal | None,
    right: int | float | Decimal | None,
) -> Any:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def effective_budget(configured: Budget, requested: Budget) -> Budget:
    return Budget(
        max_cost_usd=_minimum_optional(configured.max_cost_usd, requested.max_cost_usd),
        max_tokens=_minimum_optional(configured.max_tokens, requested.max_tokens),
        max_latency_ms=_minimum_optional(configured.max_latency_ms, requested.max_latency_ms),
        max_model_calls=_minimum_optional(configured.max_model_calls, requested.max_model_calls),
        strict_cost=configured.strict_cost or requested.strict_cost,
    )


class BudgetLedger:
    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.started = time.monotonic()
        self.deadline = (
            None if budget.max_latency_ms is None else self.started + budget.max_latency_ms / 1000
        )
        self.calls = 0
        self.tokens_reserved = 0
        self.cost = Decimal(0)
        self._lock = asyncio.Lock()

    def remaining_seconds(self, default: float) -> float:
        if self.deadline is None:
            return default
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise DeadlineExceededError("reranking latency budget was exhausted")
        return min(default, remaining)

    async def reserve(
        self,
        *,
        estimated_tokens: int,
        estimated_cost: Decimal | None,
        calls: int = 1,
    ) -> None:
        async with self._lock:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                raise DeadlineExceededError("reranking latency budget was exhausted")
            next_calls = self.calls + calls
            next_tokens = self.tokens_reserved + estimated_tokens
            if self.budget.max_model_calls is not None and next_calls > self.budget.max_model_calls:
                raise BudgetExceededError("model-call budget was exhausted")
            if self.budget.max_tokens is not None and next_tokens > self.budget.max_tokens:
                raise BudgetExceededError("token budget was exhausted")
            if self.budget.max_cost_usd is not None:
                if estimated_cost is None and self.budget.strict_cost:
                    raise BudgetUnverifiableError("strict cost budget requires a priced backend")
                if (
                    estimated_cost is not None
                    and self.cost + estimated_cost > self.budget.max_cost_usd
                ):
                    raise BudgetExceededError("cost budget was exhausted")
            self.calls = next_calls
            self.tokens_reserved = next_tokens
            if estimated_cost is not None:
                self.cost += estimated_cost

    async def reconcile(
        self,
        *,
        estimated_tokens: int,
        estimated_cost: Decimal | None,
        actual_tokens: int,
        actual_cost: Decimal | None,
        reserved_calls: int = 1,
        actual_calls: int = 1,
    ) -> None:
        async with self._lock:
            self.calls += actual_calls - reserved_calls
            self.tokens_reserved += actual_tokens - estimated_tokens
            if estimated_cost is not None:
                self.cost -= estimated_cost
            if actual_cost is not None:
                self.cost += actual_cost
            if self.budget.max_tokens is not None and self.tokens_reserved > self.budget.max_tokens:
                raise BudgetExceededError("actual token usage exceeded the token budget")
            if self.budget.max_cost_usd is not None:
                if actual_cost is None and self.budget.strict_cost:
                    raise BudgetUnverifiableError("provider did not report priced usage")
                if self.cost > self.budget.max_cost_usd:
                    raise BudgetExceededError("actual usage exceeded the cost budget")

    async def settle_failure(
        self,
        *,
        reserved_calls: int,
        actual_calls: int,
        reserved_tokens: int,
        estimated_tokens_per_attempt: int,
        reserved_cost: Decimal | None,
    ) -> None:
        """Release unused retries; retain a conservative estimate for attempts made."""
        async with self._lock:
            self.calls += actual_calls - reserved_calls
            self.tokens_reserved += estimated_tokens_per_attempt * actual_calls - reserved_tokens
            if reserved_cost is not None:
                self.cost -= reserved_cost
                self.cost += reserved_cost * Decimal(actual_calls) / Decimal(reserved_calls)


class ExecutionServices:
    def __init__(
        self,
        *,
        backend: ModelBackend | None,
        cache: CacheBackend | None,
        config: RerankerConfig,
        context: RerankContext,
        prompt: RerankPrompt,
        observer: ObserverDispatcher | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.backend = backend
        self.cache = cache
        self.config = config
        self.context = context
        self.prompt = prompt
        self.statistics = MutableStatistics()
        self.observer = observer
        self.ledger = BudgetLedger(effective_budget(config.budget, context.budget))
        self._semaphore = semaphore or asyncio.Semaphore(config.max_concurrency)
        self._inflight: dict[str, tuple[asyncio.Task[ModelResponse], int]] = {}
        self._inflight_lock = asyncio.Lock()

    def _estimated_cost(self, tokens: int) -> Decimal | None:
        assert self.backend is not None
        price = getattr(self.backend, "input_price_per_million_usd", None)
        if price is None:
            return None
        if hasattr(self.backend, "output_price_per_million_usd"):
            output_price = self.backend.output_price_per_million_usd
            if output_price is None:
                return None
            price = max(price, output_price)
        return cast(Decimal, price) * Decimal(tokens) / Decimal(1_000_000)

    def _model_key(
        self,
        query: str,
        candidates: Sequence[CandidateView[object]],
        mode: str,
        prompt: RerankPrompt,
        include_reasoning: bool,
        reasoning_level: str | None,
    ) -> str:
        assert self.backend is not None
        return cache_key(
            self.config.cache.namespace,
            {
                "schema": "model-v2",
                "tenant": self.context.tenant,
                "authorization_revision": self.context.authorization_revision,
                "query": query,
                "candidates": [candidate.text for candidate in candidates],
                "ordered_occurrences": [candidate.occurrence_id for candidate in candidates]
                if mode == "listwise"
                else None,
                "strategy": mode,
                "prompt": prompt.fingerprint,
                "model": self.backend.cache_identity,
                "include_reasoning": include_reasoning,
                "reasoning_level": reasoning_level,
            },
        )

    async def model_scores(
        self,
        *,
        query: str,
        candidates: Sequence[CandidateView[object]],
        mode: str,
        allow_partial: bool = False,
        prompt: RerankPrompt | None = None,
        include_reasoning: bool | None = None,
        reasoning_level: str | None = None,
    ) -> RankingOutcome:
        if self.backend is None:
            raise CapabilityError("the selected strategy requires a model backend")
        capabilities = self.backend.capabilities
        if mode == "pointwise" and not capabilities.pointwise:
            raise CapabilityError("model backend does not support pointwise scoring")
        if mode == "listwise" and not capabilities.listwise:
            raise CapabilityError("model backend does not support listwise scoring")
        if (
            mode == "listwise"
            and capabilities.max_batch_size is not None
            and len(candidates) > capabilities.max_batch_size
        ):
            raise CapabilityError("listwise request exceeds the backend batch limit")
        active_prompt = prompt or self.prompt
        reasoning_level = reasoning_level or self.config.reasoning_level
        want_reasoning = (
            self.config.include_reasoning or active_prompt.include_reasoning
            if include_reasoning is None
            else include_reasoning
        )
        if want_reasoning and not capabilities.explanations:
            raise CapabilityError("model backend does not support explanations")
        if reasoning_level is not None and reasoning_level not in capabilities.reasoning_levels:
            raise CapabilityError("model backend does not support the requested reasoning level")

        if mode == "pointwise":
            responses: list[ModelResponse] = []
            window = max(1, self.config.max_concurrency * 2)
            for start in range(0, len(candidates), window):
                chunk = candidates[start : start + window]
                responses.extend(
                    await _gather_owned(
                        *(
                            self._call_model(
                                query,
                                (candidate,),
                                mode,
                                allow_partial,
                                active_prompt,
                                want_reasoning,
                                reasoning_level,
                            )
                            for candidate in chunk
                        )
                    )
                )
        else:
            responses = [
                await self._call_model(
                    query,
                    tuple(candidates),
                    mode,
                    allow_partial,
                    active_prompt,
                    want_reasoning,
                    reasoning_level,
                )
            ]
        entries: list[RankingEntry] = []
        missing: list[str] = []
        for response in responses:
            for score in response.scores:
                entries.append(
                    RankingEntry(
                        score.candidate_id,
                        score.score,
                        metrics={
                            "llm_relevance": MetricScore(
                                score.score,
                                utility=score.score,
                                raw_kind="noul_probability",
                                reasoning=score.reasoning,
                            )
                        },
                        reasoning=score.reasoning,
                    )
                )
            missing.extend(response.missing_candidate_ids)
        warnings = (f"model response omitted {len(missing)} candidates",) if missing else ()
        return RankingOutcome(tuple(entries), ScoreKind.UTILITY, bool(missing), warnings)

    async def _call_model(
        self,
        query: str,
        candidates: tuple[CandidateView[object], ...],
        mode: str,
        allow_partial: bool,
        prompt: RerankPrompt,
        include_reasoning: bool,
        reasoning_level: str | None,
    ) -> ModelResponse:
        assert self.backend is not None
        key = self._model_key(query, candidates, mode, prompt, include_reasoning, reasoning_level)
        cache_allowed = self.cache is not None and bool(
            getattr(self.backend, "cache_stable", False)
        )
        if cache_allowed and self.cache is not None:
            record = await self.cache.get(key)
            if record is not None:
                if isinstance(record.value, ModelResponse):
                    try:
                        restored = self._restore_cached(record.value, candidates, mode)
                    except OutputValidationError:
                        await self.cache.delete(key)
                    else:
                        self.statistics.cache_hits += 1
                        return restored
                else:
                    await self.cache.delete(key)
            self.statistics.cache_misses += 1

        inflight_key = f"{key}:{allow_partial}:{','.join(c.occurrence_id for c in candidates)}"
        async with self._inflight_lock:
            active = self._inflight.get(inflight_key)
            if active is None:
                task = asyncio.create_task(
                    self._execute_model(
                        query,
                        candidates,
                        mode,
                        allow_partial,
                        prompt,
                        include_reasoning,
                        reasoning_level,
                    ),
                    name="jev-rankkit-model",
                )
                self._inflight[inflight_key] = (task, 1)
            else:
                task, waiters = active
                self._inflight[inflight_key] = (task, waiters + 1)
        try:
            response = await asyncio.shield(task)
        finally:
            cancelled_task: asyncio.Task[ModelResponse] | None = None
            async with self._inflight_lock:
                active = self._inflight.get(inflight_key)
                if active is not None and active[0] is task:
                    remaining = active[1] - 1
                    if remaining == 0:
                        self._inflight.pop(inflight_key, None)
                        if not task.done():
                            task.cancel()
                            cancelled_task = task
                    else:
                        self._inflight[inflight_key] = (task, remaining)
            if cancelled_task is not None:
                await asyncio.gather(cancelled_task, return_exceptions=True)
        self._validate_response(response, candidates, allow_partial)
        if (
            cache_allowed
            and self.cache is not None
            and not response.missing_candidate_ids
            and len(response.scores) == len(candidates)
        ):
            await self.cache.set(
                key,
                CacheRecord(self._cache_value(response, mode), time.time()),
                ttl_seconds=self.config.cache.ttl_seconds,
            )
        return response

    @staticmethod
    def _cache_value(response: ModelResponse, mode: str) -> ModelResponse:
        if mode != "pointwise":
            return response
        score = response.scores[0]
        return ModelResponse(
            (type(score)("slot0", score.score, score.reasoning),),
            response.statistics,
            response.resolved_model,
        )

    @classmethod
    def _restore_cached(
        cls, response: ModelResponse, candidates: tuple[CandidateView[object], ...], mode: str
    ) -> ModelResponse:
        if mode != "pointwise":
            cls._validate_response(response, candidates, False)
            return response
        if len(candidates) != 1 or len(response.scores) != 1 or response.missing_candidate_ids:
            raise OutputValidationError("cached pointwise response has invalid coverage")
        score = response.scores[0]
        if score.candidate_id != "slot0":
            raise OutputValidationError("cached pointwise response has an invalid slot ID")
        validate_utility(score.score, metric_name="llm_relevance")
        return ModelResponse(
            (type(score)(candidates[0].occurrence_id, score.score, score.reasoning),),
            response.statistics,
            response.resolved_model,
        )

    async def _execute_model(
        self,
        query: str,
        candidates: tuple[CandidateView[object], ...],
        mode: str,
        allow_partial: bool,
        prompt: RerankPrompt,
        include_reasoning: bool,
        reasoning_level: str | None,
    ) -> ModelResponse:
        assert self.backend is not None
        request = ModelRequest(
            query=query,
            candidates=tuple(
                BackendCandidate(candidate.occurrence_id, candidate.text)
                for candidate in candidates
            ),
            prompt=prompt,
            mode=cast(Any, mode),
            include_reasoning=include_reasoning,
            reasoning_level=reasoning_level,
            allow_partial=allow_partial,
        )
        estimated_tokens = estimate_request_tokens(self.backend, request)
        context_limit = self.backend.capabilities.max_context_tokens
        if context_limit is not None and estimated_tokens > context_limit:
            raise CapabilityError("estimated model request exceeds its context limit")
        max_attempts = max(1, getattr(getattr(self.backend, "retry", None), "max_attempts", 1))
        chargeable_upper_bound = getattr(self.backend, "chargeable_token_upper_bound", None)
        cost_upper_bound = getattr(self.backend, "chargeable_cost_upper_bound", None)
        strict_cost = self.ledger.budget.strict_cost and self.ledger.budget.max_cost_usd is not None
        if strict_cost and not callable(cost_upper_bound):
            raise BudgetUnverifiableError(
                "strict monetary budget needs a provider-backed chargeable-cost upper bound"
            )
        per_attempt_tokens = estimated_tokens
        if callable(chargeable_upper_bound):
            upper_bound_fn = cast(Callable[..., int], chargeable_upper_bound)
            per_attempt_tokens = upper_bound_fn(
                query=query,
                candidates=tuple(candidate.text for candidate in candidates),
                prompt=prompt,
            )
            if (
                isinstance(per_attempt_tokens, bool)
                or not isinstance(per_attempt_tokens, int)
                or per_attempt_tokens < 1
            ):
                raise BudgetUnverifiableError("backend returned an invalid token upper bound")
        reserved_tokens = per_attempt_tokens * max_attempts
        estimated_cost = self._estimated_cost(reserved_tokens)
        if strict_cost:
            cost_bound_fn = cast(Callable[..., Decimal | float], cost_upper_bound)
            raw_bound = cost_bound_fn(
                query=query,
                candidates=tuple(candidate.text for candidate in candidates),
                prompt=prompt,
            )
            if isinstance(raw_bound, bool) or not isinstance(raw_bound, (Decimal, float, int)):
                raise BudgetUnverifiableError("backend returned an invalid cost upper bound")
            per_attempt_cost = Decimal(str(raw_bound))
            if not per_attempt_cost.is_finite() or per_attempt_cost < 0:
                raise BudgetUnverifiableError("backend returned an invalid cost upper bound")
            estimated_cost = per_attempt_cost * max_attempts
        await self.ledger.reserve(
            estimated_tokens=reserved_tokens,
            estimated_cost=estimated_cost,
            calls=max_attempts,
        )
        started_call = [False]
        try:
            response = await self._run_model_request(request, mode, candidates, started_call)
        except BaseException as exc:
            attempts = 1 if started_call[0] else 0
            if isinstance(exc, RerankError) and exc.details.safe_context is not None:
                reported = exc.details.safe_context.get("attempts")
                if isinstance(reported, int) and 1 <= reported <= max_attempts:
                    attempts = reported
            await self.ledger.settle_failure(
                reserved_calls=max_attempts,
                actual_calls=attempts,
                reserved_tokens=reserved_tokens,
                estimated_tokens_per_attempt=per_attempt_tokens,
                reserved_cost=estimated_cost,
            )
            if attempts:
                self.statistics.model_calls += attempts - 1
                self.statistics.retries += attempts - 1
            raise
        await self.ledger.reconcile(
            estimated_tokens=reserved_tokens,
            estimated_cost=estimated_cost,
            actual_tokens=(
                response.statistics.usage.input_tokens + response.statistics.usage.output_tokens
                if response.statistics.usage.tokens_reported
                else per_attempt_tokens * response.statistics.attempts
            ),
            actual_cost=(
                Decimal(str(response.statistics.usage.cost_usd))
                if response.statistics.usage.cost_usd is not None
                else (
                    estimated_cost * Decimal(response.statistics.attempts) / Decimal(max_attempts)
                    if estimated_cost is not None
                    else None
                )
            ),
            reserved_calls=max_attempts,
            actual_calls=response.statistics.attempts,
        )
        self._record_response(response)
        self._validate_response(response, candidates, allow_partial)
        return response

    async def _run_model_request(
        self,
        request: ModelRequest,
        mode: str,
        candidates: tuple[CandidateView[object], ...],
        started_call: list[bool],
    ) -> ModelResponse:
        assert self.backend is not None
        async with self._semaphore:
            timeout = self.ledger.remaining_seconds(self.config.timeout_s)
            self.statistics.active += 1
            self.statistics.max_active = max(self.statistics.max_active, self.statistics.active)
            self.statistics.model_calls += 1
            self.statistics.batches += 1
            call_started = time.perf_counter()
            if self.observer is not None:
                self.observer.emit(
                    TraceEvent(
                        TraceKind.MODEL_START,
                        self.context.request_id,
                        stage=mode,
                        candidate_count=len(candidates),
                    )
                )
            succeeded = False
            try:
                async with asyncio.timeout(timeout):
                    started_call[0] = True
                    response = await self.backend.score(request)
                succeeded = True
            except TimeoutError as exc:
                raise DeadlineExceededError("model call timed out") from exc
            finally:
                self.statistics.active -= 1
                elapsed_ms = (time.perf_counter() - call_started) * 1000
                if self.observer is not None:
                    self.observer.emit(
                        TraceEvent(
                            TraceKind.MODEL_END if succeeded else TraceKind.MODEL_ERROR,
                            self.context.request_id,
                            stage=mode,
                            candidate_count=len(candidates),
                            latency_ms=elapsed_ms,
                        )
                    )
                if not succeeded:
                    self.statistics.model_latency_ms += elapsed_ms
                    self.statistics.usage_complete = False
                    self.statistics.estimated_cost_usd = None
                    self.statistics.cost_missing = True
                    self.statistics.cost_confidence = CostConfidence.UNKNOWN
        return response

    def _record_response(self, response: ModelResponse) -> None:
        stats = response.statistics
        self.statistics.model_latency_ms += stats.latency_ms
        self.statistics.model_calls += max(0, stats.attempts - 1)
        self.statistics.input_tokens += stats.usage.input_tokens
        self.statistics.output_tokens += stats.usage.output_tokens
        self.statistics.usage_complete &= stats.usage.tokens_reported
        self.statistics.retries += max(0, stats.attempts - 1)
        if stats.usage.cost_usd is None:
            self.statistics.estimated_cost_usd = None
            self.statistics.cost_missing = True
            self.statistics.cost_confidence = CostConfidence.UNKNOWN
        elif not self.statistics.cost_missing:
            first_cost = self.statistics.estimated_cost_usd is None
            self.statistics.estimated_cost_usd = (
                self.statistics.estimated_cost_usd or 0.0
            ) + stats.usage.cost_usd
            if first_cost:
                self.statistics.cost_confidence = stats.usage.cost_confidence
            elif (
                self.statistics.cost_confidence is CostConfidence.UNKNOWN
                or stats.usage.cost_confidence is CostConfidence.UNKNOWN
            ):
                self.statistics.cost_confidence = CostConfidence.UNKNOWN
            elif (
                self.statistics.cost_confidence is CostConfidence.ESTIMATED
                or stats.usage.cost_confidence is CostConfidence.ESTIMATED
            ):
                self.statistics.cost_confidence = CostConfidence.ESTIMATED

    @staticmethod
    def _validate_response(
        response: ModelResponse,
        candidates: Sequence[CandidateView[object]],
        allow_partial: bool,
    ) -> None:
        expected = {candidate.occurrence_id for candidate in candidates}
        actual = [score.candidate_id for score in response.scores]
        if len(actual) != len(set(actual)):
            raise OutputValidationError("model backend returned duplicate candidate IDs")
        if not set(actual) <= expected:
            raise OutputValidationError("model backend returned unexpected candidate IDs")
        if set(actual) != expected and not allow_partial:
            raise OutputValidationError("model backend returned partial candidate coverage")
        missing = expected - set(actual)
        declared_missing = response.missing_candidate_ids
        if len(declared_missing) != len(set(declared_missing)) or set(declared_missing) != missing:
            raise OutputValidationError("model backend returned inconsistent missing IDs")
        for score in response.scores:
            validate_utility(score.score, metric_name="llm_relevance")

    async def metric_scores(
        self,
        *,
        query: str,
        candidates: Sequence[CandidateView[object]],
        metrics: WeightedMetrics[object],
    ) -> RankingOutcome:
        per_candidate: dict[str, dict[str, float]] = {
            candidate.occurrence_id: {} for candidate in candidates
        }
        for metric in metrics.metrics:
            if metrics.weights[metric.name] == 0:
                continue
            if isinstance(metric, LLMRelevance):
                if self.backend is None or metric.backend is not self.backend:
                    raise CapabilityError(
                        "LLMRelevance must use the Reranker's configured model backend"
                    )
                model_outcome = await self.model_scores(
                    query=query,
                    candidates=candidates,
                    mode="pointwise",
                    prompt=metric.prompt,
                    include_reasoning=metric.include_reasoning,
                    reasoning_level=metric.reasoning_level,
                )
                model_values = {
                    entry.occurrence_id: validate_utility(
                        cast(float, entry.score), metric_name=metric.name
                    )
                    for entry in model_outcome.entries
                }
                if set(model_values) != set(per_candidate):
                    raise OutputValidationError("LLM metric returned incomplete IDs")
                for occurrence_id, value in model_values.items():
                    per_candidate[occurrence_id][metric.name] = value
                continue
            batch_method = getattr(metric, "score_many", None)
            if batch_method is not None and inspect.iscoroutinefunction(batch_method):
                async with self._semaphore:
                    raw = await cast(BatchMetric[object], metric).score_many(
                        query, candidates, self.context
                    )
                if set(raw) != set(per_candidate):
                    raise OutputValidationError(
                        f"batch metric {metric.name!r} returned incomplete IDs"
                    )
                for occurrence_id, value in raw.items():
                    per_candidate[occurrence_id][metric.name] = validate_utility(
                        value, metric_name=metric.name
                    )
                continue

            async def evaluate(
                candidate: CandidateView[object],
                active_metric: Metric[object] = metric,
            ) -> tuple[str, float]:
                extension_identity = None
                if active_metric.cacheable and not isinstance(
                    active_metric, (LexicalRelevance, MetadataNumericMetric)
                ):
                    identity_fn = getattr(active_metric, "candidate_cache_identity", None)
                    if not callable(identity_fn):
                        raise ConfigurationError(
                            f"cacheable metric {active_metric.name!r} needs "
                            "candidate_cache_identity"
                        )
                    extension_identity = identity_fn(candidate, self.context)
                key = cache_key(
                    self.config.cache.namespace,
                    {
                        "schema": "metric-v1",
                        "tenant": self.context.tenant,
                        "authorization_revision": self.context.authorization_revision,
                        "query": query,
                        "candidate": candidate.text,
                        "metadata": candidate.metadata,
                        "metric": active_metric.cache_identity,
                        "extension_identity": extension_identity,
                        "weights": metrics.fingerprint,
                        "prompt": self.prompt.fingerprint,
                        "model": self.backend.cache_identity if self.backend else None,
                    },
                )
                if self.cache is not None and active_metric.cacheable:
                    record = await self.cache.get(key)
                    if record is not None and isinstance(record.value, (int, float)):
                        self.statistics.cache_hits += 1
                        return candidate.occurrence_id, validate_utility(
                            float(record.value), metric_name=active_metric.name
                        )
                    self.statistics.cache_misses += 1
                async with self._semaphore:
                    value = await active_metric.score(query, candidate, self.context)
                value = validate_utility(value, metric_name=active_metric.name)
                if self.cache is not None and active_metric.cacheable:
                    await self.cache.set(
                        key,
                        CacheRecord(value, time.time()),
                        ttl_seconds=self.config.cache.ttl_seconds,
                    )
                return candidate.occurrence_id, value

            window = max(1, self.config.max_concurrency * 2)
            for start in range(0, len(candidates), window):
                values = await _gather_owned(
                    *(evaluate(candidate) for candidate in candidates[start : start + window])
                )
                for occurrence_id, value in values:
                    per_candidate[occurrence_id][metric.name] = value
        entries = []
        for candidate in candidates:
            raw = per_candidate[candidate.occurrence_id]
            entries.append(
                RankingEntry(
                    candidate.occurrence_id,
                    metrics.combine(raw),
                    metrics={
                        name: MetricScore(value, utility=value, raw_kind="utility")
                        for name, value in raw.items()
                    },
                )
            )
        return RankingOutcome(tuple(entries), ScoreKind.UTILITY)
