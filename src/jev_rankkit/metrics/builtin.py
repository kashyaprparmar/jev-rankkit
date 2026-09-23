"""Dependency-light metrics."""

from __future__ import annotations

import asyncio
import inspect
import math
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Generic, TypeVar, cast

from ..backends import BackendCandidate, EmbeddingBackend, ModelBackend, ModelRequest
from ..candidates import CandidateView
from ..context import RerankContext
from ..errors import ConfigurationError, ProjectionError
from ..prompts import DEFAULT_PROMPT, RerankPrompt
from .base import validate_utility

T = TypeVar("T")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(value: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN_RE.finditer(value)}


@dataclass(frozen=True, slots=True)
class LexicalRelevance(Generic[T]):
    name: str = "lexical"
    version: str = "token-recall-v1"
    cacheable: bool = True

    @property
    def cache_identity(self) -> str:
        return f"{self.name}:{self.version}"

    async def score(self, query: str, candidate: CandidateView[T], context: RerankContext) -> float:
        return (await self.score_many(query, (candidate,), context))[candidate.occurrence_id]

    async def score_many(
        self,
        query: str,
        candidates: Sequence[CandidateView[T]],
        context: RerankContext,
    ) -> dict[str, float]:
        del context
        query_terms = _tokens(query)
        if not query_terms:
            return {candidate.occurrence_id: 0.0 for candidate in candidates}
        denominator = len(query_terms)
        return {
            candidate.occurrence_id: len(query_terms & _tokens(candidate.text)) / denominator
            for candidate in candidates
        }


@dataclass(frozen=True, slots=True)
class BM25Metric(Generic[T]):
    """Bounded candidate-pool BM25 with an explicit monotonic utility map."""

    k1: float = 1.2
    b: float = 0.75
    pivot: float = 1.0
    name: str = "bm25"
    version: str = "pool-bm25-v1"
    cacheable: bool = False

    def __post_init__(self) -> None:
        if not (math.isfinite(self.k1) and self.k1 > 0):
            raise ValueError("k1 must be finite and positive")
        if not (math.isfinite(self.b) and 0 <= self.b <= 1):
            raise ValueError("b must be in [0, 1]")
        if not (math.isfinite(self.pivot) and self.pivot > 0):
            raise ValueError("pivot must be finite and positive")

    @property
    def cache_identity(self) -> str:
        return f"{self.name}:{self.version}:{self.k1}:{self.b}:{self.pivot}"

    async def score(self, query: str, candidate: CandidateView[T], context: RerankContext) -> float:
        scores = await self.score_many(query, (candidate,), context)
        return scores[candidate.occurrence_id]

    async def score_many(
        self,
        query: str,
        candidates: Sequence[CandidateView[T]],
        context: RerankContext,
    ) -> dict[str, float]:
        del context
        if not candidates:
            return {}
        query_terms = _tokens(query)
        documents = [
            Counter(match.group(0).casefold() for match in _TOKEN_RE.finditer(view.text))
            for view in candidates
        ]
        lengths = [sum(document.values()) for document in documents]
        average_length = sum(lengths) / len(lengths) or 1.0
        document_frequency = {
            term: sum(term in document for document in documents) for term in query_terms
        }
        count = len(candidates)
        scores: dict[str, float] = {}
        for view, document, length in zip(candidates, documents, lengths, strict=True):
            raw = 0.0
            for term in query_terms:
                frequency = document.get(term, 0)
                if not frequency:
                    continue
                idf = math.log1p(
                    (count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
                )
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / average_length)
                raw += idf * frequency * (self.k1 + 1) / denominator
            scores[view.occurrence_id] = raw / (raw + self.pivot)
        return scores


@dataclass(frozen=True, slots=True)
class RecencyMetric(Generic[T]):
    metadata_key: str = "published_at"
    half_life: timedelta = timedelta(days=30)
    reject_future: bool = True
    name: str = "recency"
    version: str = "half-life-v1"
    cacheable: bool = False

    @property
    def cache_identity(self) -> str:
        return (
            f"{self.name}:{self.version}:{self.metadata_key}:"
            f"{self.half_life.total_seconds()}:{self.reject_future}"
        )

    def __post_init__(self) -> None:
        if self.half_life.total_seconds() <= 0:
            raise ValueError("half_life must be positive")

    async def score(self, query: str, candidate: CandidateView[T], context: RerankContext) -> float:
        del query
        value = candidate.metadata.get(self.metadata_key)
        if not isinstance(value, datetime):
            raise ProjectionError(f"metadata {self.metadata_key!r} must be a datetime")
        age = (context.as_of - value).total_seconds()
        if age < 0:
            if self.reject_future:
                raise ProjectionError(f"metadata {self.metadata_key!r} is in the future")
            age = 0
        return 2 ** (-age / self.half_life.total_seconds())


@dataclass(frozen=True, slots=True)
class MetadataNumericMetric(Generic[T]):
    metadata_key: str
    minimum: float = 0.0
    maximum: float = 1.0
    name: str = "metadata_numeric"
    version: str = "range-v1"
    cacheable: bool = True

    @property
    def cache_identity(self) -> str:
        return (
            f"{self.name}:{self.version}:{self.metadata_key}:"
            f"{self.minimum:.17g}:{self.maximum:.17g}"
        )

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("metric bounds must be finite")
        if self.maximum <= self.minimum:
            raise ValueError("maximum must be greater than minimum")

    async def score(self, query: str, candidate: CandidateView[T], context: RerankContext) -> float:
        del query, context
        value = candidate.metadata.get(self.metadata_key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProjectionError(f"metadata {self.metadata_key!r} must be numeric")
        numeric = float(value)
        if not self.minimum <= numeric <= self.maximum:
            raise ProjectionError(f"metadata {self.metadata_key!r} is outside configured bounds")
        return (numeric - self.minimum) / (self.maximum - self.minimum)


MetricCallback = Callable[[str, T, RerankContext], float | Awaitable[float]]


@dataclass(frozen=True, slots=True)
class CallableMetric(Generic[T]):
    name: str
    callback: MetricCallback[T]
    version: str = "1"
    cacheable: bool = False
    cache_key_fn: Callable[[T, RerankContext], str] | None = None

    def __post_init__(self) -> None:
        if self.cacheable and self.cache_key_fn is None:
            raise ConfigurationError("cacheable callbacks require cache_key_fn")

    @property
    def cache_identity(self) -> str:
        return f"{self.name}:{self.version}:callback"

    def candidate_cache_identity(self, candidate: CandidateView[T], context: RerankContext) -> str:
        if self.cache_key_fn is None:
            raise ConfigurationError("cacheable callbacks require cache_key_fn")
        key = self.cache_key_fn(candidate.item, context)
        if not isinstance(key, str) or not key:
            raise ConfigurationError("cache_key_fn must return a non-empty string")
        return key

    async def score(self, query: str, candidate: CandidateView[T], context: RerankContext) -> float:
        if inspect.iscoroutinefunction(self.callback):
            value = self.callback(query, candidate.item, context)
        else:
            value = await asyncio.to_thread(
                cast(Any, self.callback), query, candidate.item, context
            )
        if inspect.isawaitable(value):
            value = await value
        return validate_utility(value, metric_name=self.name)


@dataclass(frozen=True, slots=True)
class EmbeddingSimilarity(Generic[T]):
    backend: EmbeddingBackend
    name: str = "semantic"
    version: str = "cosine-v1"
    cacheable: bool = False
    batch_size: int = 32

    @property
    def cache_identity(self) -> str:
        return f"{self.name}:{self.version}:{self.backend.cache_identity}"

    async def score(self, query: str, candidate: CandidateView[T], context: RerankContext) -> float:
        scores = await self.score_many(query, (candidate,), context)
        return scores[candidate.occurrence_id]

    async def score_many(
        self,
        query: str,
        candidates: Sequence[CandidateView[T]],
        context: RerankContext,
    ) -> dict[str, float]:
        del context
        if self.batch_size < 1:
            raise ProjectionError("embedding batch_size must be positive")
        query_method = getattr(self.backend, "embed_query", None)
        if callable(query_method):
            query_vector = await query_method(query)
        else:
            query_result = await self.backend.embed((query,))
            if len(query_result) != 1:
                raise ProjectionError("embedding backend returned the wrong query vector count")
            query_vector = query_result[0]
        scores: dict[str, float] = {}
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            document_method = getattr(self.backend, "embed_documents", None)
            vectors = (
                await document_method([candidate.text for candidate in batch])
                if callable(document_method)
                else await self.backend.embed([candidate.text for candidate in batch])
            )
            if len(vectors) != len(batch):
                raise ProjectionError("embedding backend returned the wrong vector count")
            scores.update(
                {
                    candidate.occurrence_id: _mapped_cosine(query_vector, vector)
                    for candidate, vector in zip(batch, vectors, strict=True)
                }
            )
        return scores


def _mapped_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ProjectionError("embedding vectors must have equal non-zero dimensions")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ProjectionError("embedding vectors must have non-zero norms")
    cosine = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    return (cosine + 1) / 2


@dataclass(frozen=True, slots=True)
class LLMRelevance(Generic[T]):
    backend: ModelBackend
    prompt: RerankPrompt = DEFAULT_PROMPT
    include_reasoning: bool = False
    reasoning_level: str | None = None
    name: str = "llm_relevance"
    version: str = "noul-relevance-v1"
    cacheable: bool = True

    @property
    def cache_identity(self) -> str:
        return (
            f"{self.name}:{self.version}:{self.backend.cache_identity}:"
            f"{self.prompt.fingerprint}:{self.include_reasoning}:{self.reasoning_level}"
        )

    async def score(self, query: str, candidate: CandidateView[T], context: RerankContext) -> float:
        del context
        response = await self.backend.score(
            ModelRequest(
                query=query,
                candidates=(BackendCandidate(candidate.occurrence_id, candidate.text),),
                prompt=self.prompt,
                mode="pointwise",
                include_reasoning=self.include_reasoning,
                reasoning_level=self.reasoning_level,
            )
        )
        if len(response.scores) != 1 or response.scores[0].candidate_id != candidate.occurrence_id:
            raise ProjectionError("model backend returned invalid pointwise coverage")
        return validate_utility(response.scores[0].score, metric_name=self.name)
