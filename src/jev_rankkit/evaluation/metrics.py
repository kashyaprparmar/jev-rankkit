"""Offline ranking-quality metrics over position-aligned relevance judgments."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast


class EvaluationMetric(Protocol):
    @property
    def name(self) -> str: ...

    def score(
        self, ranked_indices: Sequence[int], relevance: Sequence[float]
    ) -> float | Awaitable[float]: ...


def _cutoff(k: int | None, ranked: Sequence[int]) -> int:
    if k is not None and (isinstance(k, bool) or k < 1):
        raise ValueError("evaluation cutoff k must be positive")
    return len(ranked) if k is None else k


def _relevant(value: float) -> bool:
    return value > 0


@dataclass(frozen=True, slots=True)
class Precision:
    k: int

    @property
    def name(self) -> str:
        return f"precision@{self.k}"

    def score(self, ranked_indices: Sequence[int], relevance: Sequence[float]) -> float:
        cutoff = _cutoff(self.k, ranked_indices)
        return sum(_relevant(relevance[index]) for index in ranked_indices[:cutoff]) / cutoff


@dataclass(frozen=True, slots=True)
class Recall:
    k: int

    @property
    def name(self) -> str:
        return f"recall@{self.k}"

    def score(self, ranked_indices: Sequence[int], relevance: Sequence[float]) -> float:
        cutoff = _cutoff(self.k, ranked_indices)
        total = sum(map(_relevant, relevance))
        return (
            sum(_relevant(relevance[index]) for index in ranked_indices[:cutoff]) / total
            if total
            else 0.0
        )


@dataclass(frozen=True, slots=True)
class HitRate:
    k: int

    @property
    def name(self) -> str:
        return f"hit_rate@{self.k}"

    def score(self, ranked_indices: Sequence[int], relevance: Sequence[float]) -> float:
        cutoff = _cutoff(self.k, ranked_indices)
        return float(any(_relevant(relevance[index]) for index in ranked_indices[:cutoff]))


@dataclass(frozen=True, slots=True)
class Success(HitRate):
    @property
    def name(self) -> str:
        return f"success@{self.k}"


@dataclass(frozen=True, slots=True)
class MRR:
    k: int | None = None

    @property
    def name(self) -> str:
        return "mrr" if self.k is None else f"mrr@{self.k}"

    def score(self, ranked_indices: Sequence[int], relevance: Sequence[float]) -> float:
        cutoff = _cutoff(self.k, ranked_indices)
        for rank, index in enumerate(ranked_indices[:cutoff], start=1):
            if _relevant(relevance[index]):
                return 1 / rank
        return 0.0


@dataclass(frozen=True, slots=True)
class MAP:
    k: int | None = None

    @property
    def name(self) -> str:
        return "map" if self.k is None else f"map@{self.k}"

    def score(self, ranked_indices: Sequence[int], relevance: Sequence[float]) -> float:
        cutoff = _cutoff(self.k, ranked_indices)
        relevant_count = sum(map(_relevant, relevance))
        if relevant_count == 0:
            return 0.0
        found = 0
        precision_sum = 0.0
        for rank, index in enumerate(ranked_indices[:cutoff], start=1):
            if _relevant(relevance[index]):
                found += 1
                precision_sum += found / rank
        denominator = min(relevant_count, cutoff) if self.k is not None else relevant_count
        return precision_sum / denominator


@dataclass(frozen=True, slots=True)
class NDCG:
    k: int

    @property
    def name(self) -> str:
        return f"ndcg@{self.k}"

    def score(self, ranked_indices: Sequence[int], relevance: Sequence[float]) -> float:
        cutoff = _cutoff(self.k, ranked_indices)
        maximum = max(relevance, default=0.0)
        if maximum == 0:
            return 0.0

        def dcg(values: Sequence[float]) -> float:
            return sum(
                (math.exp2(value - maximum) - math.exp2(-maximum)) / math.log2(rank + 1)
                for rank, value in enumerate(values[:cutoff], start=1)
            )

        actual = dcg([relevance[index] for index in ranked_indices[:cutoff]])
        ideal = dcg(sorted(relevance, reverse=True))
        return actual / ideal if ideal else 0.0


EvaluationCallback = Callable[[Sequence[int], Sequence[float]], float | Awaitable[float]]


@dataclass(frozen=True, slots=True)
class CallableEvaluationMetric:
    name: str
    callback: EvaluationCallback

    def score(
        self, ranked_indices: Sequence[int], relevance: Sequence[float]
    ) -> float | Awaitable[float]:
        return self.callback(ranked_indices, relevance)


async def score_metric(
    metric: EvaluationMetric, ranked_indices: Sequence[int], relevance: Sequence[float]
) -> float:
    callback = metric.score
    if inspect.iscoroutinefunction(callback):
        value = callback(ranked_indices, relevance)
    else:
        value = await asyncio.to_thread(cast(Any, callback), ranked_indices, relevance)
    if inspect.isawaitable(value):
        value = await value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"evaluation metric {metric.name!r} returned a non-finite number")
    return float(value)
