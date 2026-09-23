"""Reproducible offline evaluation with explicit judgment and coverage rules."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, Literal, TypeVar

from ..api import Reranker
from ..errors import EvaluationError
from ..types import RerankStatistics, ResultStatus
from .metrics import MAP, MRR, EvaluationMetric, score_metric

T = TypeVar("T")
UnjudgedPolicy = Literal["error", "zero"]


@dataclass(frozen=True, slots=True)
class EvaluationCase(Generic[T]):
    case_id: str
    query: str
    candidates: tuple[T, ...]
    relevance: tuple[float | None, ...]
    text_fn: Callable[[T], str] | None = None

    def __post_init__(self) -> None:
        if not self.case_id or not self.query.strip():
            raise EvaluationError("case_id and query must be non-empty")
        if len(self.candidates) != len(self.relevance):
            raise EvaluationError("relevance must align with candidate positions")
        for value in self.relevance:
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise EvaluationError("relevance grades must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EvaluationDataset(Generic[T]):
    name: str
    version: str
    cases: tuple[EvaluationCase[T], ...]

    def __post_init__(self) -> None:
        if not self.name or not self.version or not self.cases:
            raise EvaluationError("dataset name, version, and cases are required")
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise EvaluationError("evaluation case IDs must be unique")


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    metrics: Mapping[str, float]
    ranked_indices: tuple[int, ...]
    stats: RerankStatistics
    status: ResultStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True, slots=True)
class EvaluationPerformance:
    elapsed_ms: float
    throughput_cases_per_s: float
    model_calls: int
    input_tokens: int
    output_tokens: int
    usage_complete: bool
    estimated_cost_usd: float | None
    cache_hit_rate: float | None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    dataset_name: str
    dataset_version: str
    metrics: Mapping[str, float]
    cases: tuple[CaseEvaluation, ...]
    performance: EvaluationPerformance
    unjudged_policy: UnjudgedPolicy = "error"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


async def evaluate(
    reranker: Reranker,
    dataset: EvaluationDataset[T],
    *,
    metrics: Sequence[EvaluationMetric],
    rerank_top_k: int | None = None,
    unjudged_policy: UnjudgedPolicy = "error",
    allow_partial: bool = False,
) -> EvaluationReport:
    """Evaluate a fixed candidate pool; never add oracle-relevant candidates."""
    if not metrics:
        raise EvaluationError("at least one evaluation metric is required")
    names = [metric.name for metric in metrics]
    if len(names) != len(set(names)):
        raise EvaluationError("evaluation metric names must be unique")
    if unjudged_policy not in {"error", "zero"}:
        raise EvaluationError("unjudged_policy must be 'error' or 'zero'")
    if rerank_top_k is not None and (
        isinstance(rerank_top_k, bool) or not isinstance(rerank_top_k, int) or rerank_top_k < 1
    ):
        raise EvaluationError("rerank_top_k must be positive or None")
    for metric in metrics:
        cutoff = getattr(metric, "k", None)
        if rerank_top_k is not None and (
            (cutoff is None and isinstance(metric, (MRR, MAP)))
            or (isinstance(cutoff, int) and cutoff > rerank_top_k)
        ):
            raise EvaluationError("metric cutoff requires a longer reranker output")

    started = time.perf_counter()
    results: list[CaseEvaluation] = []
    for case in dataset.cases:
        if any(value is None for value in case.relevance) and unjudged_policy == "error":
            raise EvaluationError(f"case {case.case_id!r} has unjudged candidates")
        labels = tuple(float(value or 0.0) for value in case.relevance)
        response = await reranker.rerank(
            query=case.query,
            candidates=case.candidates,
            text_fn=case.text_fn,
            top_k=rerank_top_k,
        )
        if response.status is ResultStatus.PARTIAL and not allow_partial:
            raise EvaluationError(f"case {case.case_id!r} returned partial output")
        ordered = tuple(item.input_index for item in response.results)
        if len(set(ordered)) != len(ordered) or any(
            index < 0 or index >= len(labels) for index in ordered
        ):
            raise EvaluationError("reranker returned duplicate or unknown input positions")
        if rerank_top_k is None and len(ordered) != len(labels) and not allow_partial:
            raise EvaluationError("full-order evaluation requires every candidate")
        if (
            rerank_top_k is not None
            and len(ordered) < min(rerank_top_k, len(labels))
            and not allow_partial
        ):
            raise EvaluationError("top-k evaluation requires a complete requested shortlist")
        metric_values: dict[str, float] = {}
        for metric in metrics:
            try:
                metric_values[metric.name] = await score_metric(metric, ordered, labels)
            except (ValueError, OverflowError) as exc:
                raise EvaluationError(f"metric {metric.name!r} failed") from exc
        results.append(
            CaseEvaluation(case.case_id, metric_values, ordered, response.stats, response.status)
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    stats = [case.stats for case in results]
    cache_hits = sum(item.cache_hits for item in stats)
    cache_lookups = cache_hits + sum(item.cache_misses for item in stats)
    model_calls = sum(item.model_calls for item in stats)
    cost_complete = all(item.estimated_cost_usd is not None for item in stats if item.model_calls)
    performance = EvaluationPerformance(
        elapsed_ms=elapsed_ms,
        throughput_cases_per_s=len(results) * 1000 / elapsed_ms if elapsed_ms else math.inf,
        model_calls=model_calls,
        input_tokens=sum(item.input_tokens for item in stats),
        output_tokens=sum(item.output_tokens for item in stats),
        usage_complete=all(item.usage_complete for item in stats),
        estimated_cost_usd=(
            sum(item.estimated_cost_usd or 0.0 for item in stats) if cost_complete else None
        ),
        cache_hit_rate=cache_hits / cache_lookups if cache_lookups else None,
    )
    return EvaluationReport(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        metrics={name: statistics.mean(case.metrics[name] for case in results) for name in names},
        cases=tuple(results),
        performance=performance,
        unjudged_policy=unjudged_policy,
        notes=("Unjudged candidates were explicitly treated as grade zero.",)
        if unjudged_policy == "zero"
        else (),
    )
