"""Public immutable result, statistics, and execution-plan models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Generic, Literal, TypeVar

T_co = TypeVar("T_co", covariant=True)


class ScoreKind(StrEnum):
    UTILITY = "utility"
    RAW_METRIC = "raw_metric"
    RELATIVE_PREFERENCE = "relative_preference"
    FUSION = "fusion"
    RANK_ONLY = "rank_only"


class ResultStatus(StrEnum):
    OK = "ok"
    NOOP = "noop"
    FALLBACK = "fallback"
    PARTIAL = "partial"


class CostConfidence(StrEnum):
    KNOWN = "known"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class MetricStatus(StrEnum):
    OK = "ok"
    MISSING = "missing"
    ERROR = "error"
    CACHED = "cached"


@dataclass(frozen=True, slots=True)
class MetricScore:
    raw_value: float | None
    utility: float | None = None
    raw_kind: str = "utility"
    status: MetricStatus = MetricStatus.OK
    confidence: float | None = None
    reasoning: str | None = None
    normalizer: str | None = None


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    cost_confidence: CostConfidence = CostConfidence.UNKNOWN
    tokens_reported: bool = False


@dataclass(frozen=True, slots=True)
class RequestStatistics:
    latency_ms: float = 0.0
    model: str | None = None
    attempts: int = 1
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True, slots=True)
class RerankStatistics:
    total_latency_ms: float = 0.0
    model_latency_ms: float = 0.0
    candidate_count: int = 0
    eligible_count: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usage_complete: bool = True
    estimated_cost_usd: float | None = None
    cost_confidence: CostConfidence = CostConfidence.UNKNOWN
    cache_hits: int = 0
    cache_misses: int = 0
    retries: int = 0
    fallbacks: int = 0
    batches: int = 0
    max_concurrency: int = 0
    observer_events_dropped: int = 0
    observer_errors: int = 0


@dataclass(frozen=True, slots=True)
class ExecutionStage:
    name: str
    strategy: str
    input_count: int
    output_count: int
    backend: str | None = None
    approximate: bool = False
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    strategy: str
    stages: tuple[ExecutionStage, ...] = ()
    approximate: bool = False
    rationale: tuple[str, ...] = ()
    estimated_model_calls: int | None = None
    estimated_tokens: int | None = None
    estimated_cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class Coverage:
    input_count: int = 0
    eligible_count: int = 0
    scored_count: int = 0
    selected_count: int = 0
    missing_count: int = 0
    pruned_count: int = 0


@dataclass(frozen=True, slots=True)
class RerankResult(Generic[T_co]):
    item: T_co
    score: float | None
    rank: int
    metrics: Mapping[str, MetricScore] = field(default_factory=dict)
    reasoning: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    input_index: int = 0
    occurrence_id: str = ""
    candidate_id: str | None = None
    score_kind: ScoreKind = ScoreKind.UTILITY
    selection_score: float | None = None
    role: Literal["ranked", "dependency"] = "ranked"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class RerankResponse(Generic[T_co]):
    results: tuple[RerankResult[T_co], ...]
    stats: RerankStatistics = field(default_factory=RerankStatistics)
    execution_plan: ExecutionPlan = field(default_factory=lambda: ExecutionPlan("unknown"))
    status: ResultStatus = ResultStatus.OK
    coverage: Coverage = field(default_factory=Coverage)
    warnings: tuple[str, ...] = ()
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class RankingEntry:
    occurrence_id: str
    score: float | None
    metrics: Mapping[str, MetricScore] = field(default_factory=dict)
    reasoning: str | None = None
    selection_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True, slots=True)
class RankingOutcome:
    entries: tuple[RankingEntry, ...]
    score_kind: ScoreKind = ScoreKind.UTILITY
    approximate: bool = False
    warnings: tuple[str, ...] = ()
    stages: tuple[ExecutionStage, ...] = ()
