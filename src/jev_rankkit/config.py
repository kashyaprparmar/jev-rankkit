"""Immutable configuration and budget models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from .errors import ConfigurationError
from .observability import Observer


class FallbackPolicy(StrEnum):
    STRICT = "strict"
    PREVIOUS_STAGE = "previous_stage"
    SMALLER_BATCHES = "smaller_batches"
    POINTWISE = "pointwise"
    LEXICAL = "lexical"
    INPUT_ORDER = "input_order"
    PARTIAL = "partial"


class TruncationPolicy(StrEnum):
    ERROR = "error"
    HEAD = "head"
    HEAD_TAIL = "head_tail"


@dataclass(frozen=True, slots=True)
class Budget:
    max_cost_usd: Decimal | float | int | None = None
    max_tokens: int | None = None
    max_latency_ms: float | None = None
    max_model_calls: int | None = None
    strict_cost: bool = False

    def __post_init__(self) -> None:
        if self.max_cost_usd is not None:
            if isinstance(self.max_cost_usd, bool) or not isinstance(
                self.max_cost_usd, (Decimal, float, int)
            ):
                raise ConfigurationError("max_cost_usd must be a finite non-negative number")
            amount = Decimal(str(self.max_cost_usd))
            if not amount.is_finite() or amount < 0:
                raise ConfigurationError("max_cost_usd must be a finite non-negative number")
            object.__setattr__(self, "max_cost_usd", amount)
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 0
        ):
            raise ConfigurationError("max_tokens must be a non-negative integer")
        if self.max_model_calls is not None and (
            isinstance(self.max_model_calls, bool)
            or not isinstance(self.max_model_calls, int)
            or self.max_model_calls < 0
        ):
            raise ConfigurationError("max_model_calls must be a non-negative integer")
        if self.max_latency_ms is not None and (
            not math.isfinite(self.max_latency_ms) or self.max_latency_ms < 0
        ):
            raise ConfigurationError("max_latency_ms must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RetryConfig:
    max_attempts: int = 3
    initial_backoff_s: float = 0.1
    max_backoff_s: float = 2.0
    jitter_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ConfigurationError("max_attempts must be at least 1")
        if self.initial_backoff_s < 0 or self.max_backoff_s < 0:
            raise ConfigurationError("backoff values must be non-negative")
        if not 0 <= self.jitter_ratio <= 1:
            raise ConfigurationError("jitter_ratio must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class CacheConfig:
    enabled: bool = False
    max_entries: int = 1024
    ttl_seconds: float | None = 3600.0
    namespace: str = "default"

    def __post_init__(self) -> None:
        if self.max_entries < 1:
            raise ConfigurationError("max_entries must be positive")
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ConfigurationError("ttl_seconds must be positive")


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    max_concurrency: int = 8
    batch_size: int = 16
    max_candidates: int = 10_000
    max_candidate_chars: int = 32_000
    timeout_s: float = 30.0
    retry: RetryConfig = field(default_factory=RetryConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    fallback_chain: tuple[FallbackPolicy, ...] = (FallbackPolicy.STRICT,)
    budget: Budget = field(default_factory=Budget)
    score_threshold: float | None = None
    truncation: TruncationPolicy = TruncationPolicy.ERROR
    include_reasoning: bool = False
    reasoning_level: str | None = None
    observer: Observer | None = None
    observer_queue_size: int = 128

    def __post_init__(self) -> None:
        if self.max_concurrency < 1 or self.batch_size < 1:
            raise ConfigurationError("max_concurrency and batch_size must be positive")
        if self.observer_queue_size < 1:
            raise ConfigurationError("observer_queue_size must be positive")
        if self.max_candidates < 1 or self.max_candidate_chars < 1:
            raise ConfigurationError("candidate limits must be positive")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ConfigurationError("timeout_s must be finite and positive")
        if self.score_threshold is not None and not 0 <= self.score_threshold <= 1:
            raise ConfigurationError("score_threshold must be between 0 and 1")
        if not self.fallback_chain:
            raise ConfigurationError("fallback_chain cannot be empty")
        if len(set(self.fallback_chain)) != len(self.fallback_chain):
            raise ConfigurationError("fallback_chain cannot contain duplicate policies")
