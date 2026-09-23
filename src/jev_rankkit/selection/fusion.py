"""Deterministic reciprocal-rank fusion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ..errors import ConfigurationError
from ..types import RankingEntry, RankingOutcome, ScoreKind


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    *,
    weights: Sequence[float] | None = None,
    constant: float = 60.0,
    input_order: Mapping[str, int] | None = None,
) -> RankingOutcome:
    if not math.isfinite(constant) or constant <= 0:
        raise ConfigurationError("RRF constant must be finite and positive")
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ConfigurationError("RRF weights must match the number of ranked lists")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ConfigurationError("RRF weights must be finite and non-negative")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0
    for ranked, weight in zip(ranked_lists, weights, strict=True):
        if len(set(ranked)) != len(ranked):
            raise ConfigurationError("each RRF input list must contain unique IDs")
        for rank, candidate_id in enumerate(ranked, start=1):
            if candidate_id not in first_seen:
                first_seen[candidate_id] = seen_counter
                seen_counter += 1
            scores[candidate_id] = scores.get(candidate_id, 0.0) + weight / (constant + rank)
    order = input_order or first_seen
    entries = [RankingEntry(candidate_id, score) for candidate_id, score in scores.items()]
    entries.sort(key=lambda item: (-cast_score(item.score), order.get(item.occurrence_id, 10**12)))
    return RankingOutcome(tuple(entries), ScoreKind.FUSION)


def cast_score(value: float | None) -> float:
    return value if value is not None else float("-inf")
