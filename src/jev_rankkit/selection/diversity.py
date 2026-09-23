"""Greedy MMR selection with deterministic ties and incremental similarity state."""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar, cast

from ..candidates import CandidateView
from ..errors import ConfigurationError
from ..types import RankingEntry, RankingOutcome

T = TypeVar("T")
Similarity = Callable[[CandidateView[T], CandidateView[T]], float | Awaitable[float]]
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def lexical_similarity(left: CandidateView[T], right: CandidateView[T]) -> float:
    left_tokens = {match.group(0).casefold() for match in _TOKEN_RE.finditer(left.text)}
    right_tokens = {match.group(0).casefold() for match in _TOKEN_RE.finditer(right.text)}
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


async def mmr_select(
    outcome: RankingOutcome,
    candidates: Sequence[CandidateView[T]],
    *,
    top_k: int,
    lambda_: float = 0.7,
    similarity: Similarity[T] | None = None,
) -> RankingOutcome:
    if not 0 <= lambda_ <= 1:
        raise ConfigurationError("MMR lambda must be between 0 and 1")
    if top_k < 0:
        raise ConfigurationError("MMR top_k must be non-negative")
    views = {candidate.occurrence_id: candidate for candidate in candidates}
    base = {entry.occurrence_id: entry for entry in outcome.entries}
    if any(entry.score is None or not 0 <= entry.score <= 1 for entry in outcome.entries):
        raise ConfigurationError("MMR requires bounded relevance scores in [0, 1]")
    missing = set(base) - set(views)
    if missing:
        raise ConfigurationError("MMR outcome contains unknown candidate IDs")
    remaining = list(base)
    selected: list[str] = []
    max_similarity = {candidate_id: 0.0 for candidate_id in remaining}
    input_index = {candidate.occurrence_id: candidate.input_index for candidate in candidates}
    similarity_fn = similarity or lexical_similarity
    selection_scores: dict[str, float] = {}
    while remaining and len(selected) < top_k:
        scored = [
            (
                lambda_ * cast(float, base[candidate_id].score)
                - (1 - lambda_) * max_similarity[candidate_id],
                candidate_id,
            )
            for candidate_id in remaining
        ]
        best_score, best = min(
            scored,
            key=lambda pair: (-pair[0], input_index[pair[1]]),
        )
        selected.append(best)
        remaining.remove(best)
        selection_scores[best] = best_score
        for candidate_id in remaining:
            value = similarity_fn(views[candidate_id], views[best])
            if inspect.isawaitable(value):
                value = await value
            numeric = float(value)
            if not math.isfinite(numeric) or not 0 <= numeric <= 1:
                raise ConfigurationError("MMR similarity must be in [0, 1]")
            max_similarity[candidate_id] = max(max_similarity[candidate_id], numeric)
    entries = tuple(
        RankingEntry(
            occurrence_id=candidate_id,
            score=base[candidate_id].score,
            metrics=base[candidate_id].metrics,
            reasoning=base[candidate_id].reasoning,
            selection_score=selection_scores[candidate_id],
        )
        for candidate_id in selected
    )
    return RankingOutcome(
        entries,
        outcome.score_kind,
        approximate=outcome.approximate,
        warnings=outcome.warnings,
        stages=outcome.stages,
    )
