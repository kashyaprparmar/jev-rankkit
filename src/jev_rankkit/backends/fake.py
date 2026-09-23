"""Deterministic backend used by tests and examples."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable

from ..errors import OutputValidationError
from ..types import RequestStatistics, Usage
from .base import (
    BackendCapabilities,
    CandidateScore,
    ModelBackend,
    ModelRequest,
    ModelResponse,
)


class FakeModelBackend(ModelBackend):
    cache_stable = True

    def __init__(
        self,
        scores: dict[str, float] | Callable[[str, str], float] | None = None,
        *,
        delay_s: float = 0.0,
        fail: Exception | None = None,
        supports_listwise: bool = True,
        malformed: bool = False,
    ) -> None:
        self._scores = scores or {}
        self.delay_s = delay_s
        self.fail = fail
        self.malformed = malformed
        self.calls: list[ModelRequest] = []
        self.active_calls = 0
        self.max_active_calls = 0
        self._capabilities = BackendCapabilities(
            pointwise=True,
            listwise=supports_listwise,
            explanations=True,
            reasoning_levels=("low", "medium", "high"),
            usage_reporting=True,
        )

    @property
    def backend_id(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-v1"

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def cache_identity(self) -> str:
        return "fake:fake-v1"

    async def score(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            if self.fail is not None:
                raise self.fail
            results: list[CandidateScore] = []
            for candidate in request.candidates:
                if callable(self._scores):
                    value = self._scores(request.query, candidate.text)
                else:
                    value = self._scores.get(candidate.candidate_id, 0.5)
                if self.malformed or not math.isfinite(value) or not 0 <= value <= 1:
                    raise OutputValidationError("fake backend produced an invalid score")
                results.append(
                    CandidateScore(
                        candidate_id=candidate.candidate_id,
                        score=value,
                        reasoning="fake explanation" if request.include_reasoning else None,
                    )
                )
            return ModelResponse(
                scores=tuple(results),
                statistics=RequestStatistics(
                    latency_ms=self.delay_s * 1000,
                    model=self.model,
                    usage=Usage(
                        input_tokens=len(request.candidates) * 10,
                        output_tokens=2,
                        tokens_reported=True,
                    ),
                ),
                resolved_model=self.model,
            )
        finally:
            self.active_calls -= 1

    async def aclose(self) -> None:
        return None
