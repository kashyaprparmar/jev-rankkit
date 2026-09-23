"""Explicitly opt-in provider contract check; never runs billable calls by default."""

from __future__ import annotations

import os

import pytest

from jev_rankkit.backends import BackendCandidate, JevBackend, ModelRequest
from jev_rankkit.config import RetryConfig
from jev_rankkit.prompts import DEFAULT_PROMPT

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_live_jev_noul_contract() -> None:
    if os.getenv("JEV_RANKKIT_LIVE") != "1":
        pytest.skip("set JEV_RANKKIT_LIVE=1 to permit a billable provider request")
    if not os.getenv("TYPESAFE_API_KEY"):
        pytest.skip("TYPESAFE_API_KEY is required for the opt-in provider check")
    backend = JevBackend(
        model=os.getenv("JEV_RANKKIT_MODEL", "jev-1.13.0"),
        retry=RetryConfig(max_attempts=1),
    )
    try:
        response = await backend.score(
            ModelRequest(
                query="Which candidate describes vector search?",
                candidates=(
                    BackendCandidate("candidate-a", "A vector index finds similar vectors."),
                    BackendCandidate("candidate-b", "A recipe for tomato soup."),
                ),
                prompt=DEFAULT_PROMPT,
                mode="listwise",
            )
        )
        assert {score.candidate_id for score in response.scores} == {"candidate-a", "candidate-b"}
        assert all(0 <= score.score <= 1 for score in response.scores)
        assert response.statistics.attempts == 1
    finally:
        await backend.aclose()
