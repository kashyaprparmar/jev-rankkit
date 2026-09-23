from __future__ import annotations

import asyncio

import pytest

from jev_rankkit import Reranker
from jev_rankkit.backends import (
    BackendCapabilities,
    CandidateScore,
    FakeModelBackend,
    ModelResponse,
)
from jev_rankkit.cache import CacheRecord, MemoryCache
from jev_rankkit.config import CacheConfig, RerankerConfig
from jev_rankkit.errors import CapabilityError, OutputValidationError
from jev_rankkit.prompts import RerankPrompt
from jev_rankkit.strategies import ListwiseStrategy
from jev_rankkit.types import RequestStatistics, ResultStatus


@pytest.mark.asyncio
async def test_pointwise_uses_bounded_concurrency() -> None:
    backend = FakeModelBackend(lambda _query, text: float(text) / 10, delay_s=0.01)
    config = RerankerConfig(max_concurrency=3, cache=CacheConfig(enabled=False))
    response = await Reranker(backend, strategy="pointwise", config=config).rerank(
        query="numbers", candidates=[str(index) for index in range(8)]
    )
    assert backend.max_active_calls == 3
    assert response.stats.max_concurrency == 3
    assert [result.item for result in response.results[:2]] == ["7", "6"]


@pytest.mark.asyncio
async def test_listwise_rejects_uncalibrated_chunks() -> None:
    backend = FakeModelBackend(lambda _query, text: float(text) / 10)
    with pytest.raises(CapabilityError, match="cannot be merged"):
        await Reranker(backend, strategy=ListwiseStrategy(batch_size=2)).rerank(
            query="numbers", candidates=["1", "3", "2", "4"], top_k=3
        )
    assert backend.calls == []


@pytest.mark.asyncio
async def test_listwise_obeys_backend_batch_limit_before_dispatch() -> None:
    class LimitedBackend(FakeModelBackend):
        @property
        def capabilities(self) -> BackendCapabilities:
            return BackendCapabilities(pointwise=True, listwise=True, max_batch_size=1)

    backend = LimitedBackend()
    with pytest.raises(CapabilityError, match="batch limit"):
        await Reranker(backend, strategy="listwise").rerank(query="q", candidates=["a", "b"])
    assert backend.calls == []


class PartialBackend(FakeModelBackend):
    async def score(self, request):  # type: ignore[no-untyped-def]
        response = await super().score(request)
        return ModelResponse(
            scores=response.scores[:-1],
            statistics=RequestStatistics(model=self.model),
            resolved_model=self.model,
            missing_candidate_ids=(request.candidates[-1].candidate_id,),
        )


@pytest.mark.asyncio
async def test_partial_listwise_is_rejected_by_default() -> None:
    with pytest.raises(OutputValidationError):
        await Reranker(PartialBackend(), strategy=ListwiseStrategy(batch_size=3)).rerank(
            query="q", candidates=["a", "b", "c"]
        )


@pytest.mark.asyncio
async def test_partial_listwise_can_be_explicit() -> None:
    response = await Reranker(
        PartialBackend(), strategy=ListwiseStrategy(batch_size=3, allow_partial=True)
    ).rerank(query="q", candidates=["a", "b", "c"])
    assert response.status is ResultStatus.PARTIAL
    assert len(response.results) == 2
    assert response.coverage.missing_count == 1


@pytest.mark.asyncio
async def test_cache_deduplicates_and_prompt_change_invalidates() -> None:
    backend = FakeModelBackend(lambda _query, _text: 0.7)
    cache = MemoryCache()
    ranker = Reranker(backend, strategy="pointwise", cache=cache)
    await ranker.rerank(query="q", candidates=["a"])
    second = await ranker.rerank(query="q", candidates=["a"])
    await ranker.rerank(
        query="q",
        candidates=["a"],
        prompt=RerankPrompt(criteria={"different": "Different criterion"}),
    )
    assert len(backend.calls) == 2
    assert second.stats.cache_hits == 1


@pytest.mark.asyncio
async def test_listwise_backend_score_order_is_not_candidate_order() -> None:
    class ReversedBackend(FakeModelBackend):
        async def score(self, request):  # type: ignore[no-untyped-def]
            response = await super().score(request)
            return ModelResponse(
                tuple(reversed(response.scores)), response.statistics, response.resolved_model
            )

    backend = ReversedBackend(lambda _query, text: 0.9 if text == "best" else 0.1)
    response = await Reranker(backend, strategy="listwise").rerank(
        query="q", candidates=["best", "other"]
    )
    assert [(item.item, item.score) for item in response.results] == [
        ("best", 0.9),
        ("other", 0.1),
    ]


@pytest.mark.asyncio
async def test_pointwise_cache_remaps_only_valid_canonical_slot() -> None:
    backend = FakeModelBackend(lambda _query, _text: 0.8)
    ranker = Reranker(backend, strategy="pointwise", cache=MemoryCache())
    first = await ranker.rerank(query="q", candidates=["other", "same"])
    second = await ranker.rerank(query="q", candidates=["same"])
    assert first.results[0].occurrence_id == "c00000000"
    assert second.results[0].occurrence_id == "c00000000"
    assert second.stats.cache_hits == 1
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_mutable_model_identity_disables_judgment_cache() -> None:
    class MutableBackend(FakeModelBackend):
        cache_stable = False

    backend = MutableBackend()
    ranker = Reranker(backend, cache=MemoryCache())
    await ranker.rerank(query="q", candidates=["same"])
    await ranker.rerank(query="q", candidates=["same"])
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_poisoned_listwise_cache_ids_are_rejected_before_use() -> None:
    backend = FakeModelBackend(lambda _query, text: 0.9 if text == "a" else 0.1)
    cache = MemoryCache()
    ranker = Reranker(backend, strategy="listwise", cache=cache)
    await ranker.rerank(query="q", candidates=["a", "b"])
    key = next(iter(cache._items))
    poisoned = ModelResponse((CandidateScore("unknown", 0.99), CandidateScore("c00000001", 0.1)))
    await cache.set(key, CacheRecord(poisoned, 0))
    response = await ranker.rerank(query="q", candidates=["a", "b"])
    assert response.results[0].item == "a"
    assert response.stats.cache_hits == 0
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_failed_pointwise_window_cancels_sibling_calls() -> None:
    from jev_rankkit.errors import BackendError

    class FailThenSlow(FakeModelBackend):
        def __init__(self) -> None:
            super().__init__()
            self.completed: list[str] = []

        async def score(self, request):  # type: ignore[no-untyped-def]
            text = request.candidates[0].text
            if text == "fail":
                raise BackendError("injected failure")
            await asyncio.sleep(0.05)
            self.completed.append(text)
            return await super().score(request)

    backend = FailThenSlow()
    ranker = Reranker(backend, strategy="pointwise")
    with pytest.raises(BackendError):
        await ranker.rerank(query="q", candidates=["fail", "slow"])
    await asyncio.sleep(0.07)
    assert backend.completed == []
    await ranker.aclose()


@pytest.mark.asyncio
async def test_inflight_duplicate_requests_share_work() -> None:
    backend = FakeModelBackend(delay_s=0.02)
    ranker = Reranker(backend, strategy="pointwise")
    first, second = await asyncio.gather(
        ranker.rerank(query="q", candidates=["same"]),
        ranker.rerank(query="q", candidates=["same"]),
    )
    assert first.results[0].score == second.results[0].score
    # Deduplication is operation-local by design; independent rerank calls do not share futures.
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_malformed_backend_score_is_rejected() -> None:
    backend = FakeModelBackend({"c00000000": 2.0}, malformed=True)
    with pytest.raises(OutputValidationError):
        await Reranker(backend, strategy="pointwise").rerank(query="q", candidates=["a"])


def test_candidate_score_schema_is_explicit() -> None:
    score = CandidateScore(candidate_id="c1", score=0.5, reasoning=None)
    assert score.candidate_id == "c1"
