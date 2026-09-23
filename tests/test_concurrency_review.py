from __future__ import annotations

import asyncio
import sys
import types
from decimal import Decimal

import pytest

from jev_rankkit import Reranker
from jev_rankkit._runtime.executor import BudgetLedger
from jev_rankkit.backends import FakeModelBackend
from jev_rankkit.backends.sentence_transformers import SentenceTransformerBackend
from jev_rankkit.cache import CacheRecord, MemoryCache
from jev_rankkit.config import Budget, RerankerConfig, TruncationPolicy
from jev_rankkit.context import RerankContext
from jev_rankkit.errors import BudgetExceededError, DeadlineExceededError, ProjectionError
from jev_rankkit.prompts import RerankPrompt


@pytest.mark.asyncio
async def test_concurrent_requests_share_instance_concurrency_limit() -> None:
    backend = FakeModelBackend(delay_s=0.01)
    ranker = Reranker(backend, strategy="pointwise", config=RerankerConfig(max_concurrency=2))
    first, second = await asyncio.gather(
        ranker.rerank(query="one", candidates=[str(index) for index in range(5)]),
        ranker.rerank(query="two", candidates=[str(index) for index in range(5)]),
    )
    assert len(first.results) == len(second.results) == 5
    assert backend.max_active_calls == 2
    await ranker.aclose()


@pytest.mark.asyncio
async def test_cancellation_releases_permit_for_next_request() -> None:
    backend = FakeModelBackend(delay_s=0.05)
    ranker = Reranker(backend, strategy="pointwise", config=RerankerConfig(max_concurrency=1))
    pending = asyncio.create_task(ranker.rerank(query="q", candidates=["a", "b"]))
    await asyncio.sleep(0.01)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    response = await asyncio.wait_for(ranker.rerank(query="q", candidates=["c"]), timeout=0.5)
    assert len(response.results) == 1
    assert backend.active_calls == 0
    await ranker.aclose()


@pytest.mark.asyncio
async def test_scope_exit_cancels_active_work_before_closing_owned_resources() -> None:
    backend = FakeModelBackend(delay_s=0.05)
    ranker = Reranker(backend)
    pending = asyncio.create_task(ranker.rerank(query="q", candidates=["a"]))
    await asyncio.sleep(0.01)
    await ranker.aclose()
    assert pending.cancelled()
    assert backend.active_calls == 0
    with pytest.raises(RuntimeError, match="closed"):
        await ranker.rerank(query="q", candidates=["a"])


@pytest.mark.asyncio
async def test_concurrent_cache_misses_do_not_corrupt_results() -> None:
    backend = FakeModelBackend(delay_s=0.01)
    ranker = Reranker(backend, cache=MemoryCache())
    first, second = await asyncio.gather(
        ranker.rerank(query="same", candidates=["same"]),
        ranker.rerank(query="same", candidates=["same"]),
    )
    third = await ranker.rerank(query="same", candidates=["same"])
    assert first.results[0].score == second.results[0].score == third.results[0].score
    assert third.stats.cache_hits == 1
    await ranker.aclose()


@pytest.mark.asyncio
async def test_long_candidate_policy_is_explicit() -> None:
    long_text = "a" * 100
    with pytest.raises(ProjectionError, match="exceeds"):
        await Reranker(config=RerankerConfig(max_candidate_chars=10)).rerank(
            query="a", candidates=[long_text]
        )
    response = await Reranker(
        config=RerankerConfig(max_candidate_chars=10, truncation=TruncationPolicy.HEAD)
    ).rerank(query="a", candidates=[long_text])
    assert response.results[0].item == long_text


@pytest.mark.asyncio
async def test_atomic_budget_reservations_prevent_concurrent_overspend() -> None:
    ledger = BudgetLedger(Budget(max_model_calls=1, max_tokens=10))
    outcomes = await asyncio.gather(
        ledger.reserve(estimated_tokens=5, estimated_cost=None),
        ledger.reserve(estimated_tokens=5, estimated_cost=None),
        return_exceptions=True,
    )
    assert sum(isinstance(value, BudgetExceededError) for value in outcomes) == 1
    assert ledger.calls == 1


@pytest.mark.asyncio
async def test_token_budget_rejects_before_dispatch_and_latency_cancels_call() -> None:
    backend = FakeModelBackend(delay_s=0.05)
    ranker = Reranker(backend)
    with pytest.raises(BudgetExceededError):
        await ranker.rerank(
            query="vector",
            candidates=["x" * 100],
            context=RerankContext(budget=Budget(max_tokens=1)),
        )
    assert not backend.calls
    with pytest.raises(DeadlineExceededError):
        await ranker.rerank(
            query="vector",
            candidates=["vector"],
            context=RerankContext(budget=Budget(max_latency_ms=10)),
        )
    assert backend.active_calls == 0
    await ranker.aclose()


@pytest.mark.asyncio
async def test_latency_budget_covers_embedding_route() -> None:
    class SlowEmbeddings:
        backend_id = "slow"
        cache_identity = "slow-v1"

        async def embed(self, texts):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.05)
            return [[1.0, 0.0] for _ in texts]

        async def aclose(self) -> None:
            return None

    ranker = Reranker(embedding_backend=SlowEmbeddings())
    with pytest.raises(DeadlineExceededError):
        await ranker.rerank(
            query="q",
            candidates=["a"],
            metrics=["semantic"],
            context=RerankContext(budget=Budget(max_latency_ms=5)),
        )


@pytest.mark.asyncio
async def test_strict_cost_upper_bound_rejects_before_provider_call() -> None:
    class BoundedBackend(FakeModelBackend):
        def chargeable_cost_upper_bound(
            self, *, query: str, candidates: tuple[str, ...], prompt: RerankPrompt
        ) -> Decimal:
            del query, candidates, prompt
            return Decimal("0.002")

    backend = BoundedBackend()
    with pytest.raises(BudgetExceededError):
        await Reranker(backend).rerank(
            query="q",
            candidates=["candidate"],
            context=RerankContext(budget=Budget(max_cost_usd=0.001, strict_cost=True)),
        )
    assert not backend.calls


@pytest.mark.asyncio
async def test_memory_cache_lru_ttl_and_namespace_clear() -> None:
    cache = MemoryCache(max_entries=2)
    await cache.set("tenant:a", CacheRecord(1, 0))
    await cache.set("tenant:b", CacheRecord(2, 0))
    assert (await cache.get("tenant:a")).value == 1  # type: ignore[union-attr]
    await cache.set("other:c", CacheRecord(3, 0))
    assert await cache.get("tenant:b") is None
    await cache.clear(namespace="tenant")
    assert await cache.get("tenant:a") is None
    await cache.set("short", CacheRecord(4, 0), ttl_seconds=0.001)
    await asyncio.sleep(0.05)
    assert await cache.get("short") is None


@pytest.mark.asyncio
async def test_optional_embedding_backend_loads_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    class DummySentenceTransformer:
        def __init__(self, name: str, **kwargs: object) -> None:
            del kwargs
            created.append(name)

        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            del kwargs
            return [[float(len(text))] for text in texts]

        def encode_query(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            return self.encode(texts, **kwargs)

        def encode_document(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            return self.encode(texts, **kwargs)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=DummySentenceTransformer),
    )
    backend = SentenceTransformerBackend("demo")
    assert created == []
    assert await backend.embed(["a", "bb"]) == [[1.0], [2.0]]
    assert await backend.embed_query("abc") == [3.0]
    assert await backend.embed_documents(["abcd"]) == [[4.0]]
    assert created == ["demo"]
    await backend.aclose()
