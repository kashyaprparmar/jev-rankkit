from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from jev_rankkit import RerankContext, Reranker
from jev_rankkit.backends import FakeModelBackend
from jev_rankkit.backends.base import ModelBackend
from jev_rankkit.candidates import prepare_candidates
from jev_rankkit.config import Budget, RerankerConfig
from jev_rankkit.errors import ConfigurationError, ConstraintError, ProjectionError
from jev_rankkit.observability import TraceEvent, TraceKind
from jev_rankkit.types import ResultStatus


@dataclass
class Product:
    sku: str
    name: str
    description: str


@pytest.mark.asyncio
async def test_arbitrary_objects_preserve_identity_and_stable_ties() -> None:
    products = [
        Product("a", "Alpha", "vector search"),
        Product("b", "Beta", "vector search"),
        Product("c", "Gamma", "relational"),
    ]
    backend = FakeModelBackend({"c00000000": 0.8, "c00000001": 0.8, "c00000002": 0.2})
    response = await Reranker(backend, strategy="pointwise").rerank(
        query="vector database",
        candidates=products,
        text_fn=lambda item: f"{item.name}\n{item.description}",
        id_fn=lambda item: item.sku,
        top_k=2,
    )

    assert [result.item for result in response.results] == products[:2]
    assert response.results[0].item is products[0]
    assert [result.rank for result in response.results] == [1, 2]
    assert [result.candidate_id for result in response.results] == ["a", "b"]


@pytest.mark.asyncio
async def test_duplicate_occurrences_are_not_deduplicated() -> None:
    item = Product("same", "Alpha", "vector search")
    response = await Reranker(FakeModelBackend(), strategy="pointwise").rerank(
        query="vector", candidates=[item, item], text_fn=lambda value: value.description
    )
    assert len(response.results) == 2
    assert response.results[0].item is response.results[1].item
    assert response.results[0].occurrence_id != response.results[1].occurrence_id


@pytest.mark.asyncio
async def test_zero_top_k_does_not_project_candidates() -> None:
    response = await Reranker().rerank(query="q", candidates=[object()], top_k=0)
    assert response.status is ResultStatus.NOOP
    assert response.results == ()


@pytest.mark.asyncio
async def test_empty_candidates_are_noop() -> None:
    response = await Reranker().rerank(query="q", candidates=[])
    assert response.status is ResultStatus.NOOP
    assert response.stats.model_calls == 0


def test_non_string_requires_projection() -> None:
    with pytest.raises(ProjectionError):
        prepare_candidates([object()], config=RerankerConfig())


def test_context_requires_aware_time() -> None:
    with pytest.raises(ConfigurationError):
        RerankContext(as_of=datetime(2026, 1, 1))
    assert RerankContext(as_of=datetime(2026, 1, 1, tzinfo=UTC)).as_of.tzinfo is UTC


def test_budget_accepts_float_example_without_binary_decimal_drift() -> None:
    assert Budget(max_cost_usd=0.01).max_cost_usd == Decimal("0.01")
    with pytest.raises(ConfigurationError):
        Budget(max_cost_usd=float("nan"))
    with pytest.raises(ConfigurationError):
        Budget(max_tokens=True)


@pytest.mark.asyncio
async def test_sync_api_rejects_active_loop() -> None:
    with pytest.raises(RuntimeError, match="active event loop"):
        Reranker().rerank_sync(query="q", candidates=["q"])


def test_sync_api() -> None:
    response = Reranker().rerank_sync(query="vector", candidates=["other", "vector"])
    assert response.results[0].item == "vector"


def test_owned_sync_model_can_be_reused_across_event_loops() -> None:
    class TestReranker(Reranker):
        def _resolve_backend(self, model: str | ModelBackend | None) -> ModelBackend | None:
            return FakeModelBackend() if model == "test:fake" else super()._resolve_backend(model)

    ranker = TestReranker(model="test:fake")
    first = ranker.rerank_sync(query="q", candidates=["q"])
    second = ranker.rerank_sync(query="q", candidates=["q"])
    assert first.results[0].score == second.results[0].score == 0.5


@pytest.mark.asyncio
async def test_final_eligibility_change_fails_closed() -> None:
    calls = 0

    def eligible(_item: str) -> bool:
        nonlocal calls
        calls += 1
        return calls == 1

    with pytest.raises(ConstraintError):
        await Reranker().rerank(query="q", candidates=["q"], eligible_fn=eligible)


@pytest.mark.asyncio
async def test_invalid_top_k_type_is_rejected_before_projection() -> None:
    with pytest.raises(ConfigurationError):
        await Reranker().rerank(query="q", candidates=[object()], top_k=1.2)  # type: ignore[arg-type]


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    async def on_event(self, event: TraceEvent) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_observer_receives_payload_safe_events() -> None:
    observer = RecordingObserver()
    config = RerankerConfig(observer=observer)
    async with Reranker(FakeModelBackend(), config=config) as ranker:
        response = await ranker.rerank(query="secret query", candidates=["secret candidate"])
    kinds = [event.kind for event in observer.events]
    assert kinds == [
        TraceKind.MODEL_START,
        TraceKind.MODEL_END,
        TraceKind.STAGE_END,
        TraceKind.RERANK_END,
    ]
    assert response.stats.observer_events_dropped == 0
    assert all("secret" not in repr(event) for event in observer.events)


@pytest.mark.asyncio
async def test_slow_observer_drops_events_without_blocking_ranking() -> None:
    class SlowObserver:
        async def on_event(self, event: TraceEvent) -> None:
            del event
            await asyncio.sleep(0.05)

    config = RerankerConfig(observer=SlowObserver(), observer_queue_size=1)
    async with Reranker(FakeModelBackend(), config=config) as ranker:
        response = await ranker.rerank(query="query", candidates=["candidate"])
    assert len(response.results) == 1
    assert response.stats.observer_events_dropped >= 1
