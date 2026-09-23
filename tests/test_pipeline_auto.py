from __future__ import annotations

from decimal import Decimal

import pytest

from jev_rankkit import AutoReranker, RerankContext, Reranker
from jev_rankkit.backends import FakeModelBackend, ModelRequest, ModelResponse
from jev_rankkit.candidates import prepare_candidates
from jev_rankkit.config import Budget, FallbackPolicy, RerankerConfig
from jev_rankkit.errors import BudgetExceededError, BudgetUnverifiableError, OutputValidationError
from jev_rankkit.pipeline import (
    BM25Filter,
    CandidateFilter,
    DiversityReranker,
    JevReranker,
    RerankPipeline,
)
from jev_rankkit.selection import mmr_select, reciprocal_rank_fusion
from jev_rankkit.strategies import LexicalStrategy
from jev_rankkit.types import (
    CostConfidence,
    RankingEntry,
    RankingOutcome,
    RequestStatistics,
    ScoreKind,
    Usage,
)


def test_rrf_is_deterministic_and_validates_duplicate_ids() -> None:
    lists = [["a", "b", "c"], ["b", "a", "d"]]
    first = reciprocal_rank_fusion(lists, constant=60)
    second = reciprocal_rank_fusion(lists, constant=60)
    assert first.entries == second.entries
    assert first.entries[0].occurrence_id == "a"
    assert first.entries[1].occurrence_id == "b"
    assert first.score_kind is ScoreKind.FUSION
    with pytest.raises(Exception, match="unique IDs"):
        reciprocal_rank_fusion([["a", "a"]])


@pytest.mark.asyncio
async def test_mmr_preserves_base_scores_and_reports_selection_scores() -> None:
    candidates = prepare_candidates(
        ["red apple", "red apple", "green pear"], config=RerankerConfig()
    )
    outcome = RankingOutcome(
        tuple(
            RankingEntry(candidate.occurrence_id, score)
            for candidate, score in zip(candidates, [0.9, 0.8, 0.7], strict=True)
        )
    )
    selected = await mmr_select(outcome, candidates, top_k=2, lambda_=0.5)
    assert [entry.occurrence_id for entry in selected.entries] == ["c00000000", "c00000002"]
    assert selected.entries[0].score == 0.9
    assert selected.entries[1].selection_score is not None


@pytest.mark.asyncio
async def test_pipeline_passes_only_survivors_and_records_stages() -> None:
    backend = FakeModelBackend(lambda _query, text: 1.0 if text.endswith("1") else 0.2)
    pipeline = RerankPipeline(
        [
            CandidateFilter(limit=3),
            JevReranker(limit=2, mode="listwise"),
            DiversityReranker(limit=1),
        ]
    )
    response = await Reranker(backend, strategy=pipeline).rerank(
        query="item", candidates=[f"item {index}" for index in range(5)], top_k=1
    )
    assert [stage.output_count for stage in response.execution_plan.stages] == [3, 2, 1]
    assert len(backend.calls) == 1
    assert len(backend.calls[0].candidates) == 3
    assert response.results[0].selection_score is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("count", "expected"),
    [(5, "listwise"), (50, "pointwise"), (120, "auto")],
)
async def test_auto_routes_and_explains(count: int, expected: str) -> None:
    backend = FakeModelBackend(lambda _query, _text: 0.7)
    ranker = AutoReranker(backend)
    candidates = [f"candidate {index}" for index in range(count)]
    plan = await ranker.plan(query="candidate", candidates=candidates, top_k=5)
    response = await ranker.rerank(query="candidate", candidates=candidates, top_k=5)
    assert plan.strategy == expected
    assert response.execution_plan.strategy == expected
    assert response.execution_plan.rationale
    assert len(response.results) == 5


@pytest.mark.asyncio
async def test_auto_offline_and_zero_call_budget_never_call_backend() -> None:
    backend = FakeModelBackend()
    ranker = AutoReranker(backend)
    offline = await ranker.rerank(
        query="vector",
        candidates=["vector", "other"],
        context=RerankContext(quality_mode="offline"),
    )
    budgeted = await ranker.rerank(
        query="vector",
        candidates=["vector", "other"],
        context=RerankContext(budget=Budget(max_model_calls=0)),
    )
    assert offline.execution_plan.strategy == "lexical"
    assert budgeted.execution_plan.strategy == "lexical"
    assert not backend.calls


@pytest.mark.asyncio
async def test_auto_full_order_never_selects_pruning_cascade() -> None:
    backend = FakeModelBackend()
    response = await AutoReranker(backend).rerank(
        query="vector", candidates=[f"item {index}" for index in range(120)]
    )
    assert response.execution_plan.strategy == "lexical"
    assert len(response.results) == 120
    assert not backend.calls


@pytest.mark.asyncio
async def test_model_budget_and_explicit_lexical_fallback() -> None:
    backend = FakeModelBackend(fail=OutputValidationError("bad"))
    config = RerankerConfig(fallback_chain=(FallbackPolicy.LEXICAL, FallbackPolicy.STRICT))
    response = await Reranker(backend, config=config).rerank(
        query="vector", candidates=["other", "vector"]
    )
    assert response.status == "fallback"
    assert response.results[0].item == "vector"
    assert response.stats.fallbacks == 1
    with pytest.raises(BudgetExceededError):
        await Reranker(FakeModelBackend()).rerank(
            query="vector",
            candidates=["vector"],
            context=RerankContext(budget=Budget(max_model_calls=0)),
        )


@pytest.mark.asyncio
async def test_previous_stage_fallback_uses_complete_checkpoint() -> None:
    backend = FakeModelBackend(fail=OutputValidationError("bad"))
    pipeline = RerankPipeline([BM25Filter(limit=2), JevReranker(limit=1)])
    config = RerankerConfig(fallback_chain=(FallbackPolicy.PREVIOUS_STAGE, FallbackPolicy.STRICT))
    response = await Reranker(backend, strategy=pipeline, config=config).rerank(
        query="vector", candidates=["unrelated", "vector", "vector search"], top_k=1
    )
    assert response.status == "fallback"
    assert response.results[0].item in {"vector", "vector search"}
    assert response.execution_plan.stages[0].name == "BM25Filter"


@pytest.mark.asyncio
async def test_partial_checkpoint_fallback_is_labeled_partial() -> None:
    backend = FakeModelBackend(fail=OutputValidationError("bad"))
    pipeline = RerankPipeline([BM25Filter(limit=2), JevReranker(limit=1)])
    config = RerankerConfig(fallback_chain=(FallbackPolicy.PARTIAL, FallbackPolicy.STRICT))
    response = await Reranker(backend, strategy=pipeline, config=config).rerank(
        query="vector", candidates=["vector", "other"], top_k=1
    )
    assert response.status == "partial"


@pytest.mark.asyncio
async def test_full_order_pipeline_disallows_pruning() -> None:
    with pytest.raises(Exception, match="full ordering"):
        await Reranker(FakeModelBackend(), strategy=RerankPipeline([BM25Filter(limit=1)])).rerank(
            query="vector", candidates=["vector", "other"], top_k=None
        )


class UnmeteredPricedBackend(FakeModelBackend):
    input_price_per_million_usd = Decimal("1")

    async def score(self, request: ModelRequest) -> ModelResponse:
        response = await super().score(request)
        return ModelResponse(
            response.scores,
            RequestStatistics(usage=Usage(tokens_reported=False)),
            response.resolved_model,
        )


@pytest.mark.asyncio
async def test_missing_provider_usage_stays_unknown() -> None:
    response = await Reranker(UnmeteredPricedBackend()).rerank(
        query="vector", candidates=["vector"]
    )
    assert response.stats.usage_complete is False
    assert response.stats.estimated_cost_usd is None
    assert response.stats.cost_confidence is CostConfidence.UNKNOWN


@pytest.mark.asyncio
async def test_reported_provider_cost_keeps_its_confidence() -> None:
    class CostReportingBackend(FakeModelBackend):
        async def score(self, request: ModelRequest) -> ModelResponse:
            response = await super().score(request)
            return ModelResponse(
                response.scores,
                RequestStatistics(
                    usage=Usage(
                        input_tokens=10,
                        output_tokens=2,
                        cost_usd=0.001,
                        cost_confidence=CostConfidence.KNOWN,
                        tokens_reported=True,
                    )
                ),
                response.resolved_model,
            )

    response = await Reranker(CostReportingBackend()).rerank(
        query="vector", candidates=["vector", "search"]
    )
    assert response.stats.estimated_cost_usd == pytest.approx(0.002)
    assert response.stats.cost_confidence is CostConfidence.KNOWN


@pytest.mark.asyncio
async def test_strict_cost_rejects_unverifiable_backend_and_auto_routes_lexical() -> None:
    budget = Budget(max_cost_usd=Decimal("0.01"), strict_cost=True)
    with pytest.raises(BudgetUnverifiableError):
        await Reranker(UnmeteredPricedBackend()).rerank(
            query="vector", candidates=["vector"], context=RerankContext(budget=budget)
        )
    backend = UnmeteredPricedBackend()
    response = await AutoReranker(backend).rerank(
        query="vector", candidates=["vector"], context=RerankContext(budget=budget)
    )
    assert response.execution_plan.strategy == "lexical"
    assert backend.calls == []


@pytest.mark.asyncio
async def test_failed_model_reservation_releases_unused_attempts_for_fallback() -> None:
    from jev_rankkit.config import RetryConfig
    from jev_rankkit.errors import BackendError, ErrorDetails

    class FirstModeFails(FakeModelBackend):
        retry = RetryConfig(max_attempts=2)

        async def score(self, request):  # type: ignore[no-untyped-def]
            if request.mode == "listwise":
                raise BackendError("injected", details=ErrorDetails(safe_context={"attempts": 1}))
            return await super().score(request)

    ranker = Reranker(
        FirstModeFails(),
        strategy="listwise",
        config=RerankerConfig(
            fallback_chain=(FallbackPolicy.POINTWISE,), budget=Budget(max_model_calls=3)
        ),
    )
    response = await ranker.rerank(query="q", candidates=["a"])
    assert response.status == "fallback"
    assert response.stats.model_calls == 2


@pytest.mark.asyncio
async def test_threshold_rejects_rank_only_and_fusion_scales() -> None:
    from jev_rankkit.errors import ConfigurationError

    class FusionStrategy:
        name = "fusion"

        async def rank(self, *, query, candidates, context, services, top_k):  # type: ignore[no-untyped-def]
            del query, context, services, top_k
            return RankingOutcome(
                (RankingEntry(candidates[0].occurrence_id, 0.02),), ScoreKind.FUSION
            )

    config = RerankerConfig(score_threshold=0.5)
    with pytest.raises(ConfigurationError, match="requires utility"):
        await Reranker(strategy=FusionStrategy(), config=config).rerank(query="q", candidates=["a"])
    with pytest.raises(ConfigurationError, match="requires utility"):
        await Reranker(
            FakeModelBackend(fail=OutputValidationError("failed")),
            config=RerankerConfig(
                score_threshold=0.5, fallback_chain=(FallbackPolicy.INPUT_ORDER,)
            ),
        ).rerank(query="q", candidates=["a"])


@pytest.mark.asyncio
async def test_strategy_object_cannot_silently_ignore_metrics() -> None:
    from jev_rankkit.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="strategy object"):
        await Reranker().rerank(
            query="q", candidates=["q"], metrics=["lexical"], strategy=LexicalStrategy()
        )


@pytest.mark.asyncio
async def test_plan_includes_per_call_strategy_and_metrics() -> None:
    ranker = Reranker(FakeModelBackend(), strategy="auto")
    plan = await ranker.plan(
        query="q", candidates=["q"], metrics=["lexical"], strategy="pointwise", top_k=1
    )
    assert plan.strategy == "weighted_metrics"
    assert plan.stages[0].output_count == 1
    assert (await ranker.plan(query="q", candidates=["q"], top_k=0)).strategy == "noop"


@pytest.mark.asyncio
async def test_auto_respects_configured_model_call_budget_for_chunks() -> None:
    backend = FakeModelBackend()
    ranker = AutoReranker(backend, config=RerankerConfig(budget=Budget(max_model_calls=1)))
    response = await ranker.rerank(
        query="vector",
        candidates=[f"candidate {index}" for index in range(50)],
        context=RerankContext(quality_mode="quality"),
        top_k=5,
    )
    assert response.execution_plan.strategy == "lexical"
    assert backend.calls == []
