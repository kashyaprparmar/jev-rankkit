from __future__ import annotations

import math

import pytest

from jev_rankkit import Reranker
from jev_rankkit.config import RerankerConfig
from jev_rankkit.errors import EvaluationError
from jev_rankkit.evaluation import (
    MAP,
    MRR,
    NDCG,
    CallableEvaluationMetric,
    EvaluationCase,
    EvaluationDataset,
    HitRate,
    Precision,
    Recall,
    Success,
    evaluate,
)


def test_quality_metrics_have_known_values_and_zero_relevance_rules() -> None:
    ranked = (1, 0, 2)
    labels = (1.0, 0.0, 1.0)
    assert Precision(2).score(ranked, labels) == 0.5
    assert Recall(2).score(ranked, labels) == 0.5
    assert HitRate(1).score(ranked, labels) == 0.0
    assert Success(2).score(ranked, labels) == 1.0
    assert MRR().score(ranked, labels) == 0.5
    assert MAP().score(ranked, labels) == pytest.approx((0.5 + 2 / 3) / 2)
    assert NDCG(3).score(ranked, labels) == pytest.approx(
        (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    )
    for metric in (Recall(2), HitRate(2), MRR(), MAP(), NDCG(2)):
        assert metric.score(ranked, (0.0, 0.0, 0.0)) == 0.0


@pytest.mark.asyncio
async def test_evaluate_uses_original_positions_and_custom_async_metric() -> None:
    item = "vector"
    dataset = EvaluationDataset(
        "duplicates",
        "1",
        (EvaluationCase("q1", "vector", (item, "other", item), (1.0, 0.0, 1.0)),),
    )

    async def custom(indices: tuple[int, ...], labels: tuple[float, ...]) -> float:
        return float(indices[0] == 0 and labels[0] == 1)

    report = await evaluate(
        Reranker(),
        dataset,
        metrics=[NDCG(k=3), Recall(k=2), CallableEvaluationMetric("custom", custom)],
    )
    assert report.metrics["custom"] == 1
    assert report.cases[0].ranked_indices == (0, 2, 1)
    assert report.performance.model_calls == 0
    assert report.performance.estimated_cost_usd == 0


@pytest.mark.asyncio
async def test_unjudged_labels_require_explicit_policy() -> None:
    dataset = EvaluationDataset(
        "unjudged", "1", (EvaluationCase("q", "vector", ("vector", "other"), (1, None)),)
    )
    with pytest.raises(EvaluationError, match="unjudged"):
        await evaluate(Reranker(), dataset, metrics=[Recall(k=2)])
    report = await evaluate(Reranker(), dataset, metrics=[Recall(k=2)], unjudged_policy="zero")
    assert report.notes


@pytest.mark.asyncio
async def test_evaluation_rejects_cutoff_beyond_returned_results() -> None:
    dataset = EvaluationDataset(
        "small", "1", (EvaluationCase("q", "vector", ("vector", "other"), (1, 0)),)
    )
    with pytest.raises(EvaluationError, match="cutoff"):
        await evaluate(Reranker(), dataset, metrics=[NDCG(k=2)], rerank_top_k=1)
    with pytest.raises(EvaluationError, match="cutoff"):
        await evaluate(Reranker(), dataset, metrics=[MRR()], rerank_top_k=1)


@pytest.mark.asyncio
async def test_evaluation_rejects_underfilled_top_k_by_default() -> None:
    dataset = EvaluationDataset(
        "thresholded", "1", (EvaluationCase("q", "vector", ("vector", "other"), (1, 0)),)
    )
    ranker = Reranker(config=RerankerConfig(score_threshold=0.9))
    with pytest.raises(EvaluationError, match="complete requested shortlist"):
        await evaluate(ranker, dataset, metrics=[NDCG(k=2)], rerank_top_k=2)


def test_dataset_rejects_duplicate_case_ids_and_bad_labels() -> None:
    case = EvaluationCase("a", "query", ("a",), (1.0,))
    with pytest.raises(EvaluationError, match="unique"):
        EvaluationDataset("name", "1", (case, case))
    with pytest.raises(EvaluationError, match="align"):
        EvaluationCase("a", "query", ("a",), ())
    with pytest.raises(EvaluationError, match="grades"):
        EvaluationCase("a", "query", ("a",), (float("nan"),))
