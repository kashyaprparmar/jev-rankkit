# Evaluation contract

`EvaluationCase` aligns a fixed candidate tuple with a relevance grade for each input position. This preserves labels for duplicate objects and texts. Grades are finite, non-negative numbers; values greater than zero count as relevant for binary metrics. Missing judgments fail by default. Set `unjudged_policy="zero"` only when that treatment is intentional and documented.

```python
from jev_rankkit.evaluation import EvaluationCase, EvaluationDataset, NDCG, Recall, MRR, evaluate

dataset = EvaluationDataset(
    "demo", "1", (EvaluationCase("q1", "vector search", ("other", "vector search guide"), (0, 3)),)
)
report = await evaluate(reranker, dataset, metrics=[NDCG(k=2), Recall(k=2), MRR()])
print(report.metrics, report.performance)
```

Available metrics are Precision@K, Recall@K, HitRate@K, Success@K, MRR, MAP, and nDCG@K. When a query has no relevant candidate, binary retrieval metrics return zero. Precision divides by the requested `k`; Recall divides by all relevant candidates in the fixed pool. MAP@K divides by `min(number of relevant candidates, k)`. nDCG uses graded gain `2^grade - 1`, implemented with a numerically stable common scale. Custom sync or async metrics implement `EvaluationMetric` or use `CallableEvaluationMetric`.

`evaluate(..., rerank_top_k=None)` requests a full order. If ranking is intentionally truncated, set `rerank_top_k` and use cutoffs no larger than that value. Unbounded MRR/MAP require full output. Partial outputs fail by default. The report includes aggregate quality, per-case order and statistics, elapsed time, throughput, model calls, tokens, estimated cost, and cache hit rate. Unreported provider usage/cost stays unknown; zero is reserved for measured no-model runs.

Evaluate downstream RAG answer quality, evidence support, shortlist recall, domain slices, positional sensitivity, and adversarial candidates separately. This API scores a fixed pool; it cannot measure retrieval recall for documents that were never candidates. Use held-out, independently judged data for actual quality claims.
