"""Evaluate a fixed pool with graded relevance labels."""

import asyncio

from jev_rankkit import Reranker
from jev_rankkit.evaluation import MRR, NDCG, EvaluationCase, EvaluationDataset, Recall, evaluate


async def main() -> None:
    dataset = EvaluationDataset(
        "tiny-demo",
        "1",
        (
            EvaluationCase(
                "q1",
                "vector search",
                ("Cooking tips", "Vector search overview", "Database setup"),
                (0.0, 3.0, 1.0),
            ),
        ),
    )
    report = await evaluate(Reranker(), dataset, metrics=[NDCG(k=3), Recall(k=2), MRR()])
    print(dict(report.metrics))
    print(report.performance)


if __name__ == "__main__":
    asyncio.run(main())
