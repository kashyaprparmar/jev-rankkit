"""Coarse BM25 shortlist -> stronger fake model judgment."""

import asyncio

from jev_rankkit import Reranker
from jev_rankkit.backends import FakeModelBackend
from jev_rankkit.pipeline import BM25Filter, JevReranker, RerankPipeline


async def main() -> None:
    pipeline = RerankPipeline([BM25Filter(limit=8), JevReranker(limit=3)])
    reranker = Reranker(
        FakeModelBackend(lambda _query, text: 0.9 if "vector" in text else 0.2),
        strategy=pipeline,
    )
    candidates = [f"database guide {index}" for index in range(20)] + [
        "vector database comparison",
        "vector search database setup",
    ]
    response = await reranker.rerank(query="vector database", candidates=candidates, top_k=3)
    print(
        [
            (stage.name, stage.input_count, stage.output_count)
            for stage in response.execution_plan.stages
        ]
    )
    print([item.item for item in response.results])


if __name__ == "__main__":
    asyncio.run(main())
