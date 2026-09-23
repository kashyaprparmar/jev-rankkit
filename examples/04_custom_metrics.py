"""Combine lexical relevance with an application-defined business score."""

import asyncio

from jev_rankkit import Reranker
from jev_rankkit.metrics import CallableMetric, LexicalRelevance


async def main() -> None:
    products = [
        {"name": "Vector DB", "priority": 0.9},
        {"name": "Relational DB", "priority": 0.3},
    ]
    priority = CallableMetric("priority", lambda _query, item, _context: item["priority"])
    response = await Reranker().rerank(
        query="vector database",
        candidates=products,
        text_fn=lambda item: item["name"],
        metrics=[LexicalRelevance(), priority],
        weights={"lexical": 0.7, "priority": 0.3},
    )
    for result in response.results:
        print(result.item, result.score, result.metrics)


if __name__ == "__main__":
    asyncio.run(main())
