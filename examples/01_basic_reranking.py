"""Run with: uv run python examples/01_basic_reranking.py"""

import asyncio

from jev_rankkit import Reranker


async def main() -> None:
    response = await Reranker().rerank(
        query="vector search database",
        candidates=["A cooking recipe", "A vector search database", "A SQL tutorial"],
        top_k=2,
    )
    for result in response.results:
        print(result.rank, result.item, result.score)


if __name__ == "__main__":
    asyncio.run(main())
