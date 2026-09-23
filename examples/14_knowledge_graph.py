"""Rank graph nodes, edges, paths, and subgraphs as ordinary objects."""

import asyncio

from jev_rankkit import Reranker


async def main() -> None:
    graph_candidates = [
        {"kind": "node", "summary": "Customer Alice"},
        {"kind": "edge", "summary": "Alice placed order 42"},
        {"kind": "path", "summary": "Alice -> order 42 -> paid invoice"},
        {"kind": "subgraph", "summary": "Warehouse inventory and suppliers"},
    ]
    response = await Reranker().rerank(
        query="Alice paid order",
        candidates=graph_candidates,
        text_fn=lambda item: f"{item['kind']}: {item['summary']}",
        top_k=2,
    )
    print([result.item for result in response.results])


if __name__ == "__main__":
    asyncio.run(main())
