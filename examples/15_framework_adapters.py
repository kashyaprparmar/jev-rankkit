"""Duck-typed retrieval adapters; install no framework to run this demonstration."""

import asyncio

from jev_rankkit import Reranker
from jev_rankkit.integrations import elasticsearch_adapter, qdrant_adapter


async def main() -> None:
    qdrant_points = [{"payload": {"text": "vector search guide"}}]
    elastic_hits = [{"_source": {"body": "vector database guide"}}]
    qdrant = await Reranker().rerank(
        query="vector", candidates=qdrant_points, adapter=qdrant_adapter()
    )
    elastic = await Reranker().rerank(
        query="vector",
        candidates=elastic_hits,
        adapter=elasticsearch_adapter(text_key="body"),
    )
    print(qdrant.results[0].item, elastic.results[0].item)


if __name__ == "__main__":
    asyncio.run(main())
