"""Offline mechanics demo: lexical + toy vectors + fake model relevance.

Toy vectors and fake model scores are not quality baselines; use real backends in production.
"""

import asyncio

from jev_rankkit import Reranker
from jev_rankkit.backends import FakeModelBackend


class ToyEmbeddingBackend:
    backend_id = "toy"
    cache_identity = "toy-keywords-v1"

    async def embed(self, texts: list[str] | tuple[str, ...]) -> list[list[float]]:
        vocabulary = ("vector", "database", "search")
        return [[float(word in text.casefold()) for word in vocabulary] for text in texts]

    async def aclose(self) -> None:
        return None


async def main() -> None:
    reranker = Reranker(
        FakeModelBackend(lambda _query, text: 0.9 if "vector" in text else 0.2),
        embedding_backend=ToyEmbeddingBackend(),
    )
    response = await reranker.rerank(
        query="vector database search",
        candidates=["vector database", "full text search", "database administration"],
        metrics=["lexical", "semantic", "llm_relevance"],
        weights={"lexical": 0.3, "semantic": 0.3, "llm_relevance": 0.4},
        top_k=2,
    )
    print([(result.item, result.score) for result in response.results])


if __name__ == "__main__":
    asyncio.run(main())
