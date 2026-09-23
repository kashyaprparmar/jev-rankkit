"""Any Python object can be ranked without inheritance or conversion."""

import asyncio
from dataclasses import dataclass

from jev_rankkit import Reranker


@dataclass
class Product:
    sku: str
    name: str
    description: str


async def main() -> None:
    products = [
        Product("a", "SearchDB", "Vector search database"),
        Product("b", "LedgerDB", "Relational transaction database"),
    ]
    response = await Reranker().rerank(
        query="vector search",
        candidates=products,
        text_fn=lambda item: f"{item.name}\n{item.description}",
        id_fn=lambda item: item.sku,
        top_k=1,
    )
    assert response.results[0].item is products[0]
    print(response.results[0].item)


if __name__ == "__main__":
    asyncio.run(main())
