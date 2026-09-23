"""Mention and context -> candidate entities -> selected entity."""

import asyncio

from jev_rankkit import Reranker


async def main() -> None:
    entities = [
        {"name": "Mercury", "description": "a chemical element"},
        {"name": "Mercury", "description": "the innermost planet orbiting the Sun"},
        {"name": "Mercury Records", "description": "a music label"},
    ]
    response = await Reranker().rerank_entities(
        query="Mercury is the planet closest to the Sun", candidates=entities, top_k=1
    )
    print(response.results[0].item)


if __name__ == "__main__":
    asyncio.run(main())
