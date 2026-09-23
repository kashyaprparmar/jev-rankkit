"""Retrieve many memories, then inject only relevant ones into agent context."""

import asyncio

from jev_rankkit import Reranker


async def main() -> None:
    memories = [
        {"content": "User prefers concise answers about Python."},
        {"content": "User enjoys hiking."},
        {"content": "Last Python project used async APIs."},
    ]
    response = await Reranker().rerank_memories(
        query="help with an async Python API", candidates=memories, top_k=2
    )
    agent_context = [result.item for result in response.results]
    print(agent_context)


if __name__ == "__main__":
    asyncio.run(main())
