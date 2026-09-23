"""Repeatable local-only projection and ranking overhead probe.

Run with ``uv run python benchmarks/python_overhead.py``. Results depend on the host and are
diagnostic, not a benchmark claim about model quality or end-to-end service latency.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass

from jev_rankkit import Reranker
from jev_rankkit.candidates import prepare_candidates
from jev_rankkit.config import CacheConfig, RerankerConfig


@dataclass(frozen=True)
class Item:
    text: str


async def measure(size: int, repeats: int = 12) -> tuple[float, float]:
    candidates = [Item(f"vector search database option {index}") for index in range(size)]
    config = RerankerConfig(cache=CacheConfig(enabled=False))
    ranker = Reranker(config=config)
    projection_ms: list[float] = []
    total_ms: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        prepare_candidates(candidates, config=config, text_fn=lambda item: item.text)
        projection_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        await ranker.rerank(
            query="vector database",
            candidates=candidates,
            text_fn=lambda item: item.text,
            top_k=10,
        )
        total_ms.append((time.perf_counter() - started) * 1000)
    return statistics.median(projection_ms), statistics.median(total_ms)


async def main() -> None:
    for size in (100, 1000, 5000):
        projection_ms, total_ms = await measure(size)
        print(
            f"candidates={size} projection_median_ms={projection_ms:.3f} "
            f"lexical_rerank_median_ms={total_ms:.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
