"""Fixed-pool ranking comparison. No provider calls unless --include-jev is passed.

Run: uv run python benchmarks/compare.py
Optional: uv run --extra embeddings python benchmarks/compare.py --embedding-model <model-id>
Optional Jev: TYPESAFE_API_KEY=... uv run --extra jev python benchmarks/compare.py --include-jev
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

from jev_rankkit import Reranker
from jev_rankkit.backends import SentenceTransformerBackend
from jev_rankkit.candidates import CandidateView
from jev_rankkit.context import RerankContext
from jev_rankkit.evaluation import MRR, NDCG, EvaluationCase, EvaluationDataset, Recall, evaluate
from jev_rankkit.metrics import BM25Metric, EmbeddingSimilarity, LexicalRelevance, WeightedMetrics
from jev_rankkit.pipeline import BM25Filter, EmbeddingReranker, RerankPipeline
from jev_rankkit.strategies import MetricStrategy
from jev_rankkit.types import ExecutionStage, RankingEntry, RankingOutcome, ScoreKind

DATA = Path(__file__).parent / "data" / "synthetic_v1.json"
METRICS = (NDCG(k=5), Recall(k=5), MRR(k=5))
ORDER_SEED = 20260923


class BaselineOrder:
    name = "baseline_order"

    async def rank(
        self,
        *,
        query: str,
        candidates: tuple[CandidateView[str], ...],
        context: RerankContext,
        services: object,
        top_k: int | None,
    ) -> RankingOutcome:
        del query, context, services
        selected = candidates if top_k is None else candidates[:top_k]
        return RankingOutcome(
            tuple(RankingEntry(candidate.occurrence_id, None) for candidate in selected),
            score_kind=ScoreKind.RANK_ONLY,
            stages=(ExecutionStage(self.name, self.name, len(candidates), len(selected)),),
        )


def load_dataset() -> tuple[EvaluationDataset[str], str]:
    raw = DATA.read_bytes()
    data = json.loads(raw)
    rng = random.Random(ORDER_SEED)
    cases = []
    for entry in data["cases"]:
        candidates = list(entry["candidates"])
        rng.shuffle(candidates)
        cases.append(
            EvaluationCase(
                case_id=entry["case_id"],
                query=entry["query"],
                candidates=tuple(candidate["text"] for candidate in candidates),
                relevance=tuple(float(candidate["relevance"]) for candidate in candidates),
            )
        )
    return (
        EvaluationDataset(data["name"], data["version"], tuple(cases)),
        hashlib.sha256(raw).hexdigest(),
    )


async def measured(name: str, ranker: Reranker, dataset: EvaluationDataset[str]) -> dict[str, Any]:
    try:
        report = await evaluate(ranker, dataset, metrics=METRICS, rerank_top_k=5)
        performance = report.performance
        return {
            "strategy": name,
            "status": "measured",
            "quality": dict(report.metrics),
            "latency_ms": performance.elapsed_ms,
            "throughput_cases_per_s": performance.throughput_cases_per_s,
            "model_calls": performance.model_calls,
            "input_tokens": performance.input_tokens if performance.usage_complete else None,
            "output_tokens": performance.output_tokens if performance.usage_complete else None,
            "estimated_cost_usd": performance.estimated_cost_usd,
            "cache_hit_rate": performance.cache_hit_rate,
        }
    except Exception as exc:
        return {"strategy": name, "status": "failed", "error_type": type(exc).__name__}
    finally:
        await ranker.aclose()


def unmeasured(name: str, reason: str) -> dict[str, Any]:
    return {"strategy": name, "status": "unmeasured", "reason": reason}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", help="Installed sentence-transformers model ID")
    parser.add_argument("--embedding-revision", help="Pinned model revision, if available")
    parser.add_argument("--include-jev", action="store_true", help="Permit billable Jev calls")
    parser.add_argument("--jev-model", default="jev-1.13.0")
    parser.add_argument("--output", type=Path, help="Write JSON report to this file")
    args = parser.parse_args()
    dataset, digest = load_dataset()
    rows: list[dict[str, Any]] = []
    rows.append(await measured("baseline_order", Reranker(strategy=BaselineOrder()), dataset))
    rows.append(
        await measured(
            "bm25",
            Reranker(strategy=MetricStrategy(WeightedMetrics.equal((BM25Metric(),)))),
            dataset,
        )
    )

    embedding = (
        SentenceTransformerBackend(args.embedding_model, revision=args.embedding_revision)
        if args.embedding_model
        else None
    )
    if embedding is None:
        for name in ("embedding_similarity", "hybrid_lexical_embedding", "hierarchical"):
            rows.append(unmeasured(name, "pass --embedding-model and install the embeddings extra"))
    else:
        rows.append(
            await measured(
                "embedding_similarity",
                Reranker(
                    strategy=MetricStrategy(
                        WeightedMetrics.equal((EmbeddingSimilarity(embedding),))
                    ),
                    embedding_backend=embedding,
                ),
                dataset,
            )
        )
        rows.append(
            await measured(
                "hybrid_lexical_embedding",
                Reranker(
                    strategy=MetricStrategy(
                        WeightedMetrics(
                            (LexicalRelevance(), EmbeddingSimilarity(embedding)),
                            {"lexical": 0.4, "semantic": 0.6},
                        )
                    ),
                    embedding_backend=embedding,
                ),
                dataset,
            )
        )
        rows.append(
            await measured(
                "hierarchical",
                Reranker(
                    strategy=RerankPipeline(
                        [
                            BM25Filter(limit=6),
                            EmbeddingReranker(EmbeddingSimilarity(embedding), limit=5),
                        ]
                    ),
                    embedding_backend=embedding,
                ),
                dataset,
            )
        )
        await embedding.aclose()

    if args.include_jev and os.getenv("TYPESAFE_API_KEY"):
        for mode in ("pointwise", "listwise"):
            rows.append(
                await measured(
                    f"jev_{mode}",
                    Reranker(model=f"typesafe:{args.jev_model}", strategy=mode),
                    dataset,
                )
            )
    else:
        reason = "pass --include-jev with TYPESAFE_API_KEY to permit billable API calls"
        rows.extend((unmeasured("jev_pointwise", reason), unmeasured("jev_listwise", reason)))

    report = {
        "dataset": dataset.name,
        "dataset_version": dataset.version,
        "dataset_sha256": digest,
        "candidate_order_seed": ORDER_SEED,
        "python": platform.python_version(),
        "embedding_model": args.embedding_model,
        "embedding_revision": args.embedding_revision,
        "jev_model": args.jev_model if args.include_jev else None,
        "metrics": [metric.name for metric in METRICS],
        "rows": rows,
        "note": "Synthetic smoke data; quality figures do not establish real-world effectiveness.",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    asyncio.run(main())
