# Reproducible benchmark smoke test

The [synthetic dataset](../benchmarks/data/synthetic_v1.json) has six original, small candidate pools: general retrieval, technical RAG, entity search, products, tool selection, and SQL schema selection. The script shuffles each fixed pool with seed `20260923` and records the dataset SHA-256. This is an integration and measurement smoke test, **not** a representative quality benchmark.

```bash
uv sync --group dev
uv run python benchmarks/compare.py --output benchmark-local.json
```

The default compares input ordering and candidate-pool BM25. It lists embedding similarity, lexical-plus-embedding hybrid, hierarchical ranking, Jev pointwise, and Jev listwise as `unmeasured` when their backends are unavailable. No numeric field is invented for an unavailable run.

To opt into a local embedding model, install `jev-rankkit[embeddings]` and pass `--embedding-model MODEL_ID` (ideally also `--embedding-revision REVISION`). Model weights may download on first use. To permit billable Jev requests, install `jev-rankkit[jev]`, set `TYPESAFE_API_KEY`, and pass `--include-jev`. External model versions, machine hardware, rate limits, and candidate order affect results. Never compare model quality using this toy dataset alone.

The separate [Python overhead probe](../benchmarks/python_overhead.py) measures projection and lexical reranking on this machine only. It excludes provider/model latency. For production decisions, use a held-out labeled pool from your own domain, collect uncertainty across multiple queries and runs, and compare quality against latency and cost at the same candidate budget.
