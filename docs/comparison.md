# Jev Rankkit and jev-reranker

This comparison reflects [`jev-reranker` 0.1.2](https://pypi.org/project/jev-reranker/) and the current Jev Rankkit source tree, checked 2026-09-23. Both projects are young; verify current releases before choosing one.

[`jev-reranker`](https://github.com/hotchpotch/jev-reranker), by Yuichi Tateno, provides a focused Jev integration for text documents. It offers sync and async reranking and relevance filtering, configurable prompts and thresholds, listwise, pointwise, and exhaustive pairwise modes, context-aware splitting, bounded concurrency, retries, optional tokenizer-based sizing, and detailed request/usage diagnostics. It also includes an evaluation example. These are substantial features, and its purpose-built Jev implementation may be the simpler choice for a Jev-only document workflow.

Jev Rankkit targets a broader ranking problem. It preserves arbitrary Python objects, uses backend and metric protocols, combines lexical, embedding, metadata, and model scores, and supports deterministic RRF, diversity selection, inspectable cascades, and automatic strategy routing. It adds budget policies, cache isolation, lightweight framework projections, and an offline evaluation API. Jev is an optional backend rather than a dependency of ranking algorithms.

Jev Rankkit does **not** currently implement `jev-reranker`'s exhaustive pairwise mode, tokenizer-based request partitioning, or relevance-filtering prompt preset. Its synthetic benchmark is a reproducibility smoke test, not evidence that it ranks better. Jev-specific users who need those features should evaluate the existing package directly. Choose Jev Rankkit when candidates include tools, entities, products, memories, schema objects, or other application objects, or when multiple ranking methods must share one typed result and execution report.

The two packages are independent. Jev Rankkit credits the existing project's Jev reranking work and does not claim to replace or outperform it. Compare quality, latency, and cost on your own fixed candidate pools before making a production choice.
