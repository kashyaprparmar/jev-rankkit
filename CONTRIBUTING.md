# Contributing

Jev Rankkit is MIT-licensed and welcomes bug reports, documentation corrections, tests, and narrowly scoped changes. Open an issue or pull request after a remote repository is configured; this local checkout currently has no GitHub remote.

Set up Python 3.11+ with `uv sync --group dev`. Run `uv run pytest`, `uv run pytest --cov=jev_rankkit --cov-report=term-missing`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src/jev_rankkit`, and `uv run python -m build`. Core tests must not need network access or API credentials. Run the offline examples with `uv run python examples/01_basic_reranking.py` and the benchmark smoke test with `uv run python benchmarks/compare.py`.

For ranking changes, include a small fixed-pool test covering input identity, ties, duplicates, top-k, missing scores, and failure semantics as relevant. For provider changes, mock structured responses and error classes. Live provider checks require explicit opt-in and a spending limit. Do not include API keys, private candidate text, or full provider payloads in issues, traces, or benchmark results.

Before changing a public protocol, explain the use case and how old adapters remain compatible. Update examples and the [evaluation guide](docs/evaluation.md) when score semantics change. Keep optional SDKs out of core imports.
