# Release checklist

Checked on 2026-09-23 after renaming the distribution to `jev-rankkit` and the import package to `jev_rankkit`. The package has not been published.

| Check | Status | Evidence |
| --- | --- | --- |
| Formatting | PASS | `uv run ruff format --check .`: 74 files already formatted. |
| Lint | PASS | `uv run ruff check .`: all checks passed. |
| Tests and coverage | PASS | `uv run pytest --cov=jev_rankkit --cov-report=term -q`: 103 passed, 1 skipped, 88% aggregate line coverage. The skipped case is the opt-in billable live Jev test. |
| Type checking | PASS | `uv run mypy src/jev_rankkit`: no issues in 36 source files. |
| Build | PASS | `uv run python -m build` created `jev_rankkit-0.1.0.tar.gz` and `jev_rankkit-0.1.0-py3-none-any.whl`. |
| Clean wheel install and import | PASS | Installed the wheel with `--no-deps` into a fresh Python 3.11 environment. Imported all 36 package modules in isolated mode and ran a deterministic reranking smoke test. Installed metadata reports `jev-rankkit` 0.1.0. |
| Distribution contents | PASS | Wheel contains `jev_rankkit`, `py.typed`, metadata, and license; it exposes only the selected import namespace and excludes tests/examples/benchmarks. Sdist includes source, tests, examples, benchmarks, documentation, license, and build metadata. |
| README and local links | PASS | Import examples and source links use `jev_rankkit`; distribution extras use `jev-rankkit`. All repository-local Markdown links resolve. |
| Metadata and version | PASS | `pyproject.toml`, lockfile, wheel metadata, and installed metadata agree on distribution `jev-rankkit` version `0.1.0`; import package is `jev_rankkit`. Python minimum is 3.11 and license is MIT. |
| Examples | PASS | All 15 offline examples ran successfully with the renamed imports. |
| Benchmark smoke tests | PASS | `benchmarks/compare.py` ran on six synthetic cases. Baseline and BM25 were measured; embedding, hybrid, hierarchical, and Jev routes stayed marked unmeasured. `benchmarks/python_overhead.py` also ran at 100, 1,000, and 5,000 candidates. |
| Name consistency and generated files | PASS | Distribution, import, extras, documentation, and CI configuration use the selected names consistently. `dist/`, build/test/type/lint caches, and coverage output are ignored by Git. |
| Name availability screen | CHECKED | PyPI JSON returned HTTP 404 for `jev-rankkit` on 2026-09-23. This does not reserve the name or provide trademark clearance; recheck immediately before publication. |

The opt-in live Jev test was not run because billable provider access was not enabled. The renamed synthetic benchmark smoke test measured baseline and BM25 only; embedding, hybrid, hierarchical, and Jev strategies remain unmeasured. These results do not establish production quality, latency, or cost.
