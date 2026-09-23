# Final audit resolution

Resolved against [`FINAL_AUDIT.md`](FINAL_AUDIT.md) on 2026-09-23. Status is deliberately conservative: **fixed** means the stated failure has a regression test; **partially fixed** means the implementation removes a concrete failure mode but an external contract or inherent runtime limit remains; **deferred** means no unsupported claim of resolution is made. The optional live TypeSafe test was not run because this environment has no authorized provider credential.

## Blockers

- **B-01 — fixed.** Fresh responses retain their candidate IDs, are validated by ID, and are never remapped by position. Pointwise cache records use an explicit canonical slot, and malformed listwise cache entries are deleted. Reverse-order and poisoned-cache tests cover the regression.
- **B-02 — fixed for async child tasks.** Pointwise and per-item metric windows own their children and cancel/await siblings before error propagation. The failure regression verifies no sibling model completion after `rerank` raises. An already-running synchronous extension callback in a worker thread remains non-preemptible; see H-08.

## High severity

- **H-01 — fixed.** Independent listwise chunks now raise a capability error before dispatch. Auto only chooses listwise when the entire selected group fits one batch; otherwise it uses pointwise or lexical ranking. This intentionally changes the former approximate listwise behavior because it could return an incorrect global top-k.
- **H-02 — partially fixed.** Backend `max_batch_size` is enforced and Jev estimates the full serialized payload plus output allowance before dispatch. Auto uses the same estimator and preflights the largest possible shortlist under configured token/context limits. Character estimates are not an exact tokenizer or a provider guarantee; a live contract and provider tokenizer remain necessary.
- **H-03 — partially fixed.** Failed operations release unused retry reservations; Jev reports attempted count in safe error context, and retries with unreported earlier usage mark token/cost completeness unknown. Optional pricing requires both input and output prices. Provider billing for failed attempts is unavailable, so exact spend cannot be reconstructed; strict cost still rejects Jev without a provider-backed upper bound.
- **H-04 — fixed for awaitable ranking paths.** One outer deadline starts at rerank entry and covers projection, queueing, metrics, embeddings, stages, and finalization. A slow embedding regression now raises `DeadlineExceededError`. Python cannot forcibly stop synchronous projection code or a worker thread already running; this is documented.
- **H-05 — fixed.** A cacheable `CallableMetric` must supply a versioned `cache_key_fn` that fingerprints item and context dependencies. Other custom cacheable metrics must supply `candidate_cache_identity`. A regression covers two same-text objects with different authority and two `as_of` values.
- **H-06 — fixed.** A configured `score_threshold` rejects non-utility score kinds instead of dropping valid fusion or rank-only fallback results. Tests cover both.
- **H-07 — fixed.** Zero-weight metrics are not executed or included in the score coverage requirement. A zero-weight LLM metric test verifies zero provider calls.
- **H-08 — partially fixed.** Synchronous metric, evaluation, and observer hooks run off the event loop; the metric path remains semaphore-bounded. Async callbacks still run directly. Worker-thread work cannot be forcibly killed on cancellation, and custom batch metrics can still perform their own uncontrolled I/O; extensions must honor their contracts.
- **H-09 — fixed.** Supplying weighted metrics with a per-call strategy object raises `ConfigurationError` instead of silently discarding the metrics. String strategy overrides retain the existing weighted-metric path.
- **H-10 — partially fixed, inherent model risk remains.** The default model route is pointwise, eligibility stays deterministic before projection, untrusted content remains labeled, and documentation explicitly recommends isolated pointwise judgment for hostile corpora. No prompt wording proves resistance to model manipulation; a real adversarial provider evaluation is still required. Automatic sanitization was not added because it could destroy ranking evidence while offering a false safety guarantee.
- **H-11 — partially fixed.** `tests/test_jev_live.py` provides a credentialed, opt-in wire-contract check and is skipped during normal CI. It was not executed here, so current provider compatibility is **unverified**. The mock tests and type checker do not close this gate.
- **H-12 — fixed.** `JevBackend` excludes the API key and transport from its generated representation. A synthetic-secret regression verifies redaction.

## Medium severity

- **M-01 — deferred.** Cross-request single-flight requires shared cancellation ownership and more state than a cache miss alone. Per-request child cleanup was fixed first. Add ranker-scoped dedup only after a measured stampede warrants it.
- **M-02 — partially fixed.** Batch metrics now acquire the shared semaphore. `EmbeddingSimilarity.cacheable` is false until a sound, versioned embedding-feature cache exists, avoiding a misleading contract. Reusable embedding vectors remain a future optimization.
- **M-03 — partially fixed.** Auto preflights model token/context/call feasibility, handles unavailable pointwise capability, and does not merge independent listwise groups. Count thresholds and retention factors remain uncalibrated; changing them without held-out domain data would replace one assumption with another.
- **M-04 — deferred.** The library still exposes each utility's semantics and requires explicit weights. No automatic min-max calibration was added because it can change rankings unpredictably. Measure weight sensitivity on held-out corpora first.
- **M-05 — fixed.** If `Retry-After` exceeds the configured maximum wait, Jev surfaces the rate limit instead of retrying early. The regression verifies a single attempt.
- **M-06 — fixed.** `plan()` now accepts per-call metrics, weights, prompt, and strategy overrides, matching `rerank()` selection. A test covers weighted strategy override and `top_k=0` stage counts.
- **M-07 — fixed.** Evaluation rejects an underfilled requested shortlist by default; `allow_partial=True` remains the explicit opt-in. Thresholded output has a regression test.
- **M-08 — partially fixed conservatively.** Judgment caching now requires a backend-declared `cache_stable=True`; unknown/custom backends default to no judgment caching. Jev enables caching only for exact numbered revisions. This may reduce cache hits for custom backends until they declare stability. Provider-side repointing of a numbered revision still cannot be ruled out without a resolved immutable model identifier.
- **M-09 — deferred.** Repeated Jev instructions were not relocated without a live check that TypeSafe gives shared instructions equivalent priority and semantics. Moving them speculatively risks ranking correctness or prompt security. Payload size is now checked before dispatch; measure provider token use before wire changes.
- **M-10 — partially fixed as evidence hygiene.** The synthetic comparison and Python-overhead probe were rerun; no representative quality or provider latency/cost claim is made. Held-out labeled data, live model access, hardware profiles, variance, and p95 measurements remain required before selecting production Auto thresholds.
- **M-11 — fixed for the built-in transport.** HTTP responses are streamed with a 2 MB byte cap before JSON parsing. Raw-text mock transport responses are checked too; third-party injected transports must apply equivalent limits themselves.

## Low severity

- **L-01 — deferred.** Public renaming would create compatibility churn unrelated to the release blockers. Keep `JevReranker` as an alias candidate for a later generic model-stage name.
- **L-02 — deferred.** The sync convenience signature remains `Any`; a typed request/overload design should be judged with real consumer examples rather than widening this patch.
- **L-03 — partially fixed.** This file and the README distinguish implemented behavior from unresolved design intent. A later architecture-plan refresh should mark every contract deviation before a stable API release.

## Verification and remaining gates

Targeted regressions were run immediately after each affected area. Final verification passed:

- `uv run pytest -q`: **103 passed, 1 skipped** (the opt-in live Jev test).
- `uv run pytest --cov=jev_rankkit --cov-report=term-missing -q`: **103 passed, 1 skipped; 88% aggregate line coverage**.
- `uv run ruff check .`: **passed**; `uv run ruff format --check .`: **73 files formatted**.
- `uv run mypy src`: **passed** across 36 source files.
- `uv run python -m build`: **wheel and source distribution built**.

The synthetic comparison is a six-case smoke test: local baseline and BM25 rows were measured, with NDCG@5 of 0.589 and 0.799 respectively on this toy pool; embedding, hierarchical, and Jev rows were `unmeasured`. These values do not establish production quality. The local Python-overhead probe returned median projection/rerank times of 0.300/0.786 ms (100 candidates), 3.709/8.932 ms (1,000), and 18.245/43.280 ms (5,000) on this host; these are diagnostics, not transferable service benchmarks.

Release gates still open: run the opt-in live Jev contract and adversarial prompt tests with authorized credentials; collect representative ranking-quality, latency, and cost data; review optional dependency inventory/SBOM; and decide whether failed-attempt provider usage can ever be reported precisely. None of those measurements or credentials were fabricated for this resolution.
