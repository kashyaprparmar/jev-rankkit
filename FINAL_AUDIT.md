# Final architectural audit

Audit date: 2026-09-23. Scope: the current `jev-rankkit` source, public docs, tests, and benchmark harness. This is a release-readiness review, not a claim that every provider behavior has been exercised. No production code was changed for this audit.

Severity means: **BLOCKER** can silently return incorrect rankings or leave billable work running after failure; **HIGH** can break a stated production constraint, trust boundary, or core API contract; **MEDIUM** creates material quality, cost, or extension risk; **LOW** is API/documentation debt. Locations are repository paths and current line numbers.

## BLOCKER

### B-01 — Valid backend scores can be assigned to the wrong candidates

- **Location:** `src/jev_rankkit/_runtime/executor.py:358-391` (`_remap_cached`), `src/jev_rankkit/backends/base.py:27-48` (`CandidateScore`/`ModelResponse`).
- **Problem:** Every complete response, including a *fresh* response, is remapped to candidate IDs by tuple position. The backend contract identifies scores by `candidate_id` and does not require response order. Raw cached records of the right length are also remapped before their IDs are validated.
- **Impact:** A valid reversed two-item response for `A=0.9, B=0.1` produced `B=0.9, A=0.1` in a local probe. The wrong item can be selected without an error; a malformed cached record can pass ID checks after its IDs are overwritten.
- **Recommended fix:** Validate original IDs and score coverage first. Reorder by ID, never reassign a score by position. If reusable cache records use ordinals, store an explicit ordinal-to-occurrence mapping and validate it before reconstruction. Add unordered, duplicate, unknown, and poisoned-cache response tests.

### B-02 — Failed batches leave sibling work running

- **Location:** `src/jev_rankkit/_runtime/executor.py:241-262,329-357,660-668`; `src/jev_rankkit/api.py:285-304,122-136`.
- **Problem:** Pointwise and per-candidate metric windows use `asyncio.gather` without cancelling and awaiting siblings when one fails. `_call_model` shields child tasks, while `Reranker` tracks only the outer rerank task. The resulting work can outlive the request and its fallback or error response.
- **Impact:** In a local probe, `rerank` raised for one candidate, then the other candidate's model call completed after the exception. A caller can be charged after believing the operation failed; `aclose` may race with these untracked children.
- **Recommended fix:** Give each request structured child-task ownership. On first failure or cancellation, cancel and await all siblings and their underlying model tasks before fallback or return. Keep deduplication reference counts within that scope. Test failure, cancellation, fallback, and close while siblings are queued and active.

## HIGH

### H-01 — Independent listwise chunks are compared as if calibrated

- **Location:** `src/jev_rankkit/strategies/listwise.py:34-59`; `src/jev_rankkit/auto.py:127-150`.
- **Problem:** Chunk scores are sorted globally although each listwise group has different comparison context. `approximate=True` discloses the shortcut but does not make probabilities comparable across groups.
- **Impact:** A weaker item in a generous chunk can outrank a stronger item in another; `top_k` can be wrong. `quality` mode automatically chooses this path for medium pools.
- **Recommended fix:** Do not merge independent listwise scores as a global utility. Use a common anchored calibration pass, pointwise scoring for cross-chunk finalists, or an explicit tournament with documented semantics. Keep the approximate path opt-in until measured.

### H-02 — Context and backend batch limits are not enforced at the shared boundary

- **Location:** `src/jev_rankkit/backends/base.py:17-22`; `src/jev_rankkit/strategies/listwise.py:34-41`; `src/jev_rankkit/auto.py:101-110`; `src/jev_rankkit/_runtime/executor.py:172-175,400-404`; `src/jev_rankkit/backends/jev.py:181-188`.
- **Problem:** `max_batch_size` exists but is never consumed. Auto and runtime estimates use candidate/query characters rather than the rendered prompt, repeated questions, output allowance, or a tokenizer. Jev reports no `max_context_tokens` and applies a separate character gate after planning.
- **Impact:** An accepted plan can fail at dispatch, or send a large paid request that the provider rejects. The planner may claim a one-call route that cannot fit.
- **Recommended fix:** Add a backend request-sizing method for the exact serialized payload and output allowance; enforce `max_batch_size` before dispatch. Make Auto use the same estimator and split or reject before paid work. Test long queries, long prompt examples, many short candidates, and backend-declared limits.

### H-03 — Retry ownership makes model-call and spend accounting unreliable

- **Location:** `src/jev_rankkit/backends/jev.py:191-250,341-348`; `src/jev_rankkit/_runtime/executor.py:93-150,405-447,462-535`.
- **Problem:** Jev retries inside `ModelBackend.score`, but the executor reserves up to `max_attempts` and observes attempt count only on a successful final response. Failed attempts have no usage record, reservations are not reconciled on terminal failure, and `model_calls`/`retries` undercount those attempts. Jev's optional cost estimate prices input tokens only, not output tokens or potentially billable failed attempts.
- **Impact:** Fallbacks can be rejected because stale reservations remain, while reported calls and cost can understate actual provider work. Non-strict monetary budgets are estimates, not enforceable spend ceilings; strict cost is unavailable for Jev without an upper-bound method.
- **Recommended fix:** Put retry scheduling under the budget ledger or emit one attempt-level result/error with usage for every attempt. Reconcile reservations in `finally`, including failure and cancellation. Add explicit input and output pricing only when verified, otherwise report unknown cost. Do not label a partial price as total cost. Test 429/5xx after one or more chargeable attempts and fallback afterward.

### H-04 — The latency budget does not cover the operation

- **Location:** `src/jev_rankkit/api.py:327-407`; `src/jev_rankkit/_runtime/executor.py:77-90,455-480`; `src/jev_rankkit/metrics/builtin.py:225-255`; `src/jev_rankkit/pipeline.py:220-252`.
- **Problem:** The budget clock starts after candidate preparation and is checked at model dispatch. Lexical work, embedding calls, custom metrics, and most pipeline work have no deadline. The model timeout begins after semaphore acquisition.
- **Impact:** A `Budget(max_latency_ms=1)` request using a slow embedding metric returned `status=ok` after roughly 124 ms in a local probe. Callers cannot rely on the advertised latency bound for non-model routes.
- **Recommended fix:** Start one monotonic deadline at rerank entry, include preparation and queue time, check it between stages, and wrap awaitable stages in the remaining deadline. Document that synchronous Python code and worker threads cannot be forcibly stopped. Test every route under a short deadline.

### H-05 — Cacheable custom metrics can reuse the wrong object's score

- **Location:** `src/jev_rankkit/_runtime/executor.py:620-657`; `src/jev_rankkit/metrics/builtin.py:190-205`.
- **Problem:** The per-candidate metric key includes projected text and metadata, but not the original object's other fields or `context.as_of`/`context.tags`. A `CallableMetric` receives the entire item and context and may be marked `cacheable=True` without supplying a dependency fingerprint.
- **Impact:** Two objects with the same text but different hidden authority values returned the first object's cached score for the second in a local probe. Time- or tag-dependent callbacks can also return stale scores.
- **Recommended fix:** Keep arbitrary callbacks uncached unless they provide an explicit, versioned cache-key function over every field and context dependency. Separate built-in pure metric caching from extension caching. Test same projection/different object and different context values.

### H-06 — One global threshold is applied to incompatible score kinds

- **Location:** `src/jev_rankkit/api.py:431-450`; `src/jev_rankkit/types.py:12-18`; `src/jev_rankkit/selection/fusion.py:18-45`.
- **Problem:** `score_threshold` is specified on the `[0,1]` utility scale, but final filtering applies it to fusion scores and rank-only results too. A rank-only input-order fallback has `score=None`, so any configured threshold removes every item.
- **Impact:** Valid RRF results (typically much smaller than 0.5 with the default constant) or fallback results can disappear unexpectedly; a successful strategy may return an empty list for a scale mismatch.
- **Recommended fix:** Permit the global threshold only for `ScoreKind.UTILITY`, or require an explicit score-kind-specific threshold and normalizer. Make incompatible combinations fail clearly before ranking; test RRF and rank-only fallbacks.

### H-07 — Zero-weight metrics still execute and can fail or incur cost

- **Location:** `src/jev_rankkit/metrics/base.py:69-90`; `src/jev_rankkit/_runtime/executor.py:582-620`.
- **Problem:** Zero is an accepted weight, but every configured metric runs before aggregation. A zero-weight LLM metric can make paid calls; a failing zero-weight callback fails the entire rerank. A local probe raised `RuntimeError` from a zero-weight callback.
- **Impact:** Weight changes intended to disable a signal do not disable its latency, spend, or failure risk.
- **Recommended fix:** Omit zero-weight metrics from execution after validating names, while recording disabled metrics in the plan if needed. Alternatively reject zero and require callers to remove the metric. Test that a zero-weight LLM metric makes no calls.

### H-08 — Synchronous extension hooks can block the event loop

- **Location:** `src/jev_rankkit/metrics/builtin.py:202-205`; `src/jev_rankkit/observability.py:59-67`; `src/jev_rankkit/evaluation/metrics.py:160-170`.
- **Problem:** Synchronous metric and observer callbacks are invoked directly from async tasks. A slow or blocking callback prevents deadline timers, other reranks, and even observer shutdown from progressing. The model semaphore does not make synchronous code preemptible.
- **Impact:** One user extension can stall unrelated requests and defeat bounded latency despite otherwise bounded model concurrency.
- **Recommended fix:** Define a cheap, nonblocking sync-hook contract or run blocking hooks in a bounded worker pool with explicit cancellation limits. Isolate observer callbacks from the ranking loop. Add a blocking-callback concurrency test.

### H-09 — Per-call strategy objects silently discard supplied metrics

- **Location:** `src/jev_rankkit/api.py:374-386,566-581`.
- **Problem:** `rerank(metrics=..., weights=..., strategy=<custom object>)` resolves the metrics, then selects the strategy object without passing those metrics to it or rejecting the conflicting arguments. String strategies reject most conflicts, so behavior depends on representation.
- **Impact:** The response can look like a weighted rerank while ignoring every requested business signal.
- **Recommended fix:** Reject metrics/weights with an incompatible strategy object, or formalize a composition API that passes weighted metrics explicitly. Test both string and object overrides.

### H-10 — Prompt delimiting is a mitigation, not an instruction boundary

- **Location:** `src/jev_rankkit/prompts.py:52-82`; `src/jev_rankkit/backends/jev.py:133-178`; `README.md` security and prompt sections.
- **Problem:** Candidate text is labeled untrusted, but it still enters model-visible structured input alongside ranking instructions; user query text is also inserted into instructions. In shared-context listwise mode one candidate can influence judgments of peers. There is no demonstrated provider-enforced role separation here.
- **Impact:** Adversarial documents, tool descriptions, or memories can bias selection. This must never be used as authorization or tool-permission enforcement. This is a risk finding, not a reproduced provider exploit.
- **Recommended fix:** Keep deterministic eligibility and permissions before projection, use provider-level instruction separation when available, offer isolated pointwise mode for hostile corpora, and run adversarial prompt-injection evaluations. Preserve the existing documentation caveat.

### H-11 — No live Jev contract gate protects the adapter

- **Location:** `tests/test_jev_backend.py`; `src/jev_rankkit/backends/jev.py:133-188,299-368`.
- **Problem:** Tests use mocked transports; they do not verify that the current TypeSafe endpoint accepts this payload, returns the expected Noul schema/usage, or respects the assumed context limits and retry behavior. Static type checks cannot cover an external wire contract.
- **Impact:** A package release could pass every local check yet fail or mis-score against the actual provider.
- **Recommended fix:** Add an opt-in, credentialed, minimal live contract suite with a pinned model/revision, safe fixture text, expected IDs/ranges, usage shape, timeout, and documented failure handling. Gate provider-specific release claims on its result; never run billable checks by default.

### H-12 — Backend representation prints an explicitly supplied API key

- **Location:** `src/jev_rankkit/backends/jev.py:86-100` (`@dataclass` and `api_key` field).
- **Problem:** The generated dataclass `repr` includes `api_key` and the injected transport. A local probe with a synthetic secret showed the full key in `repr(JevBackend(api_key=...))`.
- **Impact:** Normal debug logging, tracebacks, notebook output, or an object inspection can expose credentials. Environment-only keys are not printed, but the public constructor accepts explicit keys.
- **Recommended fix:** Declare `api_key` and `transport` with `field(repr=False)`; consider a deliberately safe custom representation for future credential-bearing fields. Add a regression test asserting a synthetic secret never appears in `repr` or public error strings.

## MEDIUM

### M-01 — Cross-request cache misses duplicate billable calls

- **Location:** `src/jev_rankkit/_runtime/executor.py:169,309-355`; `tests/test_strategies.py:89-101`.
- **Problem:** In-flight deduplication is per `ExecutionServices` instance, hence per rerank call. Concurrent identical requests using one cached ranker both miss and call the backend; the test explicitly expects two calls.
- **Impact:** A traffic burst can stampede the provider and amplify cost/rate limits.
- **Recommended fix:** Add optional ranker-scoped single-flight keyed by the complete tenant-aware cache key, with reference-counted cancellation. Keep cache-only behavior as the simpler default if measured stampedes are negligible.

### M-02 — Batch metrics bypass the cache and shared call controls

- **Location:** `src/jev_rankkit/_runtime/executor.py:607-620`; `src/jev_rankkit/metrics/builtin.py:225-255`.
- **Problem:** The `score_many` branch runs directly, before the per-item cache/semaphore path. Embedding similarity declares `cacheable=True` but re-embeds the query and candidates on each call. A custom batch metric can launch unconstrained work outside the executor's call budget.
- **Impact:** Repeated embedding work increases latency and resource use; custom batch network calls are not covered by advertised model limits.
- **Recommended fix:** Cache query and document embeddings with separate versioned keys and provide a bounded batch execution service. State explicitly which third-party metrics opt out of budget guarantees.

### M-03 — Auto routing has unmeasured thresholds and incomplete estimates

- **Location:** `src/jev_rankkit/auto.py:48-230`.
- **Problem:** Routes at 10/100 candidates and retention factors 20/40 are fixed heuristics. Estimated tokens omit repeated query/prompt material and pointwise call multiplication. `top_k=None` on a pool above 100 falls back to lexical full ordering even in quality mode.
- **Impact:** A route can be unexpectedly expensive, fail a budget mid-run, or sacrifice quality on a full-order request. The heuristic choices have no domain evidence yet.
- **Recommended fix:** Treat all thresholds as configurable and versioned; preflight using the same backend estimator as execution; explain why full-order mode changed quality. Benchmark candidate-count/size/top-k bands before changing defaults.

### M-04 — Weighted `[0,1]` signals are not automatically comparable

- **Location:** `src/jev_rankkit/metrics/base.py:85-90`; `src/jev_rankkit/metrics/builtin.py:28-55,60-116,210-276`.
- **Problem:** Lexical recall, pool-dependent BM25 utility, mapped cosine, recency, and Jev Noul probability all fit `[0,1]`, but represent different distributions and meanings. A weighted sum is mathematically valid, not necessarily calibrated for ranking quality.
- **Impact:** User-chosen weights may not produce the intended tradeoff; candidate-pool changes alter BM25 scale.
- **Recommended fix:** Document per-signal semantics in results, keep weight defaults task-neutral, and evaluate calibration/weight sensitivity on held-out pools. Do not normalize each request by min-max without measuring rank effects.

### M-05 — `Retry-After` can be shortened below the provider's request

- **Location:** `src/jev_rankkit/backends/jev.py:233-241`; `src/jev_rankkit/config.py:65-79`.
- **Problem:** The delay is capped at `max_backoff_s` even if the response asks for a longer `Retry-After` (default cap: two seconds).
- **Impact:** Repeated early 429s waste attempts and can worsen throttling.
- **Recommended fix:** Honor the server delay up to a separately configured maximum wait and the remaining request deadline; otherwise stop retrying and surface the rate limit. Test numeric and HTTP-date headers.

### M-06 — Prospective plans omit important per-call inputs

- **Location:** `src/jev_rankkit/api.py:230-260` versus `src/jev_rankkit/api.py:269-390`.
- **Problem:** `plan()` accepts neither per-call metrics/weights, prompt, nor strategy override. It always plans the constructor strategy, while `rerank()` can choose another path.
- **Impact:** A caller may approve or budget an inspectable plan that does not describe the subsequent rerank call.
- **Recommended fix:** Share one immutable request-spec preparation path between `plan` and `rerank`; reject unsupported plan inputs rather than silently omitting them. Test plan/actual agreement for every override.

### M-07 — Evaluation can silently compare underfilled top-k outputs

- **Location:** `src/jev_rankkit/evaluation/runner.py:132-158`; `src/jev_rankkit/api.py:443-450`.
- **Problem:** The evaluator checks `ResultStatus.PARTIAL`, but score thresholds and some strategies can return fewer than `rerank_top_k` with status `OK` or `FALLBACK`. It does not explicitly require k results or report coverage alongside aggregate metrics.
- **Impact:** Quality comparisons can mix full and underfilled outputs without an obvious diagnostic.
- **Recommended fix:** Add an explicit underfill policy and per-case coverage to reports; require `min(k, eligible_count)` unless the policy permits abstention. Test threshold and fallback underfill.

### M-08 — Cache freshness depends on a narrow alias naming rule

- **Location:** `src/jev_rankkit/_runtime/executor.py:309-312`; `src/jev_rankkit/backends/jev.py:118-120`; `src/jev_rankkit/backends/sentence_transformers.py:24-27`.
- **Problem:** Model caching is disabled only for names ending `-latest` or `-preview`. Other mutable aliases, repointed endpoints, and unpinned embedding revisions can retain the same identity. The model response's `resolved_model` is not part of lookup identity.
- **Impact:** A cache can serve scores produced by a different underlying model until TTL expiry.
- **Recommended fix:** Make immutability/revision an explicit backend capability; disable reusable model/embedding caches when unresolved, or require a short, explicit stale-result policy. Include endpoint, resolved revision, preprocessing, and task type in feature keys.

### M-09 — Listwise payload repeats static instructions for every question

- **Location:** `src/jev_rankkit/backends/jev.py:133-179`.
- **Problem:** Every question copies the same system, criteria, rubric, and query instruction fields even when the candidate pool is shared in `state`.
- **Impact:** Prompt bytes grow with candidate count, increasing context failures and potentially token cost. The exact charge depends on provider accounting and has not been measured.
- **Recommended fix:** Verify whether TypeSafe supports a shared instruction location with equivalent semantics. If so, send common instructions once and keep only per-target fields in questions. Measure token usage before changing the wire format.

### M-10 — There is no decision-grade benchmark for routing or superiority claims

- **Location:** `benchmarks/compare.py:1-183`; `benchmarks/data/synthetic_v1.json`; `docs/benchmarks.md`.
- **Problem:** The comparison uses six small synthetic pools, a fixed order seed, one pass, and optional provider rows that are normally unmeasured. The Python-overhead probe excludes provider time. The docs correctly label these as smoke tests; no benchmark number is fabricated, but none validates the Auto thresholds, model quality, or production latency/cost.
- **Impact:** Default routing and business-value assumptions remain unverified for RAG, entities, tools, SQL, and other domains.
- **Recommended fix:** Build held-out, position-labeled domain datasets and compare fixed candidate pools across baseline, lexical, embedding, hybrid, pointwise, listwise, and cascades. Report quality at k, shortlist recall, cold/warm p50/p95 latency, call/token usage, cost confidence, variance, hardware, model revision, and failures. Keep unavailable strategies marked unmeasured.

### M-11 — Provider response size is unbounded before JSON parsing

- **Location:** `src/jev_rankkit/backends/jev.py:58-79,299-309`.
- **Problem:** The HTTP client reads the whole response, and `_parse_response` then parses it without an explicit byte limit. Duplicate-key detection is applied only when a `.text` body is exposed.
- **Impact:** An unexpectedly large or malformed provider response can consume substantial memory/CPU before schema validation.
- **Recommended fix:** Bound response bytes at the transport layer, reject excess before JSON decoding, and keep duplicate-key checks on all supported transport paths. Test oversized and deeply nested responses.

## LOW

### L-01 — Public naming duplicates concepts and embeds a provider name

- **Location:** `src/jev_rankkit/api.py:143-166,716-719`; `src/jev_rankkit/pipeline.py:136-173`.
- **Problem:** `AutoReranker` and `Reranker(strategy="auto")` expose the same route; `JevReranker` uses any `ModelBackend`. The latter is a leaky provider name in a provider-independent stage.
- **Impact:** Users may infer different behavior or unnecessary Jev coupling; renaming after adoption creates compatibility cost.
- **Recommended fix:** Choose a canonical path, keep aliases during migration, and add a generic `ModelReranker` name before a stable release.

### L-02 — Sync convenience loses generic typing

- **Location:** `src/jev_rankkit/api.py:686-714`; `src/jev_rankkit/api.py:584-601`.
- **Problem:** `rerank_sync(**kwargs: Any) -> RerankResponse[Any]` and convenience methods with `**kwargs: Any` bypass the generic result type preserved by async `rerank`.
- **Impact:** Static checkers cannot connect input object type to result type for these documented entry points.
- **Recommended fix:** Use a shared typed request object or targeted overloads/`Unpack[TypedDict]` without duplicating runtime logic; add mypy/pyright consumer examples.

### L-03 — The approved architecture document describes stronger contracts than the code

- **Location:** `IMPLEMENTATION_PLAN.md:177-179,243-255,400-458,562-566` versus the current executor, backend, and result models.
- **Problem:** The plan calls for a one-attempt backend, temporary scopes for unscoped calls, cache-record validation, and richer budget/trace reports. The implementation has backend-owned retries, reusable unscoped resources, limited cache-record checks, and compact tracing. Some differences may be intentional, but they are not marked as deviations.
- **Impact:** Reviewers and integrators may implement extensions against a contract that the package does not uphold.
- **Recommended fix:** After remediation, update the plan to distinguish implemented guarantees, deferred design, and explicit deviations. Treat released code and API docs as the contract.

## Test scenarios missing from the current suite

The suite passes, but line coverage does not establish these behaviors. Add the following focused regressions alongside each fix:

1. A valid complete backend response in reverse/random order, then the same response from cache; corrupted cached IDs and duplicate scores.
2. One pointwise or callback task failing while siblings are queued/running, with zero work left after error, fallback, cancellation, and `aclose`.
3. Zero-weight LLM/callback metrics causing zero work; custom strategy plus metrics failing explicitly.
4. Same projected text/metadata with different object-only callback dependencies; `as_of`, tenant, authorization revision, and changed model revision cache behavior.
5. Max batch/context sizes with long query/prompt, short candidates, and retry output allowances; strict and non-strict budgets after failed attempts.
6. Latency deadlines on embedding, lexical, callback, and multi-stage routes; blocked observer behavior.
7. RRF and rank-only fallback with a configured score threshold; underfilled top-k evaluation.
8. Prompt-injection fixtures for individual and shared-context judgments, with a live provider run before release.
9. Secret redaction in backend representations and errors; a minimal live TypeSafe contract check and dependency/SBOM review for optional HTTP and embedding extras. This audit did not perform a vulnerability scan or claim any current dependency vulnerability.

## Remediation order and release gates

1. **Before any production use:** fix B-01 and B-02. Acceptance: ID-order invariance and no post-failure child work are covered by deterministic regression tests; cancellation and close leave zero active backend calls.
2. **Before advertising budgets/caching as production controls:** fix H-03, H-04, H-05, H-06, and H-07. Acceptance: attempt-level usage and reservation reconciliation are visible on success/failure; non-model routes respect the request deadline or explicitly return a deadline error; cache keys represent all custom metric dependencies; thresholds are scale-aware; zero weights cause no work.
3. **Before relying on Auto or listwise for quality-sensitive selection:** fix H-01/H-02, resolve H-09, and measure M-03/M-04/M-10. Acceptance: no automatic uncalibrated cross-chunk merge, backend size limits preflight, plan and execution agree, and held-out domain results justify the route.
4. **Before a Jev-backed release:** resolve H-10/H-11/H-12 and M-05/M-09/M-11. Acceptance: no secret in backend representations, documented prompt-injection limits, live wire-contract result, bounded response parsing, verified retry behavior, and measured prompt/token overhead. Keep model-cost estimates unknown until complete pricing/usage is verified.
5. **Subsequent hardening:** address cache stampedes, embedding reuse, evaluation coverage, mutable model identities, and public API/type/documentation debt. Preserve backward compatibility where users could already depend on names or result fields.

The current verification performed for this audit was `uv run pytest -q` (84 passed), `uv run pytest --cov=jev_rankkit --cov-report=term-missing -q` (88% aggregate line coverage), `uv run ruff check .` (pass), and `uv run mypy src` (pass). The five local probes above used fake backends/callbacks; no live TypeSafe call or decision-grade quality benchmark was run. Passing checks and synthetic benchmark output do not close the release gates listed here.
