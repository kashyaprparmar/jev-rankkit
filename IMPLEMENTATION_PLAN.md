# Jev Rankkit: universal reranking architecture and implementation plan

Date: **2026-09-22**. Status: **proposed design, revised after self-critique; no production implementation**.

## 1. Repository inspection and architectural decision

The current repository directory contains only [`docs/research.md`](docs/research.md). There is no package, build configuration, test suite, or Git metadata. No applicable `AGENTS.md` was found in the workspace or its inspected ancestors. The research is the evidence base for this plan; this document adds decisions and contracts, not new benchmark claims.

Build **jev-rankkit**, an async-first Python library for ranking and selecting arbitrary candidate objects. Keep the original object attached to every result. Make deterministic metrics, model judgments, ranking algorithms, and set selection independently usable. Jev is a first-class optional backend, not the definition of the package.

The initial product is a small library, not a retrieval service or an agent framework. Its differentiators are inspectable multi-stage ranking, explicit score semantics, generic object preservation, and shared reliability/budget controls across stages. Some of these capabilities already exist separately in other libraries; this design does not claim algorithmic novelty.

Python **3.11+** is the proposed baseline. The core uses the standard library. Provider clients, model runtimes, evaluation datasets, and telemetry exporters are optional dependencies. Constructor calls and package imports perform no network access, model loading, environment-file loading, or plugin discovery.

## 2. Package naming

### 2.1 Shortlist and availability evidence

The following ten exact PyPI JSON endpoints returned **HTTP 404** during this design session on 2026-09-22. A direct check of `jev-rankkit` on 2026-09-23 also returned HTTP 404. This means an accessible project was not listed at that endpoint; it does **not** prove the name is registrable, unreserved, or cleared for trademarks. Recheck normalized distribution names, import names, related projects, and branding before publishing. No name has been reserved or published.

1. **jev-rankkit** — combines ranking signals and methods; concise and pronounceable. Strongest overall. [PyPI check](https://pypi.org/pypi/jev-rankkit/json)
2. **rankplane** — suggests a ranking infrastructure layer. Professional, although “plane” is abstract. [PyPI check](https://pypi.org/pypi/rankplane/json)
3. **rankspan** — suggests broad applicability across objects and domains. Short, but “span” also has established programming meanings. [PyPI check](https://pypi.org/pypi/rankspan/json)
4. **rankgrain** — suggests fine-grained judgment. Memorable, although the word has incidental prior uses and does not clearly suggest orchestration. [PyPI check](https://pypi.org/pypi/rankgrain/json)
5. **rankstitch** — suggests composing stages and evidence. Clear, slightly less natural to say. [PyPI check](https://pypi.org/pypi/rankstitch/json)
6. **ranklattice** — suits relationships and structured candidates. Professional, but longer and could imply a particular mathematical algorithm. [PyPI check](https://pypi.org/pypi/ranklattice/json)
7. **ordiloom** — distinctive ordering/weaving metaphor. Broad and short, but less immediately searchable as ranking software. [PyPI check](https://pypi.org/pypi/ordiloom/json)
8. **candidrank** — foregrounds candidates of any type. Understandable, though slightly awkward and potentially read as “candid.” [PyPI check](https://pypi.org/pypi/candidrank/json)
9. **siftorder** — communicates selection and ordering without tying itself to a model. Two concepts make it a little less fluent. [PyPI check](https://pypi.org/pypi/siftorder/json)
10. **siftwise** — memorable judgment/selection name. Broader than ranking and therefore less precise. [PyPI check](https://pypi.org/pypi/siftwise/json)

**Selected distribution name: `jev-rankkit`; Python import name: `jev_rankkit`.** This reflects the maintainer's requested package name. PyPI normalizes the distribution name; Python identifiers use underscores. The Jev-specific name is a branding choice and creates a tighter association with that provider than the earlier provider-neutral design goal. Availability checks are only a point-in-time collision screen, not reservation or trademark clearance.

Rejected alternatives include `rankweave` and `rankkit`, which are already listed on PyPI. Several otherwise plausible names have visible software/SEO brands, including RankMesh, RankFrame, RankPilot, RankFold, RankStride, RankTide, and RankFolio. Naming around Jev, RAG, documents, or an individual ranking algorithm would unnecessarily constrain the project. [RankWeave](https://pypi.org/project/rankweave/), [RankKit metadata](https://pypi.org/pypi/rankkit/json), [RankMesh](https://rankmesh.ai/about), [RankFrame](https://rankframe.com/), [RankFold](https://www.rankfold.com/en/pricing), [RankTide](https://ranktide.app/), [RankFolio](https://www.rankfolio.app/)

## 3. Product boundaries and quality priorities

The library accepts a finite candidate sequence supplied by the application. It evaluates relevance and other declared criteria, orders or selects candidates, and returns original objects plus auditable decisions. It neither discovers candidates nor executes them.

Priorities are:

- **Quality:** preserve candidate recall, support task-specific rubrics and metadata, distinguish individual relevance from useful set composition, and require measurements for defaults that trade recall for cost.
- **Latency:** cheap checks before network work, bounded batches, limited parallelism, optional session reuse, and small expensive shortlists. Account for queueing, projection, tokenization, loading, and cache time.
- **Cost:** reuse immutable features/judgments, reserve resources before dispatch, and make retries and fallbacks share the original budget.
- **Reliability:** strict boundary validation, explicit failure modes, cancellation propagation, and coherent fallback rankings.
- **Developer experience:** one entry point and useful result objects; advanced configuration is optional and grouped by concern.
- **Type safety:** generic result preservation, typed protocol boundaries, and runtime validation of external data. Python annotations do not prove model correctness.
- **Extensibility:** explicit backend/metric instances and shallow composition. No global registration magic or general workflow language.

Authorization, eligibility, ownership, inventory constraints, schema compatibility, and tool permissions belong in deterministic code. Jev is useful for semantic relevance, ambiguous entity matching, intent/tool fit, evidence usefulness, and other bounded judgments with clear rubrics. Arithmetic, timestamps, exact matches, sorting, fusion, budget accounting, and ID reconciliation remain deterministic.

## 4. Public user experience

All code blocks in this plan are **proposed API examples**, not existing implementation. `TypeVar`-based contracts must work on Python 3.11; newer generic syntax is not required.

### 4.1 Simple usage

```python
from jev_rankkit import Reranker

reranker = Reranker(model="typesafe:jev-1.13.0")
response = await reranker.rerank(
    query="best database for vector search",
    candidates=candidates,
    top_k=5,
)

for result in response.results:
    use(result.item)  # the exact original object
```

A namespaced string selects a built-in adapter and an explicit provider model identifier. The example revision comes from the research snapshot; support is checked by the adapter, not hardcoded as an eternal default. Unknown namespaces or unsupported capabilities raise configuration errors. `model=` also accepts a `ModelBackend` instance. It never silently chooses a paid provider. Credentials use the provider's documented configuration; the library does not load `.env` files itself.

With a model and no metrics, the default is **pointwise** ranking with the backend's declared relevance metric: the versioned `llm_relevance` rubric for Jev/generative judgment backends, or `cross_encoder_relevance` with the model's native score semantics for a cross-encoder. A backend without a declared compatible default requires explicit metrics. With `model=None`, the default is lexical ranking, so basic installation works offline. Supplying explicit metrics replaces these defaults. `Reranker` does not silently switch to an approximate strategy because a request is large; users opt into `AutoReranker` or an explicit pipeline.

Pointwise means each candidate's judgment is independent of other candidates. Provider transport batching is allowed only when it preserves that meaning. Jev shared-state multi-question scoring is a separate listwise mode, not hidden pointwise batching.

Strings work without adapters. Other objects need `text_fn` or an explicit `CandidateAdapter[T]`; there is no fallback to `str()`, `repr()`, reflection, or arbitrary attribute guessing.

```python
response = await reranker.rerank(
    query=query,
    candidates=products,
    text_fn=lambda item: f"{item.name}\n{item.description}",
)
# A checker should infer RerankResponse[Product].
```

### 4.2 Multiple metrics and prompts

```python
from jev_rankkit import Reranker
from jev_rankkit.config import RerankerConfig

reranker = Reranker(
    model=jev_backend,
    embedding_backend=embedding_backend,
    config=RerankerConfig(),
)
response = await reranker.rerank(
    query=query,
    candidates=items,
    text_fn=render_item,
    metadata_fn=lambda item: {
        "authority": item.authority_0_to_1,
        "published_at": item.published_at,
    },
    prompt=my_prompt,
    metrics=[
        "semantic_similarity",
        "llm_relevance",
        "authority",
        "recency",
    ],
    weights={
        "semantic_similarity": 0.25,
        "llm_relevance": 0.50,
        "authority": 0.15,
        "recency": 0.10,
    },
    top_k=10,
)
```

`my_prompt` can be a string describing the relevance criterion, or an immutable `PromptSpec` containing instructions, rubric, and version. It affects model judgments only. Candidate content remains separately serialized data. There is no executable templating language or implicit Python formatting of candidate fields into instructions.

Named metrics have documented contracts. `semantic_similarity` requires an embedding backend or compatible supplied vectors. `authority` requires caller-supplied bounded metadata; `recency` requires an aware timestamp and an explicit/recorded time-decay configuration. Missing requirements fail before remote work. No metric is inferred by asking an LLM to invent missing metadata. The aliases `semantic_similarity` and `embedding_similarity` resolve to the same canonical metric and cannot be double-counted under two names.

For this convenience example, named `authority` reads `authority` in `[0,1]`; named `recency` reads `published_at` with a documented **30-day half-life** and rejects future timestamps by default. This is a recorded starter policy, not an empirically optimal default. Domain applications should supply a configured `RecencyMetric(half_life=...)` under the same metric name when another decay policy is appropriate. Named metrics expose their resolved configuration in the plan.

For advanced customization, `metrics` accepts `Metric[T]` objects as well as names. Example conceptual configuration: `CallbackMetric(name="business_priority", fn=priority, scale=BoundedScale(0, 100))`. Arbitrary callbacks are trusted application code. They must declare missing-value behavior and a version for persistent caching.

`Reranker` itself is **not** generic. Its `rerank` method is generic in the candidate element type, which connects `candidates`, `text_fn`, `metadata_fn`, supplied `Metric[T]`, and `RerankResponse[T]`. Constructor defaults contain only candidate-independent configuration and backends. Object-specific adapters and metric instances belong to the call or to a separately typed `RerankPipeline[T]`. This avoids a bare `Reranker(model=...)` silently becoming `Reranker[Any]`.

### 4.3 Proposed call surface

The principal keyword-only contract is:

```python
async def rerank(
    self,
    *,
    query: str,
    candidates: Sequence[T],
    top_k: int | None = None,
    text_fn: Callable[[T], str] | None = None,
    metadata_fn: Callable[[T], Mapping[str, MetadataValue]] | None = None,
    id_fn: Callable[[T], str] | None = None,
    eligible_fn: Callable[[T], bool] | None = None,
    adapter: CandidateAdapter[T] | None = None,
    metrics: Sequence[str | Metric[T]] | None = None,
    weights: Mapping[str, float] | None = None,
    prompt: str | PromptSpec | None = None,
    strategy: str | RankingStrategy[T] | None = None,
    context: RerankContext | None = None,
) -> RerankResponse[T]: ...
```

Additional knobs live in immutable `RerankerConfig`, strategy instances, or per-call `RerankContext`, not another fifty method arguments. `adapter` is mutually exclusive with the projection callbacks it replaces. `eligible_fn` always runs as a separate gate. Unknown keywords, conflicting configuration, metric-name duplicates, and unsupported prompt/strategy combinations are errors.

Configuration precedence is explicit per-call option → ranker config default → documented library default, subject to hard administrator limits. `None` means inherit for optional overrides; an empty metric list is an error rather than a request to inherit. Non-string or whitespace-only queries are rejected; application-specific query objects must be explicitly rendered before calling. Candidate sequences of mixed types retain their union type rather than being coerced to dictionaries.

`top_k=None` requests a full ordering of eligible candidates. Negative values and booleans are invalid. `top_k=0` and empty input return a validated no-op response without projection, cache access, model initialization, or provider calls. A value larger than the eligible pool is clamped. No implicit relevance threshold removes candidates; configured eligibility, abstention, or selection constraints may return fewer than k with a recorded reason. Sequence inputs are intentionally finite; generators and unbounded streams are outside the first contract.

### 4.4 Sync convenience and session lifecycle

`rerank_sync(...)` has the same arguments and result type and runs a fresh operation scope through `asyncio.run`. It raises a clear error when called from an already running event loop; notebook/async callers must `await`. Never apply `nest_asyncio` or create an invisible helper event-loop thread.

The simplest async call opens and closes its owned backend sessions within that call. Repeated calls can reuse resources explicitly:

```python
async with Reranker(model=jev_backend) as reranker:
    first = await reranker.rerank(query=q1, candidates=items1)
    second = await reranker.rerank(query=q2, candidates=items2)
```

This scope is bound to one event loop and owns adapters it creates. Injected live clients are borrowed, explicitly marked as such, and never closed by the library. A borrowed loop-bound client cannot be used by `rerank_sync`; pass a backend/session factory instead. Concurrent calls inside one scope are permitted, each with its own request budget and shared backend rate/concurrency limits. Scope exit waits for or cancels tracked work and closes owned resources. Scope re-entry, use after close, and cross-loop use fail explicitly. Stateful sync pooling is deferred; there is no second sync execution engine.

Outside an explicit async scope, calls on an unscoped ranker remain reusable but each owns a temporary execution scope. A configured cache object can outlive those scopes only if its implementation supports that ownership/loop model; otherwise cache reuse is restricted to the explicit scope. `rerank_sync` cannot be called on a ranker whose async scope is active, even from a different thread. Sharing connection/rate pools across different ranker instances requires an explicitly injected shared backend resource; there is no process-global account semaphore.

## 5. Core data contracts and invariants

Use frozen dataclasses and typed enums/unions for public value objects; validate at construction/boundaries. Frozen containers do not make a caller's mutable `T` immutable.

### 5.1 Candidate preparation

`CandidateView[T]` contains `item: T`, `input_index`, opaque `occurrence_id`, optional `candidate_id`, a frozen text/structured projection, and a copied metadata snapshot. The view is internal to evaluation but is available through the extension API for custom metrics. Model serializers receive only the allowed projection and explicit public metadata, never the `item` reference or private policy metadata.

Preparation order is: validate static arguments; snapshot the candidate sequence; evaluate deterministic eligibility; assign occurrence IDs; project eligible objects exactly once; validate/copy declared metadata; freeze the representation; estimate resources. Eligibility checks may read the object, but projection, logging of content, embedding, remote scoring, and cache lookup must not see ineligible candidates.

The application must not mutate ranking-relevant object state while preparation/execution is in progress. The immutable projection is the evidence that was ranked, while `result.item` is the original live reference and may reflect later application mutations; the library cannot make an arbitrary object snapshot immutable without changing its identity. Projection fingerprints and the captured `as_of` make this distinction auditable. A caller needing snapshot objects should supply its own immutable candidate records.

Occurrence identity is unique within a request and preserves repeated occurrences, even if they reference the same Python object. `id_fn` provides an optional stable **domain ID**, not occurrence identity. Duplicate domain IDs do not silently deduplicate. Cross-ranking fusion requires unique domain IDs or an explicit occurrence mapping; ambiguous duplicates raise. Object references remain local and are never persisted in caches.

Metadata supports an intentionally small typed value set: strings, booleans, finite numbers, aware timestamps normalized to UTC, null, and bounded immutable lists/maps of those values. Serialization canonicalizes these types explicitly. Unknown objects, cycles, non-string map keys, excessive depth, NaN, and infinity fail. The application controls allowlists and redaction. There is no automatic dataclass/Pydantic serialization of every field.

### 5.2 `RerankResult[T]`

Fields:

- `item: T`, `input_index: int`, `occurrence_id: str`, `candidate_id: str | None`.
- `rank: int`: one-based position in the returned selection, not original input position.
- `score: float | None` and `score_kind`: the final ordering score before optional set selection, with values such as `utility`, `raw_metric`, `relative_preference`, `fusion`, or `rank_only`. `raw_metric` also names its originating metric/raw kind, for example BM25 or cross-encoder logit.
- `metric_scores: Mapping[str, MetricScore]`: raw values, normalizer/version, normalized utility when defined, and judgment status.
- `selection_score: float | None`: an MMR or other step-dependent set-selection value, separate from base relevance.
- `role`: `ranked` or `dependency`, plus compact provenance identifying the scoring/fallback stage.

`MetricScore` includes `raw_value: float | None`, `raw_kind`, `utility: float | None`, `status`, optional provider confidence with its named definition, rubric ID, and optional typed distribution. Large distributions and prompt bodies are opt-in diagnostics, not default per-result payloads. No field called confidence is synthesized from an arbitrary score. Missing values remain missing.

After MMR or dependency handling, results need not be in descending `score` order. The `rank` is authoritative. A rank-only backend has `score=None`; do not manufacture a probability from its position. Dependencies included for context can have no relevance score.

### 5.3 `RerankResponse[T]`

Fields: immutable `results`, `status`, `coverage`, `statistics`, `execution_plan`, `execution_report`, warnings, and request ID. `status` is one of `ok`, `noop`, `fallback`, or `partial`. Approximation is an independent report attribute: a successful tournament can be `ok` and approximate.

Coverage reports counts for input, eligible, projected, each stage's evaluated/pruned/missing candidates, and selected candidates. It distinguishes incomplete evaluation from intentional pruning. A response describes the requested top-k; it does not claim a complete ranking of candidates pruned by a cascade. `.results` is the canonical collection; avoid simultaneously behaving as a dictionary, list, and response object.

### 5.4 `RerankContext` and `RerankerConfig`

`RerankContext` is immutable caller intent: request ID, tenant/cache namespace, authorization-policy revision, aware `as_of`, per-call budgets, quality mode, optional deadline, and safe user tags. If omitted, `as_of` is captured once at call start and returned in the report. Absolute deadlines use a monotonic clock internally; wall time is for metadata semantics only. Context is not mutable execution state and carries no API keys or raw prompts in log tags.

`RerankerConfig` groups limits, retry policy, fallback policy, cache policy, default strategy, normalization defaults, prompt/rubric defaults, and observer configuration. Per-call limits may tighten an administrator ceiling but not relax it. Mappings are defensively copied. Actual counters, semaphore leases, reservations, cancellation, and session handles are private `_ExecutionState` objects created per operation.

## 6. Public interfaces, internal APIs, and dependencies

### 6.1 Supported public surface

The root exports only `Reranker`, `AutoReranker`, `RerankResult`, `RerankResponse`, `RerankContext`, and `RerankerConfig`. Stable submodules expose:

- `metrics`: `Metric[T]`, `MetricScore`, `CallbackMetric`, built-in configured metrics, and normalization specifications.
- `strategies`: `RankingStrategy[T]`, pointwise/pairwise/listwise/tournament strategy configurations and hierarchy configuration.
- `pipeline`: `RerankPipeline[T]`, `RankStage`, `FusionStage`, and `SelectionStage`.
- `backends`: `ModelBackend`, `EmbeddingBackend`, typed requests/responses, capability descriptions; concrete providers live in explicit submodules.
- `cache`: `CacheBackend`, `MemoryCache`, cache policy and versioned cache records.
- `selection`: RRF utilities, `MMR`, and bounded dependency/coverage selection policies.
- `prompts`, `errors`, `observability`, and `evaluation`: small typed contracts described below.
- `types`: extension-facing `CandidateView[T]`, `RankingOutcome`, `EvaluationServices`, `PreparedPlan[T]`, `ExecutionPlan`, and score/coverage records. Anything appearing in a public protocol signature is public and versioned; private compiled stages never leak through it.

Public extension contracts receive the same compatibility discipline as the root API. Underscore modules/classes, planner rules, batch packers, wire DTOs, task scheduling, and cache digests are internal. Mark experimental strategies/evaluation helpers explicitly until their semantics stabilize. Do not export internal request graphs or provider SDK classes as core types.

### 6.2 Minimum protocols

**`Metric[T]`:** declares a unique name, version, score kind/normalizer, required fields, purity/cache policy, and whether it is networked or blocking. An async batch method consumes a read-only sequence of `CandidateView[T]` plus query/context and an `EvaluationServices` capability object; it returns records keyed by occurrence ID. It must provide exactly one success/missing/error record per requested occurrence, in arbitrary order. Scores do not contain `T`, which keeps object association in one validated place.

`EvaluationServices` offers controlled judgment/embedding calls and budget-aware local work submission. It is the sole supported network path for custom extensions that want library budget/retry guarantees. A malicious or bypassing Python callback cannot be sandboxed; such extensions are trusted code and must not be advertised as budget-enforced. Built-in metadata callbacks are synchronous, quick, and network-free; async/batch callbacks use a distinct constructor so no awaitable is accidentally treated as a number.

**`RankingStrategy[T]`:** consumes the prepared eligible pool, configured criteria, context, and evaluation services, returning a public `RankingOutcome` of occurrence IDs, optional scores, and coverage that the executor validates. It never returns replacement objects. It declares full-order/top-k capability, context dependence, and whether it may prune. It cannot dispatch unmetered built-in model calls or mutate eligibility. Prefer built-in strategy objects over custom implementations until an actual extension needs this protocol.

**`ModelBackend`:** exposes a capability descriptor and creates/borrows an async session. A session estimates an immutable `ModelRequest` and executes **one attempt**. Requests are a small discriminated union: independent judgments, shared-context judgments, pair comparisons, and permutation ranking. Backends reject unsupported variants before dispatch. Responses contain typed judgments/orderings, usage, provider/model identity, and request IDs, never domain objects.

Capabilities include supported primitives, score meanings, context limits, independent batching vs shared context, supported modalities, usage/token-estimation quality, retry classification, and cancellation/timeout support. Numeric limits are adapter data with a source/version, not assumptions embedded in every strategy. One backend need not implement all request variants. Estimation distinguishes known upper bounds from approximate estimates.

**`EmbeddingBackend`:** batch embeds query/candidate projections with dimension, model revision, normalization, modality, and usage information. Separate this interface from judgment generation. A vector provider is not required to pretend it supports comparisons or JSON responses.

**`CacheBackend`:** asynchronous `get_many`, `set_many`, and namespace invalidation/delete for versioned byte/JSON records; bounded TTL and size policy. Misses and cache errors are distinct. Memory cache has a small bounded implementation. No distributed locking protocol or Redis dependency in core; stampede suppression is local to one execution scope.

**`Observer`:** receives immutable typed events. A bounded internal event queue protects execution from slow consumers; loss counters are visible. Core observers do not change ranking decisions. Critical policy decisions use explicit policies, not callbacks with hidden veto power.

`CandidateAdapter[T]` is a convenience value holding projection/metadata/ID callbacks with versions. It is not an object superclass or a domain-model framework. No candidate must inherit anything.

Result/view containers expose read-only object slots and can be covariant in `T`; callback/metric protocols consume candidate types and should use appropriate contravariance where their complete signatures allow it. Keep pipelines invariant initially because they bind both typed callbacks and inputs. Confirm variance and method inference with checker fixtures before freezing signatures. Built-in named metrics use candidate-neutral views and a typed binding function, not casts that attach arbitrary backend output to `T`.

### 6.3 Dependency direction and optional extras

Dependency direction is inward: foundational value types/errors/protocols import no executor or concrete provider; metrics, strategies, preparation, and selection depend on those contracts; the executor composes those components; the facade assembles the executor and explicitly selected adapters. Concrete adapters depend on contracts, not the facade. Public `EvaluationServices` is a protocol implemented by the private runtime, preventing an import cycle. Core never imports optional provider implementations until selected; a small explicit registry resolves built-in names.

Proposed extras:

- `jev-rankkit[jev]`: official `typesafe-sdk` transport behind an adapter. Verify retry disablement and session semantics before choosing it permanently; a small direct HTTP adapter is an alternative if those contracts cannot be met, not a second default implementation.
- `jev-rankkit[sentence-transformers]`: explicit local embedding and CrossEncoder adapters, bringing their model-runtime dependencies only when requested.
- `jev-rankkit[llm]`: one selected structured-output provider adapter initially. Do not pull in a universal routing framework just to support every model string.
- `jev-rankkit[otel]`: OpenTelemetry exporter for observer events.
- `jev-rankkit[eval]`: standard ranking metrics/dataset helpers, with reproducible manifests and optional task datasets.
- Future `jev-rankkit[redis]` or faster lexical extras only after demand and measurements. An external cache implementation can already satisfy `CacheBackend`.

Core lexical scoring can use a small bounded in-memory BM25 implementation over the supplied candidate pool, with a documented tokenizer and formula. This is candidate scoring, not an index/search engine. A `bm25s` adapter is optional if benchmarks justify it. Model downloads require explicit backend configuration and count as initialization work; offline mode forbids implicit downloads.

## 7. Metric and score semantics

### 7.1 Built-in metrics

- **Lexical:** token overlap for cheap baselines and BM25 for candidate-pool relevance. BM25 records tokenizer, parameters, and corpus-statistics scope. Changing the pool changes IDF and thus judgments; it is not a per-item cache key.
- **Semantic/embedding similarity:** query-to-candidate vector similarity with model/preprocessing compatibility checks. Precomputed vectors are accepted through explicit features, not searched for inside arbitrary objects. Dot product and cosine are separate declared kinds.
- **Cross-encoder relevance:** joint query/candidate inference. Preserve model-specific raw logits/outputs and apply only a configured documented transformation.
- **LLM relevance:** precise binary proposition, anchored ordinal rubric, or relative judgment according to backend capabilities. The default relevance rubric and its mapping are versioned assets.
- **Recency:** `2 ** (-age / half_life)` with nonnegative age measured against context `as_of`; future timestamps follow an explicit reject/clamp policy. Half-life is domain configuration, not something a model guesses. Keep cached semantic judgments independent from recomputed time decay.
- **Authority:** trusted application metadata in a declared range. Source reputation lookup is the application's responsibility.
- **Popularity:** caller data with a declared transform, such as capped `log1p` against a configured reference. The reference is fixed/versioned, not a maximum silently inferred from each batch.
- **Business priority:** explicit utility/boost from trusted metadata. Record its contribution separately from semantic relevance; hard constraints stay in eligibility.
- **Python callback:** explicit batch/scalar, sync/async, range/direction, missing-value, version, and blocking behavior. No automatic type coercion of strings, booleans, or awaitables to scores.

### 7.2 Normalization and aggregation

All weighted aggregation uses finite utilities in `[0,1]` with “higher is better.” A metric's raw value retains its original scale. Suggested mappings must be named and versioned:

- Already bounded utility: range validation or a declared affine mapping. Out-of-range values fail by default; silent clipping hides bugs.
- Jev Noul: identity only when the proposition matches the utility objective; label the raw value as a proposition probability, not generic confidence.
- Jev Score: expected ordinal index divided by `L-1`, preserving rubric and optional distribution. This is utility, not probability of correctness.
- Cosine: `(cosine + 1)/2` is an explicit bounded similarity mapping; it is not relevance calibration. Validate vector norms/dimensions and finite values.
- BM25: raw scores support ordering directly. Weighted combination requires explicit `s/(s+pivot)` with positive fixed pivot, a learned calibration transform, or explicit rank normalization. The convenience default lexical-only path does not need to invent a normalized score.
- Cross-encoder logits: configured monotonic mapping such as sigmoid is allowed and labeled uncalibrated; a calibrated probability requires held-out evidence.
- Rank-only model output: no scalar metric for weighted mixing. Use rank fusion or an explicitly requested rank-to-utility transform.

Reject unknown weight names, missing weights when a weight map is supplied, nonfinite/negative weights, and all-zero weights. Normalize positive weights by their sum once; unspecified weights mean equal weights over the chosen compatible metrics. A zero-weight metric is not executed unless explicitly requested for diagnostics. Output records the effective weights. Do not normalize separately for each candidate.

Missing-score policy defaults to error. Opt-ins are: drop the metric for **the entire stage** and renormalize once; use an explicitly declared imputed utility for all affected records; or return partial results after excluding failed candidates. The last two are degraded policies and visible in status. Never silently substitute zero or average different metric sets per candidate. Pool-dependent min-max and percentile normalization are opt-in, recorded as such, and keyed by the whole pool; they cannot make scores comparable across queries.

### 7.3 Determinism and ties

For identical prepared inputs, configuration, cached/provider judgments, and local numerical backend, deterministic orchestration must produce the same order. Models, accelerator arithmetic, mutable aliases, and remote transport do not receive a blanket reproducibility guarantee.

The default tie key is descending score followed by original input index. Preserve exact duplicate occurrences. No random jitter, hash-map iteration order, or unrecorded seed breaks ties. Float comparison uses exact finite values; optional score quantization must occur before sorting and be versioned, rather than an epsilon comparator that can violate transitivity. Tournament seeding/randomized diagnostics require an explicit recorded seed. Cross-run invariance under input permutation requires an opt-in unique stable-ID tie key.

## 8. Ranking strategies and set selection

### 8.1 Pointwise

Evaluate each candidate against the query and declared metrics; aggregate compatible utilities and stable-sort. With a single raw sortable metric, preserve its kind instead of forcing normalization. Batch independent local cross-encoder jobs or provider-supported independent requests, bounded by count/tokens/memory. Work is proportional to candidates times criteria, not necessarily the number of HTTP requests. Request batching is an optimization with preserved semantics.

### 8.2 Pairwise

Compare unordered candidate pairs using a shared rubric and explicit tie/neither behavior. The exact round-robin strategy has `n(n-1)/2` pair comparisons; validate feasibility before starting. A comparison result is a relative preference, not independent relevance. Weighted metadata combinations require a separately declared common utility policy or a pointwise stage; they are not automatically blended into pairwise win rates.

Support an explicit aggregation policy such as average fractional wins, with wins=1, ties=0.5, losses=0 and stable ties. Preserve cycles and comparison coverage in diagnostics; never feed an inconsistent model comparator directly to Python sorting. For Jev, Choice with tie/neither is a proposed adapter mode; two opposite Noul judgments with declared symmetrization can be another explicit mode. They are not interchangeable cache entries. Reverse-order tests measure bias. A singleton has no pairwise evidence: return it with `score=None`, not invented relevance.

### 8.3 Listwise

Two distinct capability modes share the user concept of listwise ranking:

1. **Shared-context judgments:** all projected candidates in a batch share state and receive separately keyed judgments. Sort in Python. This matches the inspected `jev-reranker` listwise mechanism.
2. **Permutation output:** a capable generative backend returns an ordered list of allowed IDs. Validate exact coverage, uniqueness, and membership; the result is rank-only unless the backend genuinely supplied defined scores.

Record the mode in the plan. Shared-context batch membership and order affect judgments. When the entire pool cannot fit, do not score independent windows and pretend their raw scores share a scale. Require an explicit window merge strategy (overlap/rank fusion with defined missing-rank policy) or hierarchical shortlist and final common-context rerank. Both are approximate and require position/chunk-boundary evaluation. Listwise truncation is never silent.

### 8.4 Tournament

Use a deterministic seeded bracket or configured successive rounds to shortlist candidates for a final pointwise/listwise rerank. Maintain round outcomes and eliminated membership. Single elimination finds a winner under its comparison assumptions; it does not give a trustworthy full ordering or exact top-k. Reject full-order requests for that mode. Top-k tournament mode uses explicit survivor/loser reconsideration rules and marks the result approximate. Do not market an `O(n)` winner algorithm as exact top-k ranking. Pairwise cycles and bracket sensitivity remain evaluation risks.

The first concrete tournament variant is a fixed binary winner tree: seed leaves by input order or a recorded shuffle; byes advance without synthetic wins; compare siblings; extract the winner; remove its leaf and replay only its ancestor path to obtain the next survivor. Collect configured `s >= top_k` survivors and rerank them together using a declared final strategy. Reuse a pair judgment only when query, pair order/context, rubric, and backend revision are identical. Tree construction plus extraction uses `O(n + s log n)` comparison decisions for a balanced tree, excluding retries and the final rerank. Those are algorithmic counts, not measured provider latency or a quality guarantee. Additional Swiss/league variants are deferred.

### 8.5 Hierarchical and cascaded

Hierarchy groups candidates using a caller-supplied parent/group mapping: rank groups, retain a bounded number, rank members, then rerank the combined finalists. Group summaries are caller-provided projections or deterministic excerpts; generating summaries is a separate budgeted opt-in model operation, deferred from the first release. No representative can guarantee group recall.

A cascade is a sequence of cheap-to-expensive stages over one pool. Each stage declares its survivor limit, score semantics, dependencies, and whether previous scores are retained as features. Pruned candidates remain represented in coverage, not as fabricated scored results. `top_k=None` disallows pruning stages. Early-pruning recall is evaluated separately from final-stage quality.

### 8.6 Hybrid and Reciprocal Rank Fusion

Hybrid means composing evidence sources, not a new model primitive. Support weighted utility aggregation when scales have declared mappings, and RRF when only ranks are comparable. Parallel lexical/dense branches may score the **same authorized input pool** and union selected IDs before model reranking; the library does not retrieve extra candidates from external databases.

For one-based rank `r_j(d)`, weighted RRF is `sum_j weight_j / (c + r_j(d))`, with positive `c`, nonnegative weights, and absent items contributing zero. Choose a documented conventional default such as `c=60` only as configuration, not a proven optimum. Each source contains an ID at most once; duplicates error. Source cutoffs, ranks, weights, constant, and fusion universe enter provenance/cache keys. Stable input-index ties apply. RRF scores are fusion values, not probabilities; globally absent candidates do not acquire invented ranks. An explicit input-order baseline can supply ranks for otherwise uncovered candidates when a full ordering is required.

### 8.7 MMR/diversity

MMR is a final greedy selection policy over a relevance shortlist: at each step maximize `lambda * relevance - (1-lambda) * maximum_similarity_to_selected`. The initial redundancy penalty is zero. Require `lambda` in `[0,1]`, an explicit bounded relevance utility, and a compatible declared similarity function. Default embedding redundancy maps negative cosine to zero and preserves positive cosine; record that mapping. Do not automatically treat raw BM25 or RRF values as the required relevance utility.

Maintain each remaining item's maximum similarity incrementally, with roughly `O(nk)` comparisons and `O(n)` working state, avoiding an unconditional quadratic matrix. Preserve base score and per-step selection score separately. Hard source/parent quotas or dependency closure are separate deterministic constraints, not implied by diversity. MMR does not guarantee fairness, truth, or optimal coverage.

## 9. Domain support without domain frameworks

All domains use the same `T`, projections, metadata, eligibility, metrics, and selection contracts. Lightweight example adapters can ship later; they must not import a database, agent SDK, or graph framework in core.

- **Documents, chunks, search results:** title/body/evidence spans and source/parent IDs. Parent caps and diversity reduce duplicates; retain evidence for multi-hop answers. The ranking package does not fetch URLs or generate final RAG answers.
- **Entities and products:** attributes and canonical IDs with explicit missing/conflicting fields. Exact identifiers, price, stock, locale, and required attributes are hard checks. Jev can assess semantic match/fit. Ranking is not entity merging or identity proof.
- **Tools, MCP tools, APIs, agents:** descriptions, input schemas, permissions, capabilities, and verified cost/latency metadata. Filter permissions and schema compatibility first. Rank capability fit; never execute tools, agents, or API requests as part of ranking. Tool descriptions are untrusted content, not instructions.
- **SQL schemas, tables, columns:** textual names/descriptions/types plus explicit foreign-key/parent dependencies. Select coherent bundles, retaining needed join relationships. Never connect to a database, inspect unauthorized schema, or execute SQL implicitly.
- **Graph nodes/edges:** typed relations, endpoints, provenance, and bounded caller-provided neighborhoods. Edge selections may need endpoint closure. No automatic graph traversal outside the authorized supplied pool.
- **Memories:** content, owner, timestamp, supersession/conflict metadata, and revision. Ownership/deletion/revocation are hard gates. Recency does not prove truth; ranking never deletes or rewrites memories.
- **Code and arbitrary objects:** caller-provided bounded semantic descriptions. Do not execute code, call properties recursively, pickle objects, or introspect secrets.

Dependency-aware selection operates only over supplied eligible IDs. `top_k` counts **all returned objects**, including dependencies. The default strict policy rejects impossible mandatory closure; an explicit best-effort policy skips an over-budget candidate and tries the next. It records dependency-only results with no invented relevance score. Caller-supplied bundles are often simpler: a schema bundle can itself be `T`, making its contents an application concern. Full general graph optimization is outside scope.

## 10. Execution runtime and production behavior

### 10.1 One execution path

The facade prepares candidates and compiles configuration into a bounded ordered list of stages with optional independent scoring branches. A single internal executor runs it. `Reranker`, explicit pipelines, and `AutoReranker` use this same path; sync convenience invokes it rather than duplicating it.

Each operation proceeds through static validation → eligibility/projection snapshot → resource planning → stage cache lookup → bounded execution/validation → normalization/order → selection → final eligibility check → response. Planning is partly data-dependent, so projection work and local estimates count toward the deadline. Final eligibility rechecks cannot recall already-dispatched remote content if access was revoked mid-call; applications needing stronger revocation guarantees must control admission and provider data handling.

If final eligibility differs from the admitted pool, default to a `ConstraintError` with no candidate payload rather than return a selection with broken permissions or dependency closure. The caller can initiate a new request against current policy. This authorization change is not a model failure and does not trigger an availability fallback. Dependency closure and quotas are revalidated before any response, including a configured fallback; successful result ranks are contiguous after selection.

Stage outcomes form safe checkpoints. A checkpoint contains one coherent ordering, eligible occurrence mapping, score semantics, and coverage. Later failures can fall back to an allowed checkpoint without combining incomparable partial judgments. No candidate object is copied or reconstructed from a model response.

### 10.2 Batching, concurrency, and hot paths

Batch packing enforces candidate count, projected/token size, question count, provider context, response size, and local memory limits. Check both total request size and provider-specific state/question restrictions. Length-aware packing may improve utilization for independent inference; shared-context packing changes semantics and must preserve its declared grouping/order.

Use a bounded worker queue rather than creating a task for every candidate or pair. Enforce both per-call concurrency and shared backend scope concurrency/rate limits. Fair scheduling prevents one large request from occupying all slots. Do not assume concurrent requests equal concurrent useful GPU batches; a local adapter can serialize access and batch internally.

Project once, tokenize/estimate once per relevant representation, cache reusable candidate embeddings, and coalesce compatible typed questions only when context semantics match. Recompute query-specific similarity cheaply. Avoid repeated full-object serialization in inner loops. Small metadata callbacks run directly; blocking synchronous callbacks require explicit thread offload and bounded concurrency. CPU-heavy tokenization/model inference uses an adapter-managed worker, process, or runtime rather than blocking the event loop. Arbitrary thread work cannot be forcibly stopped; a cancelled call discards its results and reports the limitation.

Limits include maximum candidates, projected bytes, per-candidate bytes, queued tasks, in-flight request tokens, and result diagnostics size. Oversized inputs fail or use an explicitly configured truncation/chunking policy with coverage metadata. The library never silently cuts the most relevant part of a document because a request exceeded context.

### 10.3 Budgets and accounting

`Budget` contains optional input/output/total token limits, monetary amount/currency (initially USD only), total elapsed duration/deadline, and request/comparison limits. Monetary arithmetic uses `Decimal` or integer micro-units, not binary floats. Pricing schedules carry provider/model/revision, effective date, and version. Unknown/free/local costs are distinct; local token and device work still exist when provider monetary cost is zero.

One per-operation ledger is shared by every stage, branch, retry, repair, and fallback. Before dispatch, reserve a conservative known upper bound where available; reconcile with observed usage afterward. Atomic reservations prevent concurrent overspend relative to those bounds. Retry backoff and queue wait count toward latency. No fallback gets a fresh budget.

Distinguish **strict** and **estimated** enforcement:

- Strict token/monetary mode permits paid dispatch only when the adapter can provide a defensible upper bound for chargeable work, including output caps and known pricing. Otherwise raise `BudgetUnverifiableError` before that dispatch or choose a feasible local fallback. Estimated tokenizer counts are not exact limits.
- Estimated mode reserves estimates plus a configured margin, reports estimate quality, and may exceed the target when actual billing differs. Never label this a hard spending guarantee.
- For timeouts after possible provider acceptance, keep the ambiguous reservation until known usage arrives or conservatively charge it to the operation's internal allowance. Unknown billed usage is reported, not reset to zero. Remote cancellation and exact provider billing cannot be guaranteed by a client library.

Check remaining budget before initialization, batching, queue admission, each attempt, each next stage, and fallback. Default exhaustion raises with a content-free execution report. An explicitly selected checkpoint/partial policy may return less work. Latency budgets enforce local deadlines and stop scheduling; they cannot guarantee a remote job or a blocking third-party callback has stopped. Application/provider account limits are needed for a true external spend ceiling.

### 10.4 Retries, timeouts, cancellation

One retry layer owns attempts. Disable SDK retries where possible; if not possible, the adapter must surface their count/resource bounds or be incompatible with strict budgeting. Honor bounded `Retry-After`, exponential backoff, and jitter without exceeding the deadline. Retry only classified transient connection failures, throttling, and selected server failures. Authentication, authorization, unsupported capability, malformed input, and programming errors are not retryable.

Structured-output validation retries have a separate small limit and still spend from the same ledger. Context overflow is not retried unchanged: reject or replan only through an explicit approximate split policy. Generic LLM schema repair is an opt-in new attempt, never invisible text surgery. Do not cache failures as valid judgments.

Separate connect/read/per-attempt timeouts from the total operation deadline. Propagate caller cancellation, cancel pending tasks, and close/release owned resources in `finally`. Cancellation never initiates a fallback. Distinguish provider timeout from deadline exhaustion and user cancellation in reports; preserve `asyncio.CancelledError` semantics.

### 10.5 Failure taxonomy and fallback

Expose `RerankError` with typed subclasses: `ConfigurationError`, `ProjectionError`, `CapabilityError`, `ContextLimitError`, `BackendError`, `OutputValidationError`, `BudgetExceededError`, `BudgetUnverifiableError`, `DeadlineExceededError`, and `ConstraintError`. Error reports contain stage, reason, safe usage/counters, and retryability; payloads/secrets are excluded by default. Metric failures carry candidate coverage without embedding objects in exception text.

Default behavior is **strict failure**. Production availability policies are explicit:

1. `previous_stage`: use the last complete eligible checkpoint; retain its score kind and mark `fallback`.
2. `lexical`: rerank the entire eligible requested pool or explicitly declared checkpoint pool with lexical scoring if remaining budget permits; record scope and changed objective.
3. `input_order`: return eligible input order with `score=None` and `fallback` status. This is an availability policy, not a relevance judgment.
4. `partial`: return only validated, comparably scored candidates with incomplete coverage and `partial` status; never pad with fake scores.

A configured ordered fallback chain is finite and cycle-free. Fail-fast static configuration/authorization errors do not fall back. If no feasible policy can produce a result, raise with its report. Recheck eligibility after cache retrieval and before returning. A fallback never relaxes hard constraints. For mixed successful/failed batches, a full-stage checkpoint fallback is preferred over silently mixing original and fallback scoring scales.

## 11. Cache design and invalidation

### 11.1 Cache judgments and features, not live results

Caching is disabled unless configured. The convenience enabled cache is bounded in-memory within a ranker scope; its implicit namespace is private to that ranker, with no disk persistence. Shared or persistent stores require an explicit namespace; multi-tenant applications must also supply the tenant/trust partition in context. No implicit cross-tenant sharing is allowed. Cache reusable embeddings and immutable model judgments first. Rebuild final results, time-dependent metrics, eligibility, and selection from the current request. Final-response caching is deferred because policy, identity, and temporal invalidation are too easy to get wrong.

Canonical keys use a versioned deterministic serialization and a cryptographic digest, never Python `hash()`, `repr()`, credentials, or raw object pickles. Shared persistent stores should use a tenant-scoped keyed digest where lookup-guessing is a concern. Digesting a payload does not by itself anonymize it or authorize sharing.

The judgment key envelope includes:

- Cache schema version, namespace/tenant/trust scope, authorization-policy revision, and projection/redaction version.
- Exact query bytes and relevant instructions/rubric bytes plus their version; Unicode/whitespace normalization only when explicitly part of the versioned preprocessing contract.
- Exact prepared public candidate content/metadata consumed by the judgment, including truncation/chunking/preprocessing policy.
- Backend/provider identity, endpoint identity without secrets, **resolved immutable model revision** or explicit cache scope for mutable aliases, request primitive, generation settings/seed where applicable, and adapter version.
- Every semantically relevant stage input: ordered peers for shared context, full ordered options including tie/neither for Choice, pair direction/symmetrization for comparisons, and corpus fingerprint/tokenizer for BM25.

Embedding keys omit the query for reusable candidate vectors but include model revision, text preprocessing, dimensions, normalization, and provider/task prefix. Query embedding keys include the query projection. Metric normalizations can be recomputed from raw cached judgments; if normalized utilities are cached, the normalizer/calibration version is also required.

Stable domain IDs alone are never sufficient. Two same-ID objects with changed content must miss the content-dependent cache. Request-local occurrence IDs are mapped to deterministic request-local wire ordinals for model payloads; cached ordinal results are validated against the identical payload fingerprint and remapped to current occurrences. Do not include a random request UUID in a reusable content key.

### 11.2 Invalidations and errors

Mutable aliases such as `latest` do not receive indefinite persistent caching. Default to disabling persistent judgment cache for unresolved aliases; an explicit short TTL/version namespace is a documented stale-result tradeoff. Inventory, memory edits/deletion, schema revisions, permission changes, source updates, and redaction changes need namespace/revision invalidation as well as TTL. A cache hit never bypasses eligibility.

Recency uses the current captured `as_of`; reuse underlying timestamp/features, not yesterday's decayed score. If a custom callback depends on time, external state, or a closure, persistent caching is disabled unless the caller supplies the dependency snapshot/version. Function names and source inspection are not valid dependency fingerprints.

Validate cache record schema, checksum/fingerprint, ID coverage, finite values, score kind, and model provenance on read. Corrupt/unknown records are misses plus a warning. Cache outages default to bypass with visible statistics; sensitive policies may opt into fail-closed behavior. Writes use TTL/size limits; eviction is normal. Negative caching of transient model errors is disabled. Never deserialize executable payloads.

Local in-flight deduplication shares only identical tenant-scoped, semantically independent requests. It needs reference-counted cancellation so cancelling one waiter does not cancel another call. Defer cross-operation in-flight sharing from the first release if that complexity is not justified; ordinary cached results already provide most reuse without coupled lifetimes.

## 12. Jev and other backend adapters

### 12.1 Jev integration

Integrate the official TypeSafe API through a thin adapter, rather than depending on `jev-reranker` as the universal engine. Reuse the upstream project's ideas with license/attribution review if code is ever reused; do not assume its dictionary/string API can become the generic core by wrapping it.

The adapter translates typed ranking requests to System One state/questions and validates exact answer coverage. Noul, Score, and Choice remain distinct primitive contracts. Jev does not natively generate an arbitrary sorted JSON array, free-text explanation, or chain-of-thought response. The adapter must reject unsupported requests, not emulate them silently. Shared-state questions are independent; a question cannot depend on another answer in that same call. [TypeSafe System One](https://docs.typesafe.ai/concepts/system-one), [primitives and confidence evidence](docs/research.md#3-jev-and-typesafe-ai-apis)

For the default pointwise rubric, use a precisely defined Noul relevance condition. For graduated task utility use an explicitly selected anchored Score rubric and preserve its scale. Shared-context listwise mode serializes a bounded pool and asks one compatible judgment per occurrence. Pairwise Choice or directional Noul comparisons use explicit mode names. Jev's documented limits and pricing must be adapter capability/pricing data verified before release; this plan does not freeze today's limits or claim latency.

Typed SDK models do not eliminate semantic validation: independently check expected IDs, answer types, distributions, finite ranges, and usage. Verify unknown answer handling and SDK retry behavior against a pinned compatibility range. Scrub SDK debug logging. No default request/response body logging or automatic propagation of tracing tags into model-visible state.

### 12.2 Other adapters

Cross-encoder adapters accept independent joint-text scoring requests and return model-defined raw values. Embedding adapters produce reusable vectors and separately compute declared similarities. Both define truncation, batching, device/session ownership, and output activation. Local inference can still have expensive initialization and memory limits.

A structured-output LLM adapter can implement independent rubric judgments, pair comparisons, and permutation requests where supported. Use provider-native structured output or constrained decoding when available, followed by library validation. Exact allowed IDs, cardinality, finite bounds, and permutation coverage remain mandatory. Model refusal is a typed failure/abstention, not an empty successful ranking. Capability checks must prevent unsupported `temperature`, schema, or reasoning options from being silently discarded.

Backend-specific options live inside that adapter's configuration; the common `Reranker` API does not pretend all model settings are portable. Third-party backends opt in via explicit instances, not arbitrary import strings discovered from candidate data.

## 13. AutoReranker and visible execution plans

### 13.1 Purpose and API

`AutoReranker` chooses among **configured available backends and deterministic stages** using versioned rules and measured profiles. It is not an LLM agent and does not buy access, download unknown models, or contact new providers. `Reranker` provides stable explicit behavior; `AutoReranker` is the opt-in quality/cost policy layer over the same executor.

When metrics are explicitly supplied, they remain required objectives: Auto may change batching/order/shortlisting, but cannot silently replace `llm_relevance` with lexical relevance. Omitting metrics opts into the configured Auto relevance policy, whose allowed stages appear in the plan. Any degraded objective is possible only through an explicit fallback policy. Both facades support `plan(...)`/`execute(...)`; `Reranker.plan` compiles its explicit configuration, while `AutoReranker.plan` additionally chooses stages.

```python
from jev_rankkit import AutoReranker, RerankContext

ranker = AutoReranker(
    model=jev_backend,
    embedding_backend=embedding_backend,
    config=auto_config,
)
plan = await ranker.plan(
    query=query,
    candidates=items,
    text_fn=render_item,
    top_k=5,
    context=RerankContext(quality_mode="balanced", budget=budget),
)
response = await ranker.execute(plan)
```

The convenience `await ranker.rerank(...)` performs those steps automatically. `plan` prepares and freezes eligible projections and retains original local objects, so `execute(plan)` uses the **same snapshot** rather than reprojecting mutable objects. A prepared plan is process-local, typed in `T`, immutable, and single-use; its objects/credentials are not exportable. Expired deadlines or invalid policy/backend versions trigger rejection/replanning. It performs final eligibility checks during execution as usual.

`plan.describe()` exports a safe `ExecutionPlan` without objects, credentials, or candidate text. It contains stages, reasons, limits, approximate/exact attributes, score kinds, token/request/cost estimates or unknowns, assumed model revisions, cache assumptions, and fallback policy. An executable plan is distinct from this report. Planning invokes no model inference and no new model download; local projection, tokenizer work, and cache metadata queries are bounded and visible. Capability discovery requiring network must be explicit and cannot masquerade as free planning.

Estimates do not reserve provider spend indefinitely. Execute verifies remaining limits and reserves before each dispatch. A cache hit or miss can change observed work; the runtime records changes. Replanning only selects pre-authorized policies/backends and records the reason. No silent expansion of budget or change of task objective is allowed.

### 13.2 Planner inputs

Use candidate count **and** per-candidate/query projected sizes, context/token estimates, requested k, full-order vs shortlist intent, quality mode, active metric/normalization requirements, latency/token/monetary limits, and available backend capabilities. Include cached-feature availability, local runtime readiness, rate-limit state, and configured task profile when known. Unknown estimates stay unknown.

A profile contains task/domain, model revisions, hardware class, language/length bands, measured recall/cost/latency curves, timestamp, and confidence/sample-size notes. Start with explicit conservative heuristics labeled uncalibrated. Do not encode “500 candidates means listwise” as a universal law or infer quality from model price alone.

### 13.3 Selection procedure

1. Validate semantics and constraints before optimizing. Eliminate plans that cannot support required metric types, full-order output, provider privacy policy, context sizes, or strict budgets.
2. Include simple feasible plans: lexical-only, embedding-only, cross-encoder pointwise, Jev independent pointwise, shared-context listwise, and explicit small-pool comparison modes. No backend call is mandatory.
3. For large pools and small k, consider a bounded shortlist stage. Where lexical mismatch is likely, use parallel lexical/dense ranks and RRF over their union rather than compulsory lexical pruning. If no measured recall profile exists, label pruning approximate and expose its limits.
4. Consider one expensive judgment stage on survivors. Pairwise/tournament requires an explicit policy/profile and sufficient budget; it is not the default expensive choice. Avoid comparing every pair merely because a “quality” mode was selected.
5. Add diversity or dependency-aware selection only when configured. Request a larger preselection pool when k and available evidence justify it; its size is a visible policy parameter.
6. Estimate total work including cold load, queueing, cache misses, retries allowance, and selection. Prefer a Pareto-feasible plan for the chosen policy. Without calibrated quality predictions, use deterministic preference rules and report that quality was not estimated numerically.
7. If no plan satisfies hard limits, fail before paid work or use a caller-approved deterministic fallback. Do not claim a latency target is feasible when the adapter has no supporting estimate.

Quality modes are bounded preferences, not promises: `fast` prioritizes ready cheap/local stages; `balanced` permits a small model stage; `quality` permits broader candidate retention and more expensive configured judgments; `offline` prohibits network calls and downloads. All modes respect explicit constraints. A mode never grants permission to send private candidates to a backend that was not configured for that trust scope.

### 13.4 Requested example and its tradeoffs

An explicitly configured cascade may produce:

```text
500 authorized candidates
  → candidate-pool BM25: keep 100
  → embedding similarity: keep 30
  → Jev shared-context listwise judgments: keep 10
  → MMR: select 5
```

These are **illustrative configured survivor counts, not measured thresholds or a universal default**. The plan must confirm that 30 candidate projections and questions fit the selected Jev context, that required utility mapping for MMR is defined, and that embeddings are compatible/reusable. Earlier BM25/embedding pruning makes the overall result approximate relative to scoring all 500 with the final objective. Lexical misses cannot be recovered later. The evaluation alternative is lexical and dense branches over all 500, then RRF union → Jev → MMR. For tiny candidates with a ready cheap backend, fewer stages may be better.

An explicit pipeline represents this with `RankStage` survivor limits of 100, 30, and 10, followed by `SelectionStage(MMR(...), limit=5)`. A rank stage owns its metric weights, strategy, and normalization. A fusion stage contains a bounded tuple of independent rank branches over its input and one fusion policy. A selection stage contains one selection policy. This shallow structure supports the example and hybrid union without a user-programmable DAG, loops, remote tasks, or general workflow engine.

## 14. Security and structured-output reliability

Trust boundaries are caller configuration/callback code, candidate content, model/provider output, cache records, and telemetry sinks. Treat candidate text and model output as untrusted data. The library is not a sandbox for arbitrary Python extensions and is not an authorization service.

Enforce eligibility before projection/provider/cache handling; separate private policy metadata from model-visible fields; isolate tenant/trust scopes in batching and caches. Provider allowlists and data-handling decisions are application configuration. Backend URLs come from trusted configuration and are not inferred from candidate URLs, MCP descriptions, or model output. No network fetches of candidate links, automatic tool invocation, database access, or code execution occur during ranking.

Prompts label candidate fields as data and explicitly instruct models to ignore embedded instructions. Opaque IDs and constrained answer schemas reduce output surface. Neither technique proves prompt-injection resistance: malicious candidates can still manipulate judgments or peers in shared context. Offer isolated pointwise mode for stronger context separation, input-size caps, redacted projections, and adversarial evaluations. Do not advertise a sanitizer as making arbitrary content safe.

Validate raw/model/cache output for exact expected IDs, no extra/duplicate/unknown candidates, correct primitive types, finite values, distribution bounds and sums within a documented numerical tolerance, legal cardinality, and complete permutations where promised. Bound response bytes and parsing depth. Missing answers, refusals, and invalid schemas are explicit outcomes. “Valid JSON” alone is never treated as a valid ranking.

Model-generated explanations are deferred. If later enabled, label them as generated, potentially unfaithful text; they are not proof of correctness or internal reasoning. Trace payload capture is off by default; opt-in capture needs application redaction, retention, access controls, and explicit size limits. Exceptions should not include raw projections or SDK response bodies.

## 15. Observability and evaluation architecture

### 15.1 Hooks, statistics, and reports

Emit events for request/plan/stage start and completion, cache lookup, batch queued/dispatched, attempt/retry, validation failure, budget reservation/reconciliation, fallback, cancellation, and final selection. Fields include request/stage/backend IDs, safe model revision, counts, duration, score kind, retry reason, cache outcome, token/cost estimates vs observed values, and remaining budget. Raw candidate IDs, queries, prompts, distributions, and content are not default trace attributes.

`Statistics` separates preparation, queue, model, cache, normalization, selection, and end-to-end duration; records attempts, batch sizes, cache hits/misses, concurrency high-water mark, truncated/pruned/failed counts, usage completeness, and known/estimated/unknown cost. Concurrent stage durations need not sum to end-to-end latency. Do not subtract unobserved provider time or present missing usage as zero.

The execution report retains the intended plan plus actual stage coverage and deviations, making it possible to explain why the engine pruned, changed strategy, or returned a fallback. Default provenance is compact stage-level data, with bounded per-candidate details on request. Observers are isolated from ranking failure by default; dropped events and observer exceptions appear as diagnostics. A testing-only strict observer mode can surface integration errors. OpenTelemetry translation is an optional consumer, not an execution dependency.

### 15.2 Evaluation layers

1. **Contract and property checks:** candidate identity/duplicates, stable ties, k edge cases, exact output mapping, score normalization, cyclic comparisons, cache invalidation, cancellation, deadline/budget reservations, fallback constraints, and no-network offline behavior.
2. **Adapter conformance:** recorded/synthetic fixtures for each backend request variant, malformed output, refused request, mismatched usage, context limits, unsupported settings, retries disabled, and lifecycle ownership. Live integration runs are explicit and budget-capped; default CI does not spend provider credits.
3. **Ranking evaluation:** fixed pools with independent relevance labels, nDCG@k for graded labels, MRR for first useful match, Recall@k, Precision@k, MAP where applicable, pair preference accuracy, and shortlist recall/oracle ceiling after every pruning stage. State unjudged-item policy; distinguish missing labels from negative labels. Never silently insert relevant items into the candidate pool.
4. **Set/domain outcomes:** evidence coverage and answer grounding for RAG; correct entity match and abstention; tool/agent permission-valid selection and downstream task success in a separate harness; valid SQL schema/join coverage; graph connectivity/dependency coverage; memory usefulness/staleness/conflict; redundancy/diversity and business effects. Any downstream execution is harness/application work, not library behavior.
5. **Calibration and robustness:** Brier/log loss for genuine binary probabilities, ordinal distribution evaluation for Score, risk/coverage for abstention, query/candidate order sensitivity, length/language/domain slices, injection attacks, adversarial metadata, duplicate flooding, and noisy pair cycles. Model confidence does not substitute for calibration measurement.
6. **Operational measurements:** cold/warm, cached/uncached, local/remote, p50/p95/p99 latency and throughput under concurrency, CPU/RAM/GPU, all attempts, cost/usage completeness, and queue behavior. Compare quality/cost/latency Pareto curves, not one aggregate winner.

Evaluation records separate data manifests, immutable candidate/projection snapshots, qrels, model/prompt/calibration versions, seeds, environment/hardware, pricing version, cache policy, and output reports. Dataset loaders are optional, rights/retention are explicit, and private candidate text is not automatically exported. Splits are by query/entity/time as appropriate to prevent leakage; tune planning thresholds on development data and report held-out results with uncertainty. An LLM judge is auxiliary evidence and must be calibrated against human/task labels, especially when related to the ranking model.

### 15.3 Benchmarks required before default tuning

Measure Jev pointwise vs shared-state vs Score/Choice alternatives; shared-state fan-out cost and positional effects; lexical/dense union recall vs lexical-first cascades; local cross-encoder baselines; pairwise/tournament benefit per comparison; candidate projection length; MMR relevance loss vs coverage gain; model cache hit rates/alias drift; thread/GPU batching behavior; and planner mistakes under strict/estimated budgets. No benchmark numbers have been generated for this plan. Initial survivor counts, half-lives, RRF constants, and policy thresholds are configuration hypotheses.

## 16. Expected module structure

This is the target layout, not files to create in the design task. Create modules as their milestone needs them; do not scaffold empty packages for future ideas.

```text
pyproject.toml
README.md
LICENSE                         # select deliberately before publication
IMPLEMENTATION_PLAN.md
docs/
  research.md
  api.md
  score-semantics.md
  security.md
  evaluation.md
src/jev_rankkit/
  __init__.py                   # small stable facade exports
  py.typed
  api.py                       # Reranker; sync convenience; session scopes
  types.py                     # results, context, scores, plan/report values
  config.py                    # immutable configuration, budgets/policies
  candidates.py                # explicit projection/metadata adapters
  prompts.py                   # immutable prompt/rubric specifications
  errors.py
  auto.py                      # AutoReranker facade and policy selection
  pipeline.py                  # shallow stage composition
  cache.py                     # CacheBackend, MemoryCache, records/policy
  observability.py             # events and observer contract
  metrics/
    __init__.py                # Metric, configured metric constructors
    base.py
    lexical.py
    semantic.py
    metadata.py
    model.py
    callbacks.py
    normalization.py
  strategies/
    __init__.py
    base.py
    pointwise.py
    pairwise.py
    listwise.py
    tournament.py
    hierarchical.py
  selection/
    __init__.py
    fusion.py                  # RRF and rank-source validation
    diversity.py               # MMR
    constraints.py             # bounded quotas/dependency selection
  backends/
    __init__.py                # protocol exports, no SDK imports
    base.py                    # requests, responses, capabilities/sessions
    jev.py                     # optional typesafe-sdk import
    sentence_transformers.py   # optional embeddings and cross-encoder
    structured_llm.py          # one explicit optional provider initially
  _runtime/
    prepare.py
    executor.py
    batching.py
    budget.py
    cache_keys.py
    validation.py
    planner.py                 # private rules/profile matching
  evaluation/
    __init__.py
    datasets.py                # manifests/loaders behind eval extra
    metrics.py
    runner.py                  # explicit offline/live harness
    reports.py
tests/
  unit/
  contracts/
  typing/                      # mypy/pyright generic API examples
  integration/                 # opt-in local/provider contracts
benchmarks/
  manifests/
  scenarios/
examples/
  basic.py
  custom_objects.py
  hybrid.py
  auto.py
```

No domain subpackage per object type, global plugin manager, distributed scheduler, persistence ORM, or separate sync backend hierarchy. The layout separates responsibilities without requiring a public class for every internal function.

## 17. Implementation sequence and acceptance gates

The following is future work requiring a subsequent implementation task. This architecture task stops at this document.

### Milestone 1 — prove the core contract offline

Finalize name/license, package metadata, immutable types, generic method inference, candidate preparation, deterministic lexical/metadata/callback metrics, normalization, stable ordering, strict failures, and async/sync lifecycle. Add a bounded in-memory cache and synthetic fake backend for contracts. Acceptance: custom dataclasses/dicts/strings return exact original objects; invalid inputs cause no model/cache operations; no-op behavior is correct; async cancellation and active-loop sync errors are tested; lightweight import has no provider/runtime dependencies.

### Milestone 2 — Jev and controlled execution

Add adapter capabilities, pointwise and shared-context listwise Jev paths, execution ledger, bounded batches/concurrency, retry/timeout ownership, exact output validation, tracing/statistics, and explicit checkpoint fallback. Acceptance: adapter fixture conformance passes; concurrent reservations do not overshoot declared bounds; ambiguous timeout accounting remains visible; optional live tests are budgeted and record resolved revisions. Evaluate SDK compatibility before committing to its transport.

### Milestone 3 — composable ranking and local baselines

Add embedding/cross-encoder adapters, weighted hybrid scoring, RRF, explicit cascades, MMR, and shallow fusion branches. Acceptance: cache keys distinguish every context-dependent input; incomparable scores cannot be silently mixed; pipeline report accounts for all pruning; local baselines and candidate recall are measured on fixed pools. This is the useful initial release target; broad strategy menus are not required to ship an unreliable first version.

### Milestone 4 — advanced strategies and domain selection

Add exact bounded pairwise, explicit tournament variants, hierarchy, dependency/coverage policies, and one generic structured-output LLM adapter. Acceptance: cycle/position/bracket tests, full-order capability validation, impossible dependency constraints, rank-only semantics, and adversarial output tests pass. Benchmark whether each advanced strategy adds value before promoting it from experimental.

### Milestone 5 — AutoReranker with evidence

Implement transparent rule-based planning, dry-run prepared plans, constrained execution/replanning, and measured profiles. Acceptance: every chosen stage has an inspectable reason; no hidden provider call/download occurs in planning; unknown estimates remain unknown; full-order requests never enter pruning plans; model/policy revision changes invalidate prepared plans; Pareto comparisons include stage recall and cache states. Ship heuristic mode clearly labeled before claiming calibrated automatic optimization.

### Milestone 6 — compatibility and release hardening

Document stable protocols and deprecation policy; run typing examples and minimum/latest-supported dependency matrices; verify optional extras in clean environments; review data leakage/logging/security assumptions; publish reproducible benchmark methodology and measured limitations. Redis, additional providers, and richer integrations are demand-driven follow-up work, not mandatory for version one.

Do not expand unit tests merely to mirror dataclass implementation. Prioritize invariants, boundary failures, concurrency behavior, and end-to-end object/score mapping. This design task has not executed production tests because no implementation exists.

## 18. What not to build

- A new vector database, crawler, retrieval/indexing framework, model-serving runtime, or general distributed DAG executor.
- An agent that selects and executes tools while ranking them, or database/graph adapters that access extra data implicitly.
- A wrapper that exposes every backend as though its scores were interchangeable relevance probabilities.
- A mandatory LLM stage, automatic expensive all-pairs comparison, or a generative planner deciding how to spend money.
- An inheritance hierarchy for documents/entities/tools/products or automatic introspection of arbitrary Python objects.
- A new prompt language, regex-based universal JSON repairer, or custom transport stack before evaluating maintained clients.
- Automatic normalization/threshold calibration from the current candidate pool presented as universal relevance.
- A claim that typed output defeats prompt injection or that client timeouts guarantee unbilled remote work.
- Persistent full-object/result caches, pickle serialization, global mutable registries, hidden environment loading, or implicit credentials/provider selection.
- Large explanation payloads, chain-of-thought collection, learned strategy selection, or exact graph/set optimization in the first release.

## 19. Self-critique and design revision record

This section records issues found in review and the decisions applied to the architecture above. Remaining risks are explicit rather than treated as solved by interfaces.

### 19.1 Unnecessary abstractions

**Risk:** separate pipelines, ranking strategies, metrics, selection, backends, sessions, and planning could become a framework. **Revision:** use one executor, a shallow fixed stage structure, six root exports, candidate callbacks rather than domain classes, and explicit built-in registration. Session machinery stays behind the backend protocol; no public workflow DAG, plugin discovery, or domain base classes. Ship pointwise/listwise and useful deterministic composition before advanced strategy protocols stabilize. The review also promoted `RankingOutcome` and all extension-signature types to the public contract; custom strategies must not depend on private DTOs.

**Remaining risk:** a third-party strategy still needs a substantive extension contract. Keep it experimental until an external implementation validates the boundary; do not promise compatibility for private execution machinery.

### 19.2 Expensive hot paths

**Risk:** serializing objects repeatedly, task-per-pair fan-out, quadratic MMR, cold model loads, and unnecessary stages defeat a fast model. **Revision:** one frozen projection, bounded queues, context-aware packing, incremental MMR, cached embeddings, resource scopes, and plan estimates including startup. Dense/lexical union is an alternative to mandatory cascades; Auto can select a single cheap stage.

**Remaining risk:** projection/tokenization and trusted callbacks may dominate small requests. Measure their time separately; do not move every tiny function through a worker pool and add more overhead.

### 19.3 Fragile model assumptions

**Risk:** treating Jev as an arbitrary JSON LLM, shared-state scoring as independent batching, confidence as calibration, or tournaments as exact top-k. **Revision:** explicit request/capability variants, score kinds, semantic batch boundaries, rank-only output, and approximate coverage. Model limitations are adapter data, not core constants. Self-reported confidence cannot choose an escalation threshold without validation. The tournament is now a specific winner-tree extraction algorithm, with advanced variants deferred instead of left as an undefined promise.

**Remaining risk:** model drift and order sensitivity still change judgments; pin revisions when possible and rerun held-out evaluation. A well-typed interface cannot guarantee ranking quality.

### 19.4 Confusing APIs

**Risk:** a convenient facade that silently downloads models, enables paid calls, invents metadata, or selects a different algorithm. **Revision:** constructor does no work; explicit namespaced backends; named metrics require inputs; pointwise is stable; approximation is opt-in Auto/pipeline behavior. Group operational options in config/context and strategy-specific options in strategy instances. Keep `.results` as the sole canonical collection. Default relevance now matches backend capabilities, and named recency has a visible starter half-life. Auto cannot drop explicit metric objectives merely to meet a budget.

**Remaining risk:** “listwise” still covers two established meanings. Every plan must show `shared_context` versus `permutation`, and documentation/examples must make this visible.

### 19.5 Type-safety problems

**Risk:** constructor inference degrades to `Any`, callback/item types diverge, and scores returned by models attach to the wrong object. **Revision:** non-generic facade with generic call method, typed per-call metrics/adapters, ID-keyed evaluation records, and one validated association layer. Generic pipeline plans preserve `T`; caches and backends never reconstruct it. Public typing examples must run through mypy and pyright, including mixed/union candidates and invalid callback types. The live-object versus frozen-projection distinction is now explicit, and final eligibility changes fail closed rather than leaving dangling dependencies.

**Remaining risk:** Python callbacks, mappings, and user-supplied `Any` can defeat static checking. Runtime boundary checks remain required. TypeScript-like compile-time assurances are not claimed for arbitrary Python objects.

### 19.6 Async problems

**Risk:** nested event loops, cross-loop HTTP pools, blocking callbacks, retry storms, cancellation-triggered fallback, and shared-future cancellation. **Revision:** one async engine; fresh sync scope; explicit loop-bound async pooling; bounded workers and shared rate controls; one retry owner; cancellation propagation; defer cross-operation in-flight sharing. Borrowed clients have explicit lifecycle ownership. Unscoped cache/session lifetime and rejection of sync calls on an active async scope are now specified.

**Remaining risk:** third-party blocking/threaded inference may continue after cancellation. Surface that limitation, stop consuming its result, and constrain admission; Python cannot safely kill an arbitrary thread.

### 19.7 Cache invalidation issues

**Risk:** using only candidate IDs, omitting shared peers, mutable aliases, timestamp decay, captured callback state, or random occurrence IDs in keys. **Revision:** canonical payload fingerprints and full semantic context, deterministic wire ordinals, namespace/policy revisions, explicit model pin/alias TTL policy, no persistent impure callbacks, and recomputed final selection/recency. Cache features/judgments rather than final object responses.

**Remaining risk:** callers must accurately version external data and authorization policy. TTL reduces stale lifetime but does not implement deletion rights or revocation by itself; applications need reliable invalidation and provider controls.

### 19.8 Scoring inconsistencies

**Risk:** combining logits, ordinal expectations, BM25, pair wins, RRF, and MMR just because each is numeric. **Revision:** raw kind, normalized utility, confidence, ordering score, and selection score are separate. Explicit normalization precedes weighted metrics; missingness policies operate coherently across the stage. Rank-only outputs use fusion, and MMR needs a declared relevance utility. Fallback checkpoints preserve their original scale.

**Remaining risk:** even two `[0,1]` utilities can have different real-world meaning. Rubric/normalizer calibration and weight tuning require task evidence; boundedness is necessary for this API, not sufficient for statistical comparability.

### 19.9 Budget and planning honesty

**Risk:** a planner promising exact cost/latency from token estimates, or an exported dry-run silently reranking mutated data later. **Revision:** distinguish strict bounds from estimates and unknowns; share one ledger across retries/fallback; retain ambiguous spend; prepare immutable local snapshots and separate safe plan descriptions from executable plans. Heuristic quality modes are policies, not numerical guarantees.

**Remaining risk:** strict bounds may exclude useful providers with poor usage APIs. This is an explicit tradeoff: estimated mode is available, but the product must not rename it “hard budget enforcement.”

## 20. Decisions still requiring evidence or release-time verification

Verify final PyPI/trademark availability; select license; confirm supported SDK retry/session behavior and pinned model identifiers; validate minimum dependency versions; choose the first generative provider only if justified by a real use case. Measure default lexical tokenization across languages, relevance rubric/normalization, shortlist counts, batching limits, cache TTLs, and Auto quality profiles. Set acceptance targets from intended applications and baseline measurements, not invented benchmark numbers.

The architecture is ready to guide staged implementation, but none of the proposed APIs exists yet. This task changes documentation only and deliberately stops before production implementation.
