# Universal reranking with Jev and TypeSafe AI: research

Research date: **2026-09-22**. Status: **research and architecture direction only**.

## 1. Scope, evidence, and recommendation

Build a small, typed ranking engine for arbitrary candidate objects, with Jev as a first-class judgment backend and deterministic Python policies controlling eligibility, aggregation, ordering, selection, and budgets. A useful library must also work without Jev: lexical ranking, existing retrieval scores, fusion, local cross-encoders, metadata rules, and diversity selection all have legitimate roles.

The opportunity is **reliable selection under constraints across different object types**, rather than another uniform interface over model calls. Quality depends on candidate recall, representations, objectives, and selection policy as much as the final model.

This document covers all 25 requested research topics. It does not choose a package name, finalize a public API, create an implementation plan, or implement production code. The later steps in the supplied build workflow are outside this task.

Evidence is separated as follows:

- **Verified interface/source facts:** observed in official documentation, package metadata, or inspected release source.
- **Published evidence:** papers and author/vendor examples, without treating their results as measurements of this project.
- **Recommendations/inferences:** architectural conclusions drawn here; these remain proposals.
- **Unmeasured:** all performance, quality, calibration, robustness, and cost outcomes for the proposed library. No inference requests or benchmarks were run, and no upstream test suite was executed.

The starting workspace contained no project files and was not a Git repository. Research used public primary sources. The existing wheel was downloaded to a temporary directory, hash-checked, and read as source without installing or executing it. Only this research document is added to the workspace.

## 2. The existing `jev-reranker` package

### 2.1 Identity and reproducible inspection

The supplied piwheels page describes the PyPI distribution **`jev-reranker`**, maintained at **`hotchpotch/jev-reranker`**. On the research date, PyPI and piwheels listed version **0.1.2**, released on **2026-09-21**. Its metadata requires Python **3.11+**, declares the **MIT** license, and has `httpx` and `python-dotenv` as base dependencies. Tokenizer, Sentence Transformers, and evaluation dependencies are optional. [piwheels](https://www.piwheels.org/project/jev-reranker/), [PyPI release metadata](https://pypi.org/pypi/jev-reranker/0.1.2/json).

Inspected wheel: `jev_reranker-0.1.2-py3-none-any.whl`.

SHA-256: `55cb9625a3269ca4c61be1e6dea0a0b0a1954bbab9365a7e28ea0502370e3924`.

Repository snapshot: `d58594b393b29b7ee6398cc9337dc5e9d6c6691e`. Its package version is 0.1.2; its `reranker.py` matched the wheel after line-ending normalization. Other files were read from the wheel; this was not a byte-for-byte audit of the entire repository. [Pinned package metadata](https://github.com/hotchpotch/jev-reranker/blob/d58594b393b29b7ee6398cc9337dc5e9d6c6691e/pyproject.toml).

Do not confuse it with **`uspraveen/Jev-Reranker`**, a separate project focused on agent memory selection using relevance, utility, supersession, and conflict judgments. The latter is adjacent prior art, not the upstream repository identified by the supplied package link. [Separate memory project](https://github.com/uspraveen/Jev-Reranker).

### 2.2 Existing behavior and strengths

The public package accepts a query and a sequence of document strings. It returns dictionaries containing original document indices, scores, and optionally text and execution details. It provides ordinary reranking and a separate evidence-usefulness prompt with threshold filtering. Stable ties preserve input order; duplicate texts remain separate candidates; empty inputs and `top_k=0` avoid inference. Filtering and output `top_k` normally happen after scoring. [Public API](https://pypi.org/project/jev-reranker/).

Its three modes deserve precise descriptions:

- **Pointwise:** one query/document state and one Noul judgment per request.
- **Listwise:** several documents share a state, with one Noul question per document; Python sorts the returned values. It does not ask Jev to generate a permutation.
- **Pairwise:** each unordered document pair receives two directional Noul questions in one request. The implementation combines them as `(forward + 1 - reverse) / 2`, then averages wins over opponents. That is a relative preference score, not absolute relevance. For a singleton it returns a neutral pairwise score.

These are source observations. The ordinary model endpoint is `/v1/systemone`; this package constructs HTTP requests directly rather than depending on `typesafe-sdk`. [Scoring and payload source](https://github.com/hotchpotch/jev-reranker/blob/d58594b393b29b7ee6398cc9337dc5e9d6c6691e/src/jev_reranker/reranker.py).

The request layer checks exact answer-key coverage, expected Noul answer types, numeric bounds, and usage metadata. It has bounded transport/status retries, jittered backoff, `Retry-After` handling, and distinct context-limit and validation errors. These are existing features, not proposed differentiators. [HTTP and validation source](https://github.com/hotchpotch/jev-reranker/blob/d58594b393b29b7ee6398cc9337dc5e9d6c6691e/src/jev_reranker/_client.py).

Listwise splitting balances estimated document lengths, uses deterministic ordering within partitions, and recursively reduces groups on context rejection. Each resulting group is scored separately. A document that cannot fit with its query raises a context error. [Partitioning source](https://github.com/hotchpotch/jev-reranker/blob/d58594b393b29b7ee6398cc9337dc5e9d6c6691e/src/jev_reranker/_ranking.py).

The runtime uses asynchronous HTTP, bounded concurrency, and per-operation clients with cleanup. A caller can supply an async client for connection reuse; that client has caller-owned lifecycle and event-loop restrictions. Synchronous calls use an async runner and reject use inside a running event loop. [Runtime source](https://github.com/hotchpotch/jev-reranker/blob/d58594b393b29b7ee6398cc9337dc5e9d6c6691e/src/jev_reranker/_runtime.py).

Its repository also includes evaluation tooling comparing Jev with a local cross-encoder on the same candidate inputs, recording dataset/model configuration and filtering effects. Crucially, fixed-count selection guarantees inclusion of qrels-positive documents; the default full-pool mode preserves the original retrieved pool. The former measures controlled reranking and must not be presented as ordinary retrieval performance. The evaluation guide itself reports no benchmark results. [Evaluation methodology](https://github.com/hotchpotch/jev-reranker/blob/d58594b393b29b7ee6398cc9337dc5e9d6c6691e/docs/eval.md).

### 2.3 Limits relative to the proposed scope

The inspected package is a credible focused solution for string-based RAG reranking. Its scope does not establish a generic object/result contract, provider-independent strategy engine, shared pipeline budget ledger, RRF/MMR composition, arbitrary metrics, or domain-aware set selection.

Source-informed implications for this project:

- Mapping indices back to objects is possible today; preserving `T` and original identity throughout multiple stages would make it a first-class guarantee.
- Pairwise work grows quadratically. Output `top_k` alone does not make scoring cheaper.
- Independently scored partitions can introduce context-dependent score differences. Splitting to fit a request is not a guarantee of equivalent ranking.
- Optional diagnostics can contain full candidate content. Credential redaction is not general data minimization.
- Typed package annotations, including `py.typed`, do not substitute for a generic typed result API.
- Optional Sentence Transformers evaluation support should not be confused with a general backend abstraction in the public reranker.

These observations are not claims that every absent feature is a defect. Keep the simpler package when the requirement is just Jev scoring over document strings. A new library is justified only if it solves composition, universal objects, score semantics, and operational control substantially better.

## 3. Jev and TypeSafe AI APIs

### 3.1 Intended architecture

Here, **TypeSafe AI** means the vendor and its System One API, not merely Python type hints or a generic structured-output framework. Jev consumes supplied state and bounded questions, returning typed judgments instead of generated prose. The vendor describes training with Reinforcement Learning for Calibrated Decisions (RLCD). This is a product/training description, not evidence that this research has independently verified its internal neural architecture or calibration. [System One](https://docs.typesafe.ai/concepts/system-one), [launch announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

The intended software pattern is: prepare relevant state, ask narrow judgments, and combine outputs in ordinary code. Independent questions can be submitted together against shared state; a judgment needing a previous answer or newly retrieved evidence needs another stage. The API does not make one question's answer available to another question in the same call. Question IDs correlate responses but are not semantic instructions to the model. [Primitives](https://docs.typesafe.ai/primitives).

**Architectural inference:** Jev fits a feature/judgment service inside a ranking engine. It should not own scheduling, arithmetic, permission checks, or the application workflow. Do not infer an encoder topology, parameter count, tokenizer equivalence, open weights, or self-hosting capability from the System One label; those were not established by the reviewed sources.

### 3.2 Primitive semantics

**Noul:** a probability-like value in `[0,1]` for a defined yes/no proposition. There is no separate `confidence` field. A value near the middle expresses uncertainty about that proposition; it is not automatically a medium degree of relevance. Use a precise condition such as whether a passage supplies evidence needed for the query. [Noul](https://docs.typesafe.ai/primitives/noul).

**Score:** an expected index over ordered, descriptively anchored levels. For `L` levels indexed from zero, the raw score is `sum(i * p_i)` and lies in `[0,L-1]`. The API accepts two to ten levels. `score / (L-1)` is a possible bounded utility mapping, not a conversion to probability of relevance. Preserve the original distribution and rubric; equal expected scores can hide very different uncertainty. [Score](https://docs.typesafe.ai/primitives/score).

**Choice:** selection and probabilities over a supplied finite option set, currently limited to 255 options. This supports routing, pair comparisons, or selecting among a shortlist. Its probabilities are relative to the offered alternatives; adding another candidate changes the decision space. Include an explicit no-match option when appropriate. A Choice distribution is not a vector of independent relevance probabilities. [Choice](https://docs.typesafe.ai/primitives/choice).

Choice and Score include a distribution-derived `confidence`. Do not treat that statistic as the empirically measured probability that the selected answer is correct, and do not assume it equals maximum option probability. Calibration must be checked on the actual ranking task, including language and domain slices. [Confidence](https://docs.typesafe.ai/confidence).

**Recommendation:** separate raw model output, normalized utility, confidence, and final ranking score. An API that collapses all four to one unlabelled float would discard a major advantage of Jev.

### 3.3 Verified API and deployment facts

The HTTP interface is `POST https://api.typesafe.ai/v1/systemone` with bearer authentication and `model`, `state`, and `questions`. Responses contain `model`, keyed `answers`, and input/output token usage. State can be a string, object, or array. The API documents rate-limit and overload errors, including HTTP 429 and 529. [HTTP reference](https://docs.typesafe.ai/api).

As of the research date, the model page lists `jev-1.13.0`; `jev-latest` and `jev-preview` both resolve to it. It advertises **64k tokens for the total request** and **32k for state plus the longest question**, text-only input, **$0.042 per million input tokens**, and free output tokens. Listed limits are **250,000 tokens/second** and **1,200 requests/minute**, explicitly subject to change. These are vendor specifications and list pricing, not measured latency, throughput, or invoiced cost. English is the strongest documented language; other languages need workload-specific evaluation. [Dated model specifications](https://docs.typesafe.ai/models).

Recommendation: record requested and resolved model IDs, pin versions for evaluation, and keep pricing and limits configurable with a verification date. Do not assume an API model listing contains complete capability or pricing information. Independently budget both context inequalities, with tokenizer-estimation headroom.

### 3.4 Official Python SDK and type safety

The official package is **`typesafe-sdk`**, imported as `typesafe_sdk`. On the research date PyPI listed **0.7.1**, Python **3.10+**, with dependencies including `httpx2`, Pydantic, and Tenacity. It exposes synchronous `TypeSafeClient` and asynchronous `AsyncTypeSafeClient`. Prefer evaluating this maintained client as the Jev adapter's transport rather than reproducing all HTTP behavior. Keep it optional for users running deterministic or local backends. [SDK guide](https://docs.typesafe.ai/sdk/python), [SDK package metadata](https://pypi.org/pypi/typesafe-sdk/0.7.1/json).

The SDK supports typed questions/answers and custom Pydantic response models. Those response models validate API results; they do not expand Jev into an arbitrary JSON/prose generator. Its compatibility behavior skips unknown answer kinds with a warning and ignores unknown extra response fields. A ranking adapter therefore still needs exact requested-answer coverage checks. SDK debug logging includes unredacted bodies. Also distinguish the SDK's `TYPESAFE_BASE_URL` API root from the existing reranker's full endpoint setting. [SDK usage and compatibility](https://docs.typesafe.ai/sdk/python/usage).

The SDK provides retry controls, including a total retry budget. Choose one layer to own retry policy so nested retries do not multiply attempts. A pipeline-wide deadline must additionally cover queueing, multiple stages, local work, and fallback. [RetryPolicy](https://docs.typesafe.ai/sdk/python/api/retries), [async client](https://docs.typesafe.ai/sdk/python/api/clients/async).

No reviewed Jev contract establishes generated explanations, a general reasoning-effort knob, or arbitrary permutation schemas. Such features must be capability-gated for other providers, not silently accepted as effective Jev options.

### 3.5 Strengths, limitations, and role allocation

Jev's promising properties are bounded typed outputs, multiple judgments over shared state, explicit rubrics, and accessible distributions. The published composite-scoring pattern supports separate judgments followed by deterministic weighting. This is especially relevant when ranking needs more than generic query similarity. [Composite scoring](https://docs.typesafe.ai/patterns/composite-scoring).

The vendor's Jev 1.13 limitations document reports weaknesses with numeric precision, date comparison, indirection, distracting state, adversarial content, and contradictory criteria. It also warns that separately asked questions need not obey expected complement identities. It does not support useful free-text generation. [Known limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

**Recommended allocation:**

- **Deterministic code:** permissions, inventory, schema compatibility, numeric limits, expiry, recency calculations, exact IDs, graph connectivity, score formulas, tie-breaking, budgets, and deduplication policy.
- **Embeddings/cross-encoders:** semantic similarity and standard relevance where suitable trained models already exist; validate on the domain.
- **Jev:** narrow evidence usefulness, semantic fit, possible contradiction, entity agreement, or rubric-based utility after candidate reduction.
- **Generative LLMs:** difficult contextual comparisons, query decomposition, or optional user-facing explanations when justified by measured value. Explanations are additional model outputs, not verified reasoning traces.

No model should be the sole authority for access control or an exact business constraint.

## 4. Existing Python ecosystem

### 4.1 Libraries and services worth building alongside

**Sentence Transformers:** a strong baseline and optional adapter for bi-encoders and CrossEncoder inference. Joint query/candidate attention is distinct from comparing separately computed embeddings. Its model interface exposes batching, activation choices, and backend options. Scores may be logits or transformed outputs depending on the model/configuration. [Retrieve and rerank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html), [CrossEncoder API](https://www.sbert.net/docs/package_reference/cross_encoder/model.html).

**Answer.AI `rerankers`:** already offers a low-dependency common API across cross-encoders, T5 models, LLM methods, hosted services, FlashRank, and ColBERT-style scoring, with metadata support. A unified model wrapper alone would duplicate existing work. Consider adapter reuse where score semantics and lifecycle fit. [Repository](https://github.com/AnswerDotAI/rerankers).

**FlashRank:** lightweight local inference and ranking integrations, including compact cross-encoder paths and listwise support. It is a candidate CPU baseline for deployment environments where a network API or large model is undesirable; benchmark actual hardware and model configurations. [Repository](https://github.com/PrithivirajDamodaran/FlashRank).

**FlagEmbedding/BGE and Qwen3-Reranker:** model ecosystems worth testing as semantic baselines. Qwen's reranking examples use yes/no token logits; an LLM-derived architecture does not imply long generated ranking responses or chain-of-thought. BGE includes several reranker families. Model weights, licenses, runtime requirements, and truncation rules must be checked per model. [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding), [Qwen3 embedding and reranking](https://github.com/QwenLM/Qwen3-Embedding).

**RankLLM:** a research-oriented toolkit spanning pointwise, pairwise, and especially listwise LLM reranking, with hosted/local paths, evaluation support, and efficient inference variants. Reuse its research and comparisons; avoid rebuilding a model-serving stack within this library. [Repository](https://github.com/castorini/rank_llm).

**Cohere and Voyage:** hosted reranking APIs provide useful managed-service baselines. They have provider-specific input limits, truncation, scores, and billing units. A backend contract should expose such capabilities instead of pretending all providers are interchangeable. [Cohere rerank API](https://docs.cohere.com/reference/rerank), [Voyage rerankers](https://docs.voyageai.com/docs/reranker).

**BM25S, Pyserini, and PyTerrier:** reuse lexical retrieval and experimental tooling. BM25S is a Python lexical-search option; Pyserini covers reproducible sparse/dense retrieval; PyTerrier provides composable retrieval experiments. They demonstrate that retrieval and pipeline composition are established capabilities. [BM25S](https://github.com/xhluca/bm25s), [Pyserini](https://github.com/castorini/pyserini), [PyTerrier](https://github.com/terrier-org/pyterrier).

**LangChain and LlamaIndex:** integration surfaces, not dependencies for the core. Their retriever and node-postprocessing abstractions are useful entry points, but generic results should preserve domain objects without inheriting either framework's document type. [LangChain retrievers](https://docs.langchain.com/oss/python/integrations/retrievers), [LlamaIndex node postprocessors](https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/).

### 4.2 Opportunity beyond a wrapper

The proposed combination is valuable if it provides:

1. Original-object preservation and typed results across every stage.
2. Explicit score semantics and provenance across heterogeneous rankers.
3. Jev-specific use of native primitives without forcing every backend into that API.
4. A common budget/deadline ledger spanning attempts, stages, and fallback.
5. Separation of per-item relevance from final set utility and hard constraints.
6. Inspectable execution plans and domain-specific evaluation adapters.

These are proposed product differentiators, not claims that no existing library offers any individual feature. Benchmark and integration evidence must establish whether their combination merits a separate project.

## 5. Ranking approaches and tradeoffs

Let `n` be candidate count, `k` the requested output count, `b` a batch/group size, and `m` the number of judgment dimensions. Complexity below describes algorithmic work, not measured speed. HTTP request count, model evaluations, tokens, and sequential depth are different quantities.

### 5.1 Cross-encoder reranking

A cross-encoder evaluates the query jointly with each candidate representation, allowing token-level interaction. A bi-encoder instead embeds inputs separately, enabling cached candidate vectors and efficient retrieval. A shortlist can then receive more expensive joint scoring. Cross-encoder weights cannot generally be replaced with one reusable vector per document. [Sentence Transformers retrieval architecture](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html).

There are `n` query/candidate evaluations, often packed into local inference batches. Candidate length, padding, device, precision, and model context limit dominate real cost. This is usually a pointwise ranking strategy, even though a query/document input is sometimes called a “pair.” It is not pairwise comparison between two competing candidates.

Recommendation: include at least one compact local cross-encoder in evaluation. It may offer better repeatability, privacy, or unit economics for repeated workloads; none of those imply universal quality superiority. Do not normalize every cross-encoder score with a sigmoid and then call it calibrated probability.

### 5.2 LLM-based reranking

LLMs can judge each item, compare candidates, or produce an ordered list. They can express task-specific relevance instructions, but output generation, repeated context, positional effects, and service variance add operational complexity. RankGPT is a primary reference for prompted listwise reranking and sliding-window approaches. MonoT5 illustrates scoring through relevance-label token probabilities rather than free-form text. [RankGPT](https://arxiv.org/abs/2304.09542), [MonoT5](https://arxiv.org/abs/2003.06713).

Recommendation: classify backends by capabilities and score meaning, not simply “LLM versus non-LLM.” Jev, a decoder-based relevance classifier, and a generative permutation ranker have materially different contracts. A generative LLM should be an optional backend or escalation path when evidence supports it.

### 5.3 Pointwise ranking

Each candidate receives an independent utility/relevance estimate against the query and rubric. It is easy to parallelize, cache per item, and apply absolute thresholds when appropriately calibrated. Work is `O(n)` judgments, or `O(nm)` for multiple dimensions. Where a backend supports independent batches of size `b`, request count can be about `ceil(n/b)` without reducing the number of item judgments; the inspected package's isolated pointwise mode instead sends one request per item.

Limitations include score compression, weak discrimination among similar items, and no explicit accounting for complementary evidence or redundancy. Batched processing does not by itself change the pointwise objective. For Jev, adding all candidates to shared state can change the judgment context even when each question refers to one item; strict isolation and shared-state batching must be distinguished in the public semantics.

Recommendation: use independent scoring as the initial reference behavior. Compare shared-state multi-question scoring as a separately named mode; do not promise equivalence without tests.

### 5.4 Pairwise ranking

Ask which of two candidates better serves the query, optionally allowing a tie or neither. All-pairs ranking requires `n(n-1)/2` comparisons; this count is arithmetic, not a benchmark. Sorting-like schedules can use fewer comparisons but assume a sufficiently stable comparator. LLM comparisons may be position-sensitive and non-transitive. Pairwise Ranking Prompting studies this formulation and several scheduling strategies. [PRP paper](https://arxiv.org/abs/2306.17563).

Recommendations:

- Use pairwise methods on small ambiguous shortlists or near the selection boundary, not as the default for large pools.
- Distinguish measured preference from inferred order; record cycles, ties, and missing comparisons.
- A `Choice(A, B, tie/neither)` offers an explicit decision space. Two opposite Noul questions need not sum to one; symmetrization is a policy, not proof of calibration.
- Reversed comparisons and repeated samples can reduce some bias but increase cost; measure the tradeoff.
- Mean win rate against different opponent sets is not directly comparable. Do not merge tournament scores as absolute relevance.

### 5.5 Listwise ranking

There are at least three contracts to distinguish: generated full permutations; shared-state per-candidate judgments; and selection of one or several winners. A listwise training loss is another concept and does not imply listwise inference.

Full permutations can express relative ordering but require checks for unknown IDs, duplicates, omissions, refusal, and truncation. Context constraints encourage windows or groups. Overlapping windows introduce sequential dependencies and repeat candidates; disjoint groups create cross-group comparability problems. The input order itself can influence output. [RankGPT](https://arxiv.org/abs/2304.09542).

Recommendation: use Jev's supported judgments plus deterministic sorting, not an invented free-form permutation endpoint. For grouped relevance scores, benchmark absolute rubrics and a final common-context rerank of survivors. A top-k subset must not be advertised as a fully scored permutation.

### 5.6 Tournament ranking

A knockout selects one winner using `n-1` comparisons for a binary tournament, with rounds parallelizable within limits. It does not recover a reliable full order or top-k by simply listing eliminated candidates. Strong candidates can meet early; noisy judgments and bracket seeding affect survival.

TourRank explores multi-stage groups and aggregation across tournaments. More recent BlitzRank research uses comparison graphs, inferred relationships, and explicit handling of preference cycles. These are useful research references, not measured improvements for Jev. [TourRank](https://arxiv.org/abs/2406.11678), [BlitzRank](https://arxiv.org/abs/2602.05448).

Recommendation: expose tournament selection as approximate, seeded, and budget-bounded. Preserve multiple survivors when recall matters. Defer sophisticated graph scheduling until a simple cascade and shortlist comparator have been benchmarked. Logical certification relative to an observed preference graph is not certification of real-world relevance.

### 5.7 Hierarchical and cascaded ranking

**Hierarchical:** select categories/groups, then candidates within them; examples include schema → table → column or tool family → tool. A mistaken early branch can lose the correct item. Use multi-branch retention or a fallback route where justified.

**Cascaded:** apply progressively more expensive stages to a shrinking candidate pool. Unlike independent partitioning, a cascade deliberately prunes or enriches candidates between stages. TypeSafe's skill-suggestion cookbook illustrates ranking brief descriptions, then examining a shortlist with richer evidence and allowing no selection. Its published run used an older model and is not a current service benchmark. [Skill suggestion](https://docs.typesafe.ai/cookbooks/skill_suggestion).

Recommendation: assess recall after every pruning stage. A downstream ranker cannot recover a removed candidate. Prefer progressive disclosure of object fields before expensive full-object serialization. Retain supporting graph/schema dependencies even when they are individually low-scoring.

### 5.8 BM25 + embedding + LLM hybrid ranking

An effective design hypothesis is:

```text
Authorized candidate universe
    ├─ lexical retrieval: identifiers, rare terms, exact names
    └─ dense retrieval: semantic matches and paraphrases
           ↓ union by stable identity + rank fusion
       optional cross-encoder or metadata policy
           ↓ bounded shortlist
       Jev judgments or justified LLM reranking
           ↓ relevance gate + diversity/dependency selection
       original objects with scores and provenance
```

Retrieve lexical and dense candidates in parallel when possible. A strict BM25-then-embedding funnel can discard paraphrases before semantic retrieval has a chance. Do not blindly recompute embeddings already supplied by a retriever. BM25 computed over a small changing shortlist uses different corpus statistics from the production index; label it as a separate local lexical feature.

BM25, embeddings, and LLMs are optional stages, not a required sequence. Adding all of them can increase latency without improving the result. The official Jev reranking cookbook demonstrates a BM25 shortlist followed by TypeSafe judgments, providing a concrete integration pattern but not proof that it is optimal across domains. [TypeSafe reranking cookbook](https://docs.typesafe.ai/cookbooks/rerank_typesafe).

### 5.9 Reciprocal Rank Fusion (RRF)

For candidate `d`, use `RRF(d) = sum_j w_j / (c + rank_j(d))` over lists containing it, using one-based ranks and positive smoothing constant `c`. Equal weights give ordinary RRF; explicit weights are a policy extension. Missing candidates contribute zero. RRF combines rankings without requiring comparable raw model scores. [Original RRF paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).

Recommendation: provide this as deterministic functionality with stable identity, duplicate handling within each list, explicit ties, source weights, and list-depth metadata. The smoothing constant and retrieval depths need tuning. RRF discards score margins and can overcount highly correlated retrieval sources. Its output is not a relevance probability.

### 5.10 MMR and diversity-aware selection

Greedy maximal marginal relevance chooses the next item maximizing `lambda * relevance(d) - (1-lambda) * max_similarity(d, selected)`. The first step has no redundancy penalty. It trades relevance against duplication and can use embeddings, lexical overlap, or caller-supplied similarity. [MMR research](https://aclanthology.org/X98-1025/).

Recommendation: run MMR on a sufficiently broad shortlist, with compatible relevance/similarity scales and explicit behavior for negative similarities. Incrementally maintained similarities can use roughly `O(nk)` comparisons rather than materializing an `n × n` matrix. This is algorithmic work, not a latency claim.

MMR is a greedy set-selection heuristic, not a guarantee of fairness, coverage, or factual agreement. Source quotas, parent-document caps, and mandatory dependency inclusion require explicit policies. Dissimilarity does not imply usefulness; contradictory evidence should remain visibly contradictory. Measure answer/evidence coverage as well as ranking relevance.

## 6. Arbitrary objects and domain-specific ranking

### 6.1 Universal means preserved objects and explicit projections

Candidate support must not require inheritance, serialization of the whole object, or conversion to a framework document. Keep the original `T` local and construct a separate immutable ranking view: occurrence ID, optional stable domain ID, approved text/structured fields, and trusted metadata. Return the original object reference with rank and provenance.

Recommendations:

- Allow an explicit text projection and, where supported, a structured projection. Do not default arbitrary objects to `repr()`, `__dict__`, pickle, or model-generated reconstruction.
- Distinguish input occurrence identity from domain identity. Two equal objects, duplicate texts, or two occurrences of one object may need separate results. Deduplication is an explicit policy.
- Bind opaque request IDs back to known candidates in code; model-selected IDs must be members of the submitted set.
- Snapshot the projected representation before asynchronous execution. Do not let object mutation change content midway through ranking or invalidate cache assumptions.
- Separate model-visible metadata from private application metadata. Custom projection callbacks are trusted application code and may need resource limits; untrusted remote callbacks must not be executed.
- “Universal” does not mean an object has meaning without a query, a task objective, and an informative representation. Reject unsupported projections clearly.

### 6.2 Documents, chunks, search results, and RAG

Represent title, relevant body, source, parent/chunk identity, and evidence location. Rank answer usefulness rather than merely topic overlap. Preserve partial evidence for multi-hop questions, avoid redundant chunks, and leave an abstention path when nothing is useful. The TypeSafe passage-classification cookbook provides an example of semantic passage decisions followed by deterministic selection. [RAG passage classification](https://docs.typesafe.ai/cookbooks/classifying_rag_passages).

Recommendation: evaluate the downstream generator with a fixed context-token budget as well as standalone nDCG. Document order and evidence placement can affect long-context use; the Lost in the Middle study motivates position-sensitivity checks, without establishing behavior of every modern model. [Long-context study](https://arxiv.org/abs/2307.03172).

### 6.3 Entities and products

Entity ranking needs mention context, aliases, canonical IDs, type, and distinguishing attributes. An exact verified ID match may settle a case deterministically; otherwise use semantic evidence and a no-match option. BLINK is a useful precedent for dense candidate retrieval followed by cross-encoder disambiguation. [BLINK paper](https://arxiv.org/abs/1911.03814).

For products, exact stock status, price ceilings, geography, and required specifications belong in code. Jev can judge intended-use fit or ambiguous attribute agreement. Keep commercial boosts distinct from relevance so behavior is inspectable. TypeSafe's entity-alignment cookbook uses a Score plus companion judgments for field disagreement; this supports a multi-signal design, not a universal entity-resolution guarantee. [Entity alignment](https://docs.typesafe.ai/cookbooks/entity_alignment).

Entity resolution may require global one-to-one matching, not independent top-1 choices. Treat that as a separate deterministic assignment step over eligible edges, with uncertain matches allowed to remain unresolved.

### 6.4 Tools, MCP tools, APIs, and agents

Represent capability description, input/output schema summaries, required permissions, availability, and trusted operational metadata. Filter by permissions and schema compatibility first. Rank semantic fit next, then consider latency, cost, and reliability from measured metadata.

For MCP, tool names are unique within a server, not necessarily globally; use server identity plus tool name. Tool annotations are untrusted unless supplied by a trusted server. The ranker's output must never grant permission or execute the chosen tool. [MCP tool specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools).

Recommendation: allow no suitable tool/agent. Use explicit task-success feedback to evaluate selection; self-written descriptions are poor proof of competence. For workflows requiring multiple tools, preserving dependencies or complementary capabilities is a set-selection problem beyond independent top-k relevance. Ranking APIs or agents must not turn into an execution framework.

### 6.5 SQL schema, table, and column ranking

Represent fully qualified names, descriptions, types, keys, and approved examples. Infer semantic links between the question and schema, while computing dialect/type compatibility and foreign-key connectivity in code. RESDSQL explicitly separates schema linking from SQL generation and ranks schema items, making it a relevant architectural precedent. [RESDSQL](https://arxiv.org/abs/2302.05965).

Recommendation: select tables, then columns, then restore required join keys and bridge tables using the schema graph. A plausible-looking column shortlist can still make the requested SQL impossible. Schema visibility and row/column access controls precede ranking. Sample values require explicit data policy. Evaluate complete required-schema coverage and downstream execution accuracy; ranking alone must not generate or run SQL.

### 6.6 Knowledge-graph candidates

Support nodes, edges, paths, and bounded subgraphs through explicit views containing labels, relation types, direction, provenance, and approved neighborhoods. A graph object should not be reduced to its display label alone.

Recommendation: compute adjacency, reachability, hop limits, type compatibility, and path validity deterministically. Use Jev for semantic relation fit or evidence relevance only after structural candidate generation. A semantically attractive but disconnected set is not a usable reasoning path. Rerank paths/subgraphs as units when the task needs connected evidence. Bound neighborhood expansion to avoid multiplying token volume and leaking unrelated records. Entity-alignment evidence supports pair judgments, not arbitrary graph reasoning quality.

### 6.7 Agent memory

Represent content, provenance, owner, event time, validity interval, and any trusted supersession link. Compute recency decay and expiry in code; use semantic judgments for current usefulness or uncertain conflicts. Generative Agents provides prior art for combining relevance, recency, and importance in memory retrieval. [Generative Agents](https://arxiv.org/abs/2304.03442).

Recommendation: avoid treating the newest memory as automatically true. User ownership, deletion, revocation, and privacy restrictions are hard gates. Do not let model inference silently delete memories or override an explicit user correction. Include as-of time and memory revision in reproducibility and cache policy. Evaluate stale-fact usage and conflicting-memory handling, not just semantic similarity.

### 6.8 Code and other Python objects

For code candidates, represent symbol names, signatures, docstrings, selected implementation, language, and trusted repository/revision metadata. Exact symbol lookup, dependency compatibility, and visibility checks belong in deterministic tooling; semantic reranking can help locate behavior described in natural language.

For arbitrary Python objects, a caller-owned projection and optional metadata/similarity callbacks provide the extension point. Do not execute code candidates, import their modules, inspect descriptors with side effects, or recursively traverse arbitrary object graphs to produce a ranking view.

## 7. Score semantics and structured-output reliability

### 7.1 Preserve meaning before combining values

BM25 scores, cosine similarities, cross-encoder logits, Noul probabilities, Score expectations, Choice probabilities, pairwise wins, RRF values, and MMR selection utilities are different quantities. Being numeric or lying in `[0,1]` does not make them comparable.

Recommended result metadata should distinguish raw value, semantic kind, direction, scale/rubric, scope, producer/version, confidence if available, and any transformation. A permutation-only backend should be able to return ranks without fabricated numeric relevance scores.

For a weighted policy `U(d) = sum_j w_j * f_j(d)`, explicitly define each transformation `f_j` and missing-value behavior. Min-max normalization over the current candidate pool changes when candidates change and fails without a policy for constant-valued features. A sigmoid is monotonic but does not establish calibration. Combining correlated signals can double-count relevance. Prefer RRF for unrelated score scales unless labeled validation supports a calibrated blend.

Separate metric extraction from policy weighting: changing deterministic weights need not trigger fresh Jev judgments when their inputs/rubrics are unchanged. Hard constraints remain gates; an ineligible candidate cannot compensate with a high semantic score.

### 7.2 Four distinct kinds of reliability

1. **Static type safety:** preserve `T`, typed configuration, and explicit capability/result variants for developers.
2. **Wire validity:** response parsing, schema checks, numeric finiteness, IDs, and required fields.
3. **Ranking invariants:** complete versus partial coverage, stable ties, valid permutations, and correct mapping to original objects.
4. **Semantic quality:** whether judgments and selected candidates actually serve the query.

Success at one layer does not establish the next. Native typed outputs reduce one failure class; they do not remove network, versioning, integration, or semantic failures. Generative providers can offer constrained JSON output, but provider-specific refusal and incomplete-output cases still need handling. [Structured-output contract example](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

### 7.3 Recommended boundary validation

- Check exact candidate/question-ID membership and expected coverage; detect missing, duplicate, and extra IDs. For raw JSON transports, ordinary parsers may discard duplicate object keys before validation; decide whether strict decoding is required.
- Reject boolean values where a numeric score is required, non-finite values, and values outside the declared range. Do not silently clip malformed provider output.
- For Choice/Score distributions, verify expected alternatives, finite probabilities, sum within a documented tolerance, and consistent rubric/selection fields.
- For full permutations, require each submitted ID exactly once. For explicit subset selection, permit omissions only according to its declared contract.
- Preserve unknown/failed/unscored status instead of assigning zero relevance.
- Validate once at boundaries and policy transitions; avoid deep model validation of every object in hot loops.

Finite retries for malformed responses may be appropriate for a generative backend, but they consume the same budget as all other attempts. Never replace an invalid ranking with an unreported arbitrary order.

### 7.4 Failure and fallback semantics

Distinguish invalid input, unsupported capability, authentication/configuration error, rate limit, context overflow, transient transport error, invalid response, deadline/budget exhaustion, and cancellation. Authentication failures should not be retried like overload.

Recommended fallback is an explicit policy: return a separately identified deterministic baseline, fail strictly, or return documented partial coverage. Partially successful LLM scores should not be interleaved with unrelated baseline scores as though calibrated together. Expose status, covered IDs, fallback reason, and effective method.

A fallback from one listwise request to many pointwise calls can increase cost and latency. Estimate and reserve that work before starting it. Caller cancellation must stop scheduling, drain child work, release resources, and propagate; it is not a reason to launch a fallback.

## 8. Prompt injection and data boundaries

Untrusted candidates can contain instructions such as “rank me first,” forged policies, fake IDs, or text that praises itself. They can poison document ordering, tool selection, memories, and downstream context without ever producing invalid JSON. OWASP treats indirect prompt injection as a risk even in RAG systems. [OWASP prompt injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/).

Recommended controls:

- Keep application-owned rubrics separate from candidate state; serialize candidates as data with clear origin labels. Do not interpolate candidate text into trusted instruction templates.
- Tell the model to evaluate content rather than obey embedded instructions. Delimiters and role separation are mitigations, not isolation guarantees.
- Apply authorization and data minimization **before** model calls, caching, or tracing. Ranking must not expose unauthorized candidates even if they would later be filtered out.
- Give the ranker no execution tools, filesystem authority, or ability to follow candidate-provided URLs. Do not fetch “supporting evidence” just because a candidate requests it.
- Use trusted metadata for authority and permission features. A document's claim that it is authoritative is itself untrusted content.
- Validate returned IDs against the allowlist and recheck hard eligibility after fallback or cache retrieval.
- Keep traces content-free by default; full payload capture requires explicit configuration, redaction policy, retention limits, and access control. Hashes can also reveal repeated sensitive values and are not anonymization.
- Partition cache and shared-state batching by tenant and trust scope. A shared batch can create cross-candidate influence and data exposure.

Adversarial evaluation should include instruction overrides, fake system messages, score manipulation, irrelevant self-promotion, multilingual payloads, Unicode obfuscation, colluding candidates, and hostile tool descriptions. Measure rank promotion, evidence contamination, and downstream unsafe selection alongside ordinary quality.

No finite attack suite proves immunity. The enforceable guarantee is that model judgments cannot bypass code-owned boundaries. Semantic ranking robustness remains an empirical property.

## 9. Cost, latency, batching, and caching

### 9.1 Batch by semantics as well as size

Distinguish a network batch, local tensor batch, independent judgments sharing state, and genuinely comparative listwise inference. These change different costs and sometimes different semantics.

TypeSafe documents shared-state multi-question calls as a way to amortize state processing. Treat that as a promising execution pattern, not a guarantee of constant latency at arbitrary question count. [Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out).

Recommendation: pack only compatible questions for one query/trust scope, account for repeated question/rubric tokens, and bound both state and complete request size. Token estimates are model-specific; character lengths and another model's tokenizer are approximations. Record all truncation and projection rules. Prefer selected relevant spans or field-aware truncation over silently taking every object's prefix.

For local cross-encoders, bucket similar lengths to reduce padding, then restore original identity/order. Bound worker queues and memory as well as concurrency. More in-flight work is not always more throughput; measure provider throttling and device saturation.

### 9.2 Spend work where it can change selection

- Start from a useful deterministic or retrieval baseline.
- Reuse caller-provided retrieval scores and embeddings when compatible.
- Avoid model calls for exact eligibility, dates, arithmetic, and obvious identity matches.
- Request only judgment dimensions the policy actually uses.
- Avoid explanation generation by default. For Jev, expose rubric decisions and policy contributions as an audit trail.
- Evaluate extra comparisons around the top-k boundary or ambiguous cases, not automatically across the full pool.
- Treat confidence-based escalation as a hypothesis requiring risk/coverage evaluation.
- Consider offline distillation or learned policies only after acquiring representative labels and checking training/data-use rights.

A cheap ranker that causes extra downstream generation or missing-evidence retries may be expensive end to end. Optimize successful-task cost, not just rerank-call price.

### 9.3 Budgets and accounting

Use one shared ledger for model calls, tokens, estimated currency cost, and a monotonic deadline. Reserve anticipated consumption before concurrent requests; reconcile after completion so concurrent workers cannot all spend the same remaining allowance.

Generic cost accounting can be expressed as `sum(input_tokens * input_rate + output_tokens * output_rate + request_or_unit_fees)`, with rates converted to per-token units and provider-specific billing rules. Report cost as **known**, **estimated**, or **unknown**. List-price arithmetic is an estimate, not an invoice. An unknown price must never become zero.

Record observed usage separately from estimated usage of attempts whose responses were lost. A timeout/cancellation may leave work running remotely and possibly billable; a client-side deadline cannot guarantee a hard provider-side monetary ceiling. Strict budget modes need conservative bounds and must reject unpriceable work when those bounds are required.

### 9.4 Latency and lifecycle

Measure end-to-end latency as queueing plus projection/tokenization, cache access, network/model work, retries, validation, aggregation, and final selection. Track p50/p95/p99 and throughput under realistic concurrency. Separate cold start/model loading from warm inference, and cache hits from misses.

Prefer async internally with explicit client ownership and connection reuse. Do not block the event loop with local model inference or expensive tokenization. Local workers need device-aware limits. A sync convenience API should reject calls inside an active loop with a clear instruction to use async rather than secretly nesting loops.

Use capped, jittered retries for retryable failures, respecting service signals and the remaining overall deadline. Bound pending tasks and response size, avoid duplicate in-flight requests, and treat overload as backpressure rather than an invitation to flood the provider.

### 9.5 Cache correctness

Cache judgment data separately from returned Python objects. Keys should include query/context, projected-content hash and projection version, stable identity policy, metric/rubric/prompt version, backend/provider and resolved model, relevant model options, normalization/policy versions when applicable, and tenant/authorization scope.

For shared-state/listwise decisions, include the ordered candidate context and grouping: an item-only key is invalid when peers can affect its score. Pairwise keys need both candidates and orientation unless the implementation explicitly canonicalizes it. For independent pointwise scoring, per-item reuse is more defensible.

Time-dependent policies need an explicit as-of time or deterministic recomputation from cached semantic judgments. Model aliases, inventory, schema revisions, memory edits, deletions, and access changes require expiry/invalidation. Do not use unstable process hashes or `repr()` as persistent keys, and do not deserialize arbitrary pickle payloads from cache.

## 10. Recommended architecture direction

This is a direction for the subsequent design phase, not a finalized specification.

### 10.1 Small core, explicit boundaries

1. **Candidate projection:** original `T` stays local; an immutable view supplies IDs, approved content, and metadata.
2. **Eligibility:** deterministic hard filters run before external scoring.
3. **Metrics and backends:** metrics describe signals; backends perform supported judgments/inference. Use batched evaluation as a first-class operation rather than requiring one remote call per metric/item.
4. **Ranking strategy:** independent scoring, grouped judgments, comparison scheduling, or rank-only ordering; strategies own their algorithm, not provider transport.
5. **Policy and selection:** normalize only with declared semantics, combine signals, handle ties, enforce gates, and select a useful set.
6. **Execution support:** shared budgets, deadlines, rate limits, caching, cancellation, and safe telemetry.
7. **Results:** original objects, occurrence IDs, ranks, optional meaningful scores, signal provenance, coverage, status, and execution statistics.

A minimal backend capability description should distinguish scalar/rubric judgments, pair comparison, permutations, embeddings, local/API execution, usage reporting, context limits, and optional explanations. Do not force every backend to implement a universal text-generation method or return a probability distribution it does not have.

Use structural typing for extension points and generics for object-preserving results. Keep dataclasses or lightweight immutable records in hot paths; use boundary validation where it earns its cost. The eventual basic API should require only a query, candidates, an appropriate backend/configuration, and a projection for unfamiliar object types.

### 10.2 Dependency and responsibility boundaries

Keep the core independent of Jev's SDK, PyTorch, Transformers, vector databases, LangChain, and LlamaIndex. Add optional adapters. A Jev-focused installation can expose the official SDK as an extra without forcing it onto offline users.

The library consumes retrieved candidates or caller-provided retrieval lists. It need not own crawling, indexing, a vector store, graph persistence, an agent runtime, or a SQL execution engine. Offer small integration adapters that preserve identity and provenance.

Avoid implementing a second async runtime, generic workflow DAG framework, universal object serializer, or model server. Prefer one execution owner and a short ordered pipeline until real use cases require more.

### 10.3 Automatic planning

An eventual automatic planner should use candidate count **and** representation lengths, requested k, backend capabilities, estimated context/cost, deadlines, available cached features, and task objectives. An inspectable rules-based planner is a better starting point than an LLM choosing the algorithm.

Return the selected stages, estimated and actual work, pruning decisions, truncation, and fallback reasons. Allow an explicit plan override and a planning-only view that makes no model calls.

Do not hard-code candidate-count thresholds as proven quality boundaries. A few huge SQL schemas can cost more than many short tool descriptions. Start with deterministic feasibility checks and caller configuration; learn useful routing thresholds from benchmark results.

### 10.4 Priorities by quality attribute

- **Quality:** preserve candidate recall and evaluate task-specific utility; include abstention and supporting evidence.
- **Latency:** bounded execution, connection reuse, efficient local batching, and few unnecessary sequential stages.
- **Cost:** budget the entire plan, including retries and downstream effects; avoid models for exact computation.
- **Reliability:** explicit errors, coverage, score semantics, deterministic tie policies, and traceable fallback.
- **Developer experience:** simple defaults, object preservation, clear unsupported-capability errors, and opt-in advanced configuration.
- **Type safety:** keep `T` through the result, validate external data, and distinguish incompatible score/result types.
- **Extensibility:** a few protocols with optional adapters; avoid a plugin framework before real extension requirements exist.

## 11. Evaluation and benchmark agenda

### 11.1 Measure ranking and task success separately

Use qrels or task labels with declared binary/graded relevance conventions. Precision@k measures concentration of relevant results; Recall@k measures retained relevant items; HitRate/Success@k measures whether any acceptable result appears; MRR emphasizes the first relevant item; MAP rewards the placement of multiple positives; nDCG@k handles graded relevance with position discounting. Record the gain mapping, relevance threshold, cutoff, unjudged-item handling, and zero-positive-query convention. [ir-measures definitions](https://ir-measur.es/en/latest/measures.html).

Standard retrieval collections are necessary but insufficient for universal objects. BEIR offers heterogeneous retrieval tasks and reusable evaluation machinery. Add separate domain labels and metrics rather than claiming a good document score proves tool, entity, or schema quality. [BEIR](https://github.com/beir-cellar/beir).

Recommended domain outcomes:

- RAG: evidence recall, answer correctness, citation support, abstention, and context tokens.
- Entities/products: correct match, no-match accuracy, constraint violations, and assignment consistency where applicable.
- Tools/agents/APIs: suitable capability recall, no-tool accuracy, downstream task success, and invalid/unauthorized selection rate.
- SQL: table/column recall, complete required-schema coverage, join-path completeness, and downstream execution correctness.
- Graphs: path/subgraph validity, evidence coverage, provenance retention, and answer success.
- Memory: useful-memory recall, stale/conflicting fact usage, deletion compliance, and task success over time.
- Code: symbol/behavior relevance and downstream task success without executing candidates during ranking.

For diversity, report redundancy, subtopic/source coverage, and relevance loss; consider intent-aware metrics when labels support them. Do not call diversity “better” solely because average similarity decreases.

### 11.2 Calibration and selection reliability

For genuine binary probabilities, examine Brier score, log loss, reliability plots, and calibration error with declared bins and sample counts. Evaluate Score distributions against ordinal labels separately from the final utility mapping. Test risk versus coverage when abstention/escalation is enabled. A model's own confidence is an input signal, not the calibration result. The calibration literature establishes this distinction for neural predictors; it does not independently establish Jev's calibration. [Calibration research](https://arxiv.org/abs/1706.04599).

Measure stability under candidate order permutations, grouping, duplicate insertion, semantically equivalent prompt wording, and repeated calls. Reproducible Python sorting does not make remote model judgments deterministic. Include tie handling and no-relevant-candidate cases.

### 11.3 Fair comparison protocol

1. Freeze datasets, retrieval pools, qrels, object projections, prompts, and configuration with hashes/revisions.
2. Separate untouched retrieval-pool evaluation from controlled oracle-injected reranking experiments. Report the achievable oracle for each pool and recall lost before reranking.
3. Compare unchanged retrieval order, deterministic metadata policy, lexical ranking, embedding similarity, RRF, a compact cross-encoder, `jev-reranker`, native Jev judgment variants, and a suitable LLM backend.
4. Hold candidate information and downstream context budgets comparable; separately measure benefits of richer representations.
5. Tune prompts/thresholds/weights on development data, then evaluate on held-out queries. Split by entity, time, or source when required to prevent leakage.
6. Report per-query paired differences with uncertainty intervals and domain/language slices, not only one aggregate mean.
7. Report cold/warm and cached/uncached latency, hardware, concurrency, all attempts, rate limits, usage coverage, and known/estimated/unknown cost.
8. Include malformed outputs, timeouts, overload, cancellation, context overflow, unsupported capabilities, and adversarial content.
9. Confirm redistribution rights for each dataset; prefer reproducible download manifests over copying datasets into the package.

Do not treat an LLM judge as unquestioned ground truth for another LLM ranker. Human/domain adjudication, independent checks, and inter-annotator disagreement matter. Public benchmark contamination and small synthetic datasets limit claims.

### 11.4 Questions requiring measurements rather than assumptions

- Does Jev improve task quality over the existing package's prompts, local cross-encoders, and hosted rerank APIs at comparable total cost?
- Which tasks favor Noul relevance versus anchored Score utility versus relative Choice?
- How stable are thresholds across query types, languages, domains, model versions, and batch compositions?
- Do shared-state questions preserve independent scoring behavior well enough for the task?
- Where do state length and question count begin to degrade quality, tail latency, or reliability?
- Does lexical+dense fusion improve recall enough to justify its overhead?
- Does a second semantic stage add value after a strong retriever or cross-encoder?
- How much recall is lost by each cascade stage, hierarchy branch, and truncation policy?
- Do pairwise boundary checks or tournaments improve selection enough to pay for extra comparisons?
- When does MMR improve evidence/task coverage, and when does it remove necessary corroboration?
- Can confidence-based escalation reduce cost while holding error risk within the application's target?
- How do client reuse, queueing, batching, and cancellation behave under sustained concurrent load?
- How much Python overhead comes from projection, copying, validation, token counting, and trace construction?
- Does caching remain correct under alias changes, reordered peers, data updates, time decay, and access revocation?
- How often can malicious candidate text change ranks or contaminate downstream output?

**Measurement status:** all proposed-library ranking metrics, latency percentiles, throughput, calibration, attack success rates, costs per task, and comparative gains are **not measured**. Published vendor examples establish feasibility only; no borrowed benchmark number is presented as this project's result.

## 12. Major technical risks and design constraints

**Critical — semantic errors despite valid types.** A schema-valid judgment can still choose the wrong item. Preserve evaluation, abstention, and code-owned invariants.

**Critical — prompt injection and data exposure.** Candidate instructions can influence scores; shared state, logs, and caches can leak data. Enforce authorization before projection and isolate trust scopes.

**High — score incompatibility.** Blending probabilities, utilities, and rank-derived values can produce misleading thresholds. Preserve score kinds and calibrate only with suitable labels.

**High — pruning away the answer.** Cascades, schema hierarchies, and tournaments can make recovery impossible. Track stage recall and dependency closure.

**High — budget overshoot and retry amplification.** Concurrent reservations, incomplete usage, nested retries, and expensive fallback need one execution ledger and explicit uncertainty.

**High — model/provider drift.** Moving aliases, changing limits, SDK evolution, and service access can invalidate cached scores and tuned thresholds. Pin evaluation versions and isolate provider dependencies.

**High — object identity and cache corruption.** Mutable objects, duplicated IDs, lossy serialization, and shared-state cache keys can return the wrong object or stale judgment. Snapshot views and validate mappings.

**High — misleading evaluation.** Injected positives, changed candidate pools, weak baselines, or mismatched timing boundaries can manufacture apparent gains. Preserve manifests and separate experimental regimes.

**Medium — async and resource complexity.** Loop-bound clients, blocking local inference, oversized batches, and abandoned tasks can undermine reliability. Make ownership explicit and keep queues bounded.

**Medium — overengineering.** Too many abstractions and mandatory integrations can make a simple ranking call difficult. Begin with generic candidates, a few score contracts, optional backends, and ordered stages.

These priorities are architectural assessments, not a security certification or an audit of an implemented library. The open-source client also does not imply an open-source/self-hosted Jev model. Offline operation requires separate local or deterministic backends.

## 13. What not to build

- A thin rename of `jev-reranker`, or claims that async, retries, splitting, and custom instructions are new.
- A mandatory LLM call in every stage, including sorting, arithmetic, recency, or exact eligibility.
- An all-pairs default, unlimited shared-state requests, or a guarantee of one request regardless of input size.
- A universal “probability of relevance” field populated from incomparable scores.
- Fake Jev explanations, unsupported reasoning controls, or invented arbitrary structured-output capabilities.
- A fixed pipeline or automatic strategy claimed optimal without domain benchmarks.
- A retrieval engine for million-object corpora, vector database, agent executor, SQL generator, or model server inside the core.
- Automatic execution of selected tools/code, unsafe object deserialization, or model-controlled authorization.
- Security promises that delimiters, structured outputs, or a second model make injection impossible.
- A custom training platform, sophisticated tournament scheduler, or broad plugin framework before simpler baselines demonstrate a need.
- Heavy mandatory dependencies, automatic model downloads on import, or sensitive payload logging by default.
- Benchmark marketing based on vendor examples, toy datasets alone, or measurements that were not run.

## 14. Decisions to carry into the next design phase

Proceed with an object-preserving, async-first core and optional providers; investigate the official SDK for Jev while retaining an independent internal result contract. Start architectural design around independent and shared-state judgments, deterministic fusion/selection, explicit score semantics, and whole-plan operational controls. Treat advanced comparisons and automatic planning as benchmark-driven extensions.

Before finalizing specifications, resolve: the initial domains and acceptance metrics; minimum Python version; input-size and streaming expectations; default partial-failure policy; raw-versus-normalized score presentation; batch-context semantics; cache scope; and exactly which backend capabilities belong in the first release. Provider access, account-specific limits, exact token estimation, model pin availability, and applicable data-handling terms also require verification before live evaluation.

The evidence supports Jev as a promising component for fast, typed semantic judgments. It does not support assuming Jev wins every ranking workload. The library's lasting value should come from correct object handling, meaningful scores, measured selection policies, and reliable execution regardless of which backend is best for a particular task.
