# Lightweight integrations

The core package imports no LangChain, LlamaIndex, vector database, or search-engine SDK. Adapters only project text from objects you already retrieved; reranking never performs retrieval or changes permissions.

```python
from jev_rankkit import Reranker
from jev_rankkit.integrations import (
    langchain_adapter,
    llamaindex_adapter,
    qdrant_adapter,
    pinecone_adapter,
    elasticsearch_adapter,
    opensearch_adapter,
)

response = await Reranker().rerank(
    query="vector search",
    candidates=langchain_documents,
    adapter=langchain_adapter(),
    top_k=5,
)
```

`langchain_adapter()` reads `page_content`; `llamaindex_adapter()` reads a node's `get_content()` or `text`, including `NodeWithScore.node`; `qdrant_adapter(text_key="text")` reads `payload`; `pinecone_adapter(text_key="text")` reads match `metadata`; and `elasticsearch_adapter(text_key="text")` / `opensearch_adapter(...)` read hit `_source`. These are common shapes, not guarantees for every SDK version. If your records differ, pass `text_fn` or `CandidateAdapter` instead.

Results return the exact input objects. Keep framework-specific IDs and payloads on those objects, and apply authorization filters before ranking. Candidate content is untrusted; only explicitly projected fields reach the model backend. See [framework examples](../examples/15_framework_adapters.py), [RAG](../examples/02_rag_reranking.py), [agent/MCP tools](../examples/08_tool_reranking.py), and [SQL selection](../examples/09_sql_schema_reranking.py).
