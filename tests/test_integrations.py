from __future__ import annotations

from dataclasses import dataclass

import pytest

from jev_rankkit import (
    Reranker,
    rerank_documents,
    rerank_entities,
    rerank_memories,
    rerank_tools,
)
from jev_rankkit.errors import ProjectionError
from jev_rankkit.integrations import (
    elasticsearch_adapter,
    langchain_adapter,
    llamaindex_adapter,
    opensearch_adapter,
    pinecone_adapter,
    qdrant_adapter,
)


@dataclass
class Document:
    page_content: str


@pytest.mark.asyncio
async def test_convenience_functions_preserve_original_objects() -> None:
    ranker = Reranker()
    document = Document("vector search database")
    entity = {"name": "VectorDB", "description": "a vector database"}
    tool = {"name": "search", "description": "search vector documents"}
    memory = {"content": "user prefers vector databases"}
    calls = (
        (rerank_documents, document),
        (rerank_entities, entity),
        (rerank_tools, tool),
        (rerank_memories, memory),
    )
    for wrapper, item in calls:
        response = await wrapper(ranker, query="vector", candidates=[item], top_k=1)
        assert response.results[0].item is item


@pytest.mark.asyncio
async def test_bound_convenience_and_custom_projection() -> None:
    item = {"custom": "vector search"}
    response = await Reranker().rerank_documents(
        query="vector", candidates=[item], text_fn=lambda value: value["custom"]
    )
    assert response.results[0].item is item
    with pytest.raises(ProjectionError):
        await Reranker().rerank_tools(query="q", candidates=[object()])


@pytest.mark.asyncio
async def test_optional_adapters_need_no_framework_imports() -> None:
    class Node:
        def get_content(self) -> str:
            return "vector memory"

    cases = (
        (langchain_adapter(), Document("vector document")),
        (llamaindex_adapter(), {"node": Node()}),
        (qdrant_adapter(), {"payload": {"text": "vector point"}}),
        (pinecone_adapter(), {"metadata": {"text": "vector match"}}),
        (elasticsearch_adapter(), {"_source": {"text": "vector hit"}}),
        (opensearch_adapter(), {"_source": {"text": "vector hit"}}),
    )
    for adapter, item in cases:
        response = await Reranker().rerank(query="vector", candidates=[item], adapter=adapter)
        assert response.results[0].item is item
