"""Dependency-free projections for common retrieval result shapes.

These adapters do not import the corresponding frameworks. They preserve the original
object and only extract text for ranking. Supply ``text_fn`` directly for other shapes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .candidates import CandidateAdapter
from .errors import ProjectionError


def _field(value: object, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _require_text(value: object, *, source: str) -> str:
    if not isinstance(value, str):
        raise ProjectionError(f"{source} text field must be a string")
    return value


def langchain_adapter() -> CandidateAdapter[Any]:
    """Project ``Document.page_content`` without importing LangChain."""
    return CandidateAdapter(
        text_fn=lambda document: _require_text(
            _field(document, "page_content"), source="LangChain"
        ),
        version="langchain-document-v1",
    )


def llamaindex_adapter() -> CandidateAdapter[Any]:
    """Project a node or ``NodeWithScore.node`` using ``get_content()``."""

    def text(item: object) -> str:
        node = _field(item, "node") or item
        getter = _field(node, "get_content")
        if callable(getter):
            return _require_text(getter(), source="LlamaIndex")
        return _require_text(_field(node, "text"), source="LlamaIndex")

    return CandidateAdapter(text_fn=text, version="llamaindex-node-v1")


def qdrant_adapter(*, text_key: str = "text") -> CandidateAdapter[Any]:
    """Project ``ScoredPoint.payload[text_key]``."""

    def text(point: object) -> str:
        payload = _field(point, "payload")
        return _require_text(_field(payload, text_key), source="Qdrant")

    return CandidateAdapter(text_fn=text, version=f"qdrant-payload-{text_key}-v1")


def pinecone_adapter(*, text_key: str = "text") -> CandidateAdapter[Any]:
    """Project a query match's metadata text field."""

    def text(match: object) -> str:
        metadata = _field(match, "metadata")
        return _require_text(_field(metadata, text_key), source="Pinecone")

    return CandidateAdapter(text_fn=text, version=f"pinecone-metadata-{text_key}-v1")


def elasticsearch_adapter(*, text_key: str = "text") -> CandidateAdapter[Any]:
    """Project a hit's ``_source[text_key]``; also works with OpenSearch hits."""

    def text(hit: object) -> str:
        source = _field(hit, "_source")
        return _require_text(_field(source, text_key), source="Elasticsearch/OpenSearch")

    return CandidateAdapter(text_fn=text, version=f"elasticsearch-source-{text_key}-v1")


def opensearch_adapter(*, text_key: str = "text") -> CandidateAdapter[Any]:
    return elasticsearch_adapter(text_key=text_key)
