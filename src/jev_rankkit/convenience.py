"""Thin domain projections over the generic reranking API."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from .errors import ProjectionError
from .types import RerankResponse

if TYPE_CHECKING:
    from .api import Reranker

T = TypeVar("T")


def _field(item: object, name: str) -> object | None:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _text(item: object, fields: tuple[str, ...], *, label: str) -> str:
    if isinstance(item, str):
        return item
    parts = [value for name in fields if isinstance(value := _field(item, name), str) and value]
    if not parts:
        raise ProjectionError(f"{label} needs text_fn or a supported text field")
    return "\n".join(parts)


async def rerank_documents(
    reranker: Reranker,
    *,
    query: str,
    candidates: Sequence[T],
    text_fn: Callable[[T], str] | None = None,
    **kwargs: Any,
) -> RerankResponse[T]:
    """Project common document text fields while preserving original objects."""
    return await reranker.rerank(
        query=query,
        candidates=candidates,
        text_fn=text_fn
        or (
            lambda item: _text(item, ("page_content", "text", "content", "body"), label="document")
        ),
        **kwargs,
    )


async def rerank_entities(
    reranker: Reranker,
    *,
    query: str,
    candidates: Sequence[T],
    text_fn: Callable[[T], str] | None = None,
    **kwargs: Any,
) -> RerankResponse[T]:
    return await reranker.rerank(
        query=query,
        candidates=candidates,
        text_fn=text_fn
        or (lambda item: _text(item, ("name", "description", "summary"), label="entity")),
        **kwargs,
    )


async def rerank_tools(
    reranker: Reranker,
    *,
    query: str,
    candidates: Sequence[T],
    text_fn: Callable[[T], str] | None = None,
    **kwargs: Any,
) -> RerankResponse[T]:
    return await reranker.rerank(
        query=query,
        candidates=candidates,
        text_fn=text_fn or (lambda item: _text(item, ("name", "description"), label="tool")),
        **kwargs,
    )


async def rerank_memories(
    reranker: Reranker,
    *,
    query: str,
    candidates: Sequence[T],
    text_fn: Callable[[T], str] | None = None,
    **kwargs: Any,
) -> RerankResponse[T]:
    return await reranker.rerank(
        query=query,
        candidates=candidates,
        text_fn=text_fn
        or (lambda item: _text(item, ("content", "text", "summary"), label="memory")),
        **kwargs,
    )
