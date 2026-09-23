"""Optional local embedding backend; model weights load only on first embed call."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from ..errors import CapabilityError


class SentenceTransformerBackend:
    def __init__(self, model_name: str, *, revision: str | None = None) -> None:
        if not model_name:
            raise ValueError("model_name cannot be empty")
        self.model_name = model_name
        self.revision = revision
        self._model: Any = None
        self._lock = asyncio.Lock()

    @property
    def backend_id(self) -> str:
        return "sentence-transformers"

    @property
    def cache_identity(self) -> str:
        return f"sentence-transformers:{self.model_name}:{self.revision or 'unpinned'}"

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return await self._embed(texts, task="generic")

    async def embed_query(self, query: str) -> Sequence[float]:
        vectors = await self._embed((query,), task="query")
        return vectors[0]

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return await self._embed(texts, task="document")

    async def _embed(self, texts: Sequence[str], *, task: str) -> list[list[float]]:
        if not texts:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._encode, tuple(texts), task)

    def _encode(self, texts: tuple[str, ...], task: str) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise CapabilityError(
                    "Install jev-rankkit[embeddings] to use SentenceTransformerBackend"
                ) from exc
            self._model = SentenceTransformer(
                self.model_name, revision=self.revision, trust_remote_code=False
            )
        method = (
            getattr(self._model, "encode_query", self._model.encode)
            if task == "query"
            else getattr(self._model, "encode_document", self._model.encode)
            if task == "document"
            else self._model.encode
        )
        vectors = method(list(texts), convert_to_numpy=True)
        return [[float(value) for value in vector] for vector in vectors]

    async def aclose(self) -> None:
        self._model = None
