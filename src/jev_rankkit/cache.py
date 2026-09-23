"""Async cache protocol and bounded in-memory LRU implementation."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class CacheRecord:
    value: Any
    created_at: float
    schema_version: str = "1"


@runtime_checkable
class CacheBackend(Protocol):
    async def get(self, key: str) -> CacheRecord | None: ...

    async def set(
        self, key: str, value: CacheRecord, *, ttl_seconds: float | None = None
    ) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def clear(self, *, namespace: str | None = None) -> None: ...


class MemoryCache:
    """Loop-safe within one event loop; values never contain candidate objects by convention."""

    def __init__(
        self, *, max_entries: int = 1024, default_ttl_seconds: float | None = 3600
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._default_ttl = default_ttl_seconds
        self._items: OrderedDict[str, tuple[float | None, CacheRecord]] = OrderedDict()
        self._lock = threading.RLock()

    async def get(self, key: str) -> CacheRecord | None:
        now = time.monotonic()
        with self._lock:
            stored = self._items.get(key)
            if stored is None:
                return None
            expires_at, record = stored
            if expires_at is not None and expires_at <= now:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return record

    async def set(self, key: str, value: CacheRecord, *, ttl_seconds: float | None = None) -> None:
        ttl = self._default_ttl if ttl_seconds is None else ttl_seconds
        expires_at = None if ttl is None else time.monotonic() + ttl
        with self._lock:
            self._items[key] = (expires_at, value)
            self._items.move_to_end(key)
            while len(self._items) > self._max_entries:
                self._items.popitem(last=False)

    async def delete(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)

    async def clear(self, *, namespace: str | None = None) -> None:
        with self._lock:
            if namespace is None:
                self._items.clear()
                return
            prefix = f"{namespace}:"
            for key in tuple(self._items):
                if key.startswith(prefix):
                    del self._items[key]
