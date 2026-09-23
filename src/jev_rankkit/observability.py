"""Payload-safe, bounded tracing hooks for ranking operations."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, cast


class TraceKind(StrEnum):
    MODEL_START = "model_start"
    MODEL_END = "model_end"
    MODEL_ERROR = "model_error"
    STAGE_END = "stage_end"
    RERANK_END = "rerank_end"


@dataclass(frozen=True, slots=True)
class TraceEvent:
    kind: TraceKind
    request_id: str
    stage: str | None = None
    candidate_count: int | None = None
    latency_ms: float | None = None
    attributes: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


class Observer(Protocol):
    def on_event(self, event: TraceEvent) -> Awaitable[None] | None: ...


class ObserverDispatcher:
    """Nonblocking producer with one bounded consumer; observer failures are isolated."""

    def __init__(self, observer: Observer, *, capacity: int = 128) -> None:
        if capacity < 1:
            raise ValueError("observer queue capacity must be positive")
        self.observer = observer
        self.queue: asyncio.Queue[TraceEvent] = asyncio.Queue(capacity)
        self.dropped = 0
        self.errors = 0
        self._worker: asyncio.Task[None] | None = None

    def emit(self, event: TraceEvent) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="jev-rankkit-observer")
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1

    async def _run(self) -> None:
        while True:
            event = await self.queue.get()
            try:
                callback = self.observer.on_event
                if inspect.iscoroutinefunction(callback):
                    result = callback(event)
                else:
                    result = await asyncio.to_thread(cast(Any, callback), event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                self.errors += 1
            finally:
                self.queue.task_done()

    async def aclose(self, *, timeout_s: float = 1.0) -> None:
        if self._worker is None:
            return
        try:
            await asyncio.wait_for(self.queue.join(), timeout=timeout_s)
        except TimeoutError:
            self.dropped += self.queue.qsize()
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None
