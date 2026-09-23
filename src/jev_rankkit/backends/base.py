"""Provider-independent model and embedding backend contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from ..prompts import RerankPrompt
from ..types import RequestStatistics

BackendMode = Literal["pointwise", "listwise"]


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    pointwise: bool = True
    listwise: bool = False
    explanations: bool = False
    reasoning_levels: tuple[str, ...] = ()
    max_batch_size: int | None = None
    max_context_tokens: int | None = None
    usage_reporting: bool = False


@dataclass(frozen=True, slots=True)
class BackendCandidate:
    candidate_id: str
    text: str


@dataclass(frozen=True, slots=True)
class ModelRequest:
    query: str
    candidates: tuple[BackendCandidate, ...]
    prompt: RerankPrompt
    mode: BackendMode = "pointwise"
    include_reasoning: bool = False
    reasoning_level: str | None = None
    allow_partial: bool = False


@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate_id: str
    score: float
    reasoning: str | None = None


@dataclass(frozen=True, slots=True)
class ModelResponse:
    scores: tuple[CandidateScore, ...]
    statistics: RequestStatistics = field(default_factory=RequestStatistics)
    resolved_model: str | None = None
    missing_candidate_ids: tuple[str, ...] = ()


@runtime_checkable
class ModelBackend(Protocol):
    @property
    def backend_id(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def capabilities(self) -> BackendCapabilities: ...

    @property
    def cache_identity(self) -> str: ...

    async def score(self, request: ModelRequest) -> ModelResponse: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class EmbeddingBackend(Protocol):
    @property
    def backend_id(self) -> str: ...

    @property
    def cache_identity(self) -> str: ...

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    async def aclose(self) -> None: ...
