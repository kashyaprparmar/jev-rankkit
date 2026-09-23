"""Object-preserving candidate preparation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Generic, TypeAlias, TypeVar, cast

from .config import RerankerConfig, TruncationPolicy
from .errors import ConfigurationError, ProjectionError

T = TypeVar("T")
MetadataScalar: TypeAlias = str | bool | int | float | datetime | None
MetadataValue: TypeAlias = (
    MetadataScalar | tuple["MetadataValue", ...] | Mapping[str, "MetadataValue"]
)


@dataclass(frozen=True, slots=True)
class CandidateAdapter(Generic[T]):
    text_fn: Callable[[T], str]
    metadata_fn: Callable[[T], Mapping[str, MetadataValue]] | None = None
    id_fn: Callable[[T], str] | None = None
    version: str = "1"


@dataclass(frozen=True, slots=True)
class CandidateView(Generic[T]):
    item: T
    input_index: int
    occurrence_id: str
    candidate_id: str | None
    text: str
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _freeze_metadata(value: Any, *, depth: int = 0) -> MetadataValue:
    if depth > 8:
        raise ProjectionError("metadata nesting exceeds the maximum depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProjectionError("metadata numbers must be finite")
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ProjectionError("metadata datetimes must be timezone-aware")
        return value.astimezone(UTC)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item, depth=depth + 1) for item in value)
    if isinstance(value, Mapping):
        frozen: dict[str, MetadataValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProjectionError("metadata keys must be strings")
            frozen[key] = _freeze_metadata(item, depth=depth + 1)
        return MappingProxyType(frozen)
    raise ProjectionError(f"unsupported metadata value type: {type(value).__name__}")


def _truncate(text: str, config: RerankerConfig) -> str:
    limit = config.max_candidate_chars
    if len(text) <= limit:
        return text
    if config.truncation is TruncationPolicy.ERROR:
        raise ProjectionError(f"candidate text exceeds max_candidate_chars={limit}")
    if config.truncation is TruncationPolicy.HEAD:
        return text[:limit]
    left = limit // 2
    return text[:left] + text[-(limit - left) :]


def prepare_candidates(
    candidates: Sequence[T],
    *,
    config: RerankerConfig,
    text_fn: Callable[[T], str] | None = None,
    metadata_fn: Callable[[T], Mapping[str, MetadataValue]] | None = None,
    id_fn: Callable[[T], str] | None = None,
    eligible_fn: Callable[[T], bool] | None = None,
    adapter: CandidateAdapter[T] | None = None,
) -> tuple[CandidateView[T], ...]:
    if len(candidates) > config.max_candidates:
        raise ConfigurationError(f"candidate count exceeds max_candidates={config.max_candidates}")
    if adapter is not None and any(fn is not None for fn in (text_fn, metadata_fn, id_fn)):
        raise ConfigurationError("adapter cannot be combined with projection callbacks")
    if adapter is not None:
        text_fn, metadata_fn, id_fn = adapter.text_fn, adapter.metadata_fn, adapter.id_fn

    views: list[CandidateView[T]] = []
    for input_index, item in enumerate(tuple(candidates)):
        if eligible_fn is not None and not eligible_fn(item):
            continue
        text: str
        if text_fn is None:
            if not isinstance(item, str):
                raise ProjectionError("non-string candidates require text_fn or CandidateAdapter")
            text = cast(str, item)
        else:
            text = text_fn(item)
        if not isinstance(text, str):
            raise ProjectionError("text_fn must return str")
        text = _truncate(text, config)
        candidate_id = id_fn(item) if id_fn is not None else None
        if candidate_id is not None and (not isinstance(candidate_id, str) or not candidate_id):
            raise ProjectionError("id_fn must return a non-empty str")
        raw_metadata = metadata_fn(item) if metadata_fn is not None else {}
        frozen_metadata = _freeze_metadata(raw_metadata)
        if not isinstance(frozen_metadata, Mapping):
            raise ProjectionError("metadata_fn must return a mapping")
        views.append(
            CandidateView(
                item=item,
                input_index=input_index,
                occurrence_id=f"c{input_index:08d}",
                candidate_id=candidate_id,
                text=text,
                metadata=frozen_metadata,
            )
        )
    return tuple(views)
