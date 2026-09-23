from __future__ import annotations

import hashlib
import json
from datetime import datetime
from types import MappingProxyType
from typing import Any


def cache_key(namespace: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_canonical_default,
    )
    return f"{namespace}:{hashlib.sha256(encoded.encode()).hexdigest()}"


def _canonical_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, MappingProxyType):
        return dict(value)
    raise TypeError(f"unsupported cache-key value: {type(value).__name__}")
