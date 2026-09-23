from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from jev_rankkit.backends import BackendCandidate, JevBackend, ModelRequest
from jev_rankkit.config import RetryConfig
from jev_rankkit.errors import (
    AuthenticationError,
    BackendError,
    CapabilityError,
    ContextLimitError,
    OutputValidationError,
    RateLimitError,
)
from jev_rankkit.prompts import RerankPrompt


@dataclass
class Response:
    status_code: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        return self.body


@dataclass
class RawResponse(Response):
    text: str = ""


class Transport:
    def __init__(self, replies: list[Response | BaseException]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout_s: float
    ) -> Response:
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout_s": timeout_s})
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def aclose(self) -> None:
        return None


def request(mode: str = "listwise") -> ModelRequest:
    return ModelRequest(
        query="vector search",
        candidates=(BackendCandidate("c0", "first"), BackendCandidate("c1", "second")),
        prompt=RerankPrompt(criteria={"relevance": "Directly helps the query."}),
        mode=mode,  # type: ignore[arg-type]
    )


def valid_body() -> dict[str, Any]:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "c0": {"type": "noul", "noul": 0.9},
            "c1": {"type": "noul", "noul": 0.1},
        },
        "usage": {"input_tokens": 100, "output_tokens": 2},
    }


@pytest.mark.asyncio
async def test_jev_structured_listwise_request_and_usage() -> None:
    transport = Transport([Response(200, valid_body())])
    backend = JevBackend(api_key="test", transport=transport, retry=RetryConfig(max_attempts=1))
    response = await backend.score(request())
    payload = transport.calls[0]["json"]
    assert payload["model"] == "jev-1.13.0"
    assert len(payload["state"]["UNTRUSTED CANDIDATE CONTENT"]) == 2
    assert payload["questions"]["c0"]["instructions"]["TARGET ORDINAL"] == 0
    assert [score.score for score in response.scores] == [0.9, 0.1]
    assert response.statistics.usage.input_tokens == 100
    assert response.statistics.usage.cost_usd is None


@pytest.mark.asyncio
async def test_jev_pointwise_keeps_candidate_content_out_of_shared_state() -> None:
    transport = Transport([Response(200, valid_body())])
    backend = JevBackend(api_key="test", transport=transport, retry=RetryConfig(max_attempts=1))
    await backend.score(request("pointwise"))
    payload = transport.calls[0]["json"]
    assert "UNTRUSTED CANDIDATE CONTENT" not in payload["state"]
    assert "first" in str(payload["questions"]["c0"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda body: body.update(answers={"c0": {"type": "noul", "noul": 0.9}}), "missing"),
        (
            lambda body: body["answers"].update(extra={"type": "noul", "noul": 0.2}),
            "unexpected",
        ),
        (lambda body: body["answers"]["c0"].update(noul=2.0), "outside"),
        (lambda body: body["answers"]["c0"].pop("noul"), "missing a numeric"),
        (lambda body: body["answers"]["c0"].update(type="choice"), "answer type"),
        (lambda body: body.update(answers=[]), "answers object"),
    ],
)
@pytest.mark.asyncio
async def test_jev_invalid_structured_outputs(mutation, message: str) -> None:  # type: ignore[no-untyped-def]
    body = valid_body()
    mutation(body)
    backend = JevBackend(
        api_key="test",
        transport=Transport([Response(200, body)]),
        retry=RetryConfig(max_attempts=1),
    )
    with pytest.raises(OutputValidationError, match=message):
        await backend.score(request())


@pytest.mark.asyncio
async def test_jev_partial_output_must_be_opted_in() -> None:
    body = valid_body()
    del body["answers"]["c1"]
    backend = JevBackend(
        api_key="test",
        transport=Transport([Response(200, body)]),
        retry=RetryConfig(max_attempts=1),
    )
    response = await backend.score(
        ModelRequest(
            query="vector search",
            candidates=request().candidates,
            prompt=request().prompt,
            mode="listwise",
            allow_partial=True,
        )
    )
    assert response.missing_candidate_ids == ("c1",)


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, AuthenticationError),
        (429, RateLimitError),
        (413, ContextLimitError),
        (529, BackendError),
    ],
)
@pytest.mark.asyncio
async def test_jev_provider_error_mapping(status: int, error: type[Exception]) -> None:
    backend = JevBackend(
        api_key="test",
        transport=Transport([Response(status, {"secret": "must not leak"})]),
        retry=RetryConfig(max_attempts=1),
    )
    with pytest.raises(error) as raised:
        await backend.score(request())
    assert "secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_jev_retries_rate_limits_then_succeeds() -> None:
    transport = Transport([Response(429, {}), Response(200, valid_body())])
    backend = JevBackend(
        api_key="test",
        transport=transport,
        retry=RetryConfig(max_attempts=2, initial_backoff_s=0, max_backoff_s=0),
    )
    result = await backend.score(request())
    assert result.statistics.attempts == 2
    assert not result.statistics.usage.tokens_reported
    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_jev_timeout_is_package_error() -> None:
    backend = JevBackend(
        api_key="test",
        transport=Transport([TimeoutError("secret timeout")]),
        retry=RetryConfig(max_attempts=1),
    )
    with pytest.raises(BackendError, match="timed out") as raised:
        await backend.score(request())
    assert "secret" not in str(raised.value)


def test_jev_rejects_unsupported_reasoning_or_explanations() -> None:
    with pytest.raises(CapabilityError):
        JevBackend(api_key="test", reasoning_level="high")
    backend = JevBackend(api_key="test", transport=Transport([]))
    with pytest.raises(CapabilityError):
        backend._payload(
            ModelRequest(
                query="q",
                candidates=request().candidates,
                prompt=request().prompt,
                include_reasoning=True,
            )
        )


@pytest.mark.asyncio
async def test_cancellation_does_not_start_retry() -> None:
    transport = Transport([asyncio.CancelledError()])
    backend = JevBackend(api_key="test", transport=transport)
    with pytest.raises(asyncio.CancelledError):
        await backend.score(request())
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_duplicate_raw_json_answer_ids_are_rejected() -> None:
    raw = (
        '{"model":"jev-1.13.0","answers":{'
        '"c0":{"type":"noul","noul":0.5},'
        '"c0":{"type":"noul","noul":0.7},'
        '"c1":{"type":"noul","noul":0.1}}}'
    )
    backend = JevBackend(
        api_key="test",
        transport=Transport([RawResponse(200, None, text=raw)]),
        retry=RetryConfig(max_attempts=1),
    )
    with pytest.raises(OutputValidationError, match="duplicate JSON keys"):
        await backend.score(request())


def test_retry_after_parser_is_bounded_by_policy_at_call_site() -> None:
    assert JevBackend._retry_after_seconds({"Retry-After": "3"}) == 3.0
    assert JevBackend._retry_after_seconds({"Retry-After": "invalid"}) is None


def test_backend_repr_redacts_explicit_api_key_and_transport() -> None:
    backend = JevBackend(api_key="SYNTHETIC_SECRET", transport=Transport([]))
    assert "SYNTHETIC_SECRET" not in repr(backend)
    assert "Transport" not in repr(backend)


@pytest.mark.asyncio
async def test_long_retry_after_stops_instead_of_retrying_early() -> None:
    transport = Transport([Response(429, {}, {"Retry-After": "30"})])
    backend = JevBackend(
        api_key="test",
        transport=transport,
        retry=RetryConfig(max_attempts=3, max_backoff_s=1),
    )
    with pytest.raises(RateLimitError) as caught:
        await backend.score(request())
    assert len(transport.calls) == 1
    assert caught.value.details.safe_context == {"attempts": 1}


@pytest.mark.asyncio
async def test_jev_rejects_oversized_raw_response() -> None:
    backend = JevBackend(
        api_key="test",
        transport=Transport([RawResponse(200, {}, text="x" * 2_000_001)]),
        retry=RetryConfig(max_attempts=1),
    )
    with pytest.raises(OutputValidationError, match="size limit"):
        await backend.score(request())
