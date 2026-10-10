"""OpenAICompatBackend: HTTP details become Relay's own backend errors."""

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from relay.backends import ENGINES, Backend, LlamaCppBackend, VLLMBackend
from relay.backends.base import (
    BackendHTTPError,
    BackendStreamError,
    BackendTimeout,
    BackendUnavailable,
)
from relay.backends.openai_compat import OpenAICompatBackend
from tests.unit.fakes import FakeBackend

BODY = b'{"model":"m","messages":[{"role":"user","content":"hi"}]}'


def backend_with(handler) -> OpenAICompatBackend:
    return OpenAICompatBackend("b1", "http://engine", transport=httpx.MockTransport(handler))


def raising(exc: Exception):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


async def test_chat_returns_response_and_posts_original_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "x"})

    result = await backend_with(handler).chat(BODY)
    assert result.status_code == 200
    assert json.loads(result.content) == {"id": "x"}
    assert seen[0].url.path == "/v1/chat/completions"
    assert seen[0].content == BODY


async def test_error_status_becomes_backend_http_error_with_body() -> None:
    backend = backend_with(lambda r: httpx.Response(400, json={"error": {"message": "bad"}}))
    with pytest.raises(BackendHTTPError) as excinfo:
        await backend.chat(BODY)
    assert excinfo.value.status_code == 400
    assert json.loads(excinfo.value.content)["error"]["message"] == "bad"
    assert excinfo.value.backend == "b1"


@pytest.mark.parametrize("method", ["chat", "stream"])
@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.ConnectError("refused"), BackendUnavailable),
        (httpx.RemoteProtocolError("reset"), BackendUnavailable),
        (httpx.ConnectTimeout("slow"), BackendTimeout),
        (httpx.ReadTimeout("slow"), BackendTimeout),
    ],
)
async def test_transport_failures_are_translated(method: str, exc: Exception, expected) -> None:
    backend = backend_with(raising(exc))
    with pytest.raises(expected):
        await getattr(backend, method)(BODY)


async def test_stream_error_status_raises_before_streaming() -> None:
    backend = backend_with(lambda r: httpx.Response(503, text="loading"))
    with pytest.raises(BackendHTTPError) as excinfo:
        await backend.stream(BODY)
    assert excinfo.value.status_code == 503


async def _broken_body() -> AsyncIterator[bytes]:
    yield b"data: {}\n\n"
    raise httpx.ReadError("connection reset")


async def test_stream_failure_after_start_is_a_stream_error() -> None:
    backend = backend_with(lambda r: httpx.Response(200, content=_broken_body()))
    stream = await backend.stream(BODY)
    received = []
    with pytest.raises(BackendStreamError) as excinfo:
        async for event in stream.events():
            received.append(event)
    assert received == [b"data: {}\n\n"]
    assert excinfo.value.timeout is False


@pytest.mark.parametrize(
    ("handler", "healthy"),
    [
        (lambda r: httpx.Response(200, json={"status": "ok"}), True),
        (lambda r: httpx.Response(503, json={"status": "loading model"}), False),
        (raising(httpx.ConnectError("refused")), False),
    ],
)
async def test_health(handler, healthy: bool) -> None:
    assert await backend_with(handler).health() is healthy


def test_engines_registry_and_protocol_conformance() -> None:
    assert {"llamacpp": LlamaCppBackend, "vllm": VLLMBackend} == ENGINES
    assert LlamaCppBackend.engine == "llamacpp" and VLLMBackend.engine == "vllm"
    # Structural typing: both the real backend and the fake satisfy Backend.
    for candidate in (LlamaCppBackend("x", "http://x"), FakeBackend()):
        backend: Backend = candidate
        assert callable(backend.chat) and callable(backend.stream)
