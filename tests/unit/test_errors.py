"""Every error path, seen through the real OpenAI SDK running in-process against Relay.

TestClient is an httpx.Client, so the SDK can use it as its transport: no network, but
the SDK's own parsing decides which exception class a client would get.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import openai
import pytest
from fastapi.testclient import TestClient

from relay.backends.base import (
    BackendError,
    BackendHTTPError,
    BackendStreamError,
    BackendTimeout,
    BackendUnavailable,
)
from tests.unit.fakes import FakeBackend, FakeStream, app_with_fake

MESSAGES = [{"role": "user", "content": "hi"}]


@contextmanager
def sdk(backend: FakeBackend) -> Iterator[tuple[openai.OpenAI, TestClient]]:
    # raise_server_exceptions=False: see the 500 response instead of the raw exception.
    with TestClient(app_with_fake(backend), raise_server_exceptions=False) as http:
        client = openai.OpenAI(
            base_url="http://testserver/v1", api_key="x", http_client=http, max_retries=0
        )
        yield client, http


def call(backend: FakeBackend, *, stream: bool = False, **kwargs):
    with sdk(backend) as (client, _):
        result = client.chat.completions.create(
            model=kwargs.pop("model", "m"),
            messages=kwargs.pop("messages", MESSAGES),
            stream=stream,
            **kwargs,
        )
        return list(result) if stream else result


def test_validation_error_is_bad_request_with_param() -> None:
    with pytest.raises(openai.BadRequestError) as excinfo:
        call(FakeBackend(), messages=[{"role": "robot", "content": "hi"}])
    assert excinfo.value.body["param"] == "messages.0.role"


def test_unknown_model_is_not_found() -> None:
    with pytest.raises(openai.NotFoundError) as excinfo:
        call(FakeBackend(), model="gpt-4o")
    assert excinfo.value.code == "model_not_found"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("error", "exc_class", "status", "error_type"),
    [
        (BackendUnavailable("x", "down"), openai.InternalServerError, 502, "backend_unavailable"),
        (BackendTimeout("x", "slow"), openai.InternalServerError, 504, "backend_timeout"),
    ],
)
def test_backend_failures_before_streaming(
    error: BackendError, exc_class: type, status: int, error_type: str, stream: bool
) -> None:
    with pytest.raises(exc_class) as excinfo:
        call(FakeBackend(error=error), stream=stream)
    assert excinfo.value.status_code == status
    assert excinfo.value.type == error_type


def test_openai_shaped_backend_error_passes_through() -> None:
    body = b'{"error": {"message": "slow down", "type": "rate_limit", "code": "busy"}}'
    error = BackendHTTPError("fake", 429, body, "application/json")
    with pytest.raises(openai.RateLimitError) as excinfo:
        call(FakeBackend(error=error))
    assert excinfo.value.code == "busy" and "slow down" in excinfo.value.message


@pytest.mark.parametrize(
    "body", [b"<html><h1>502 Bad Gateway</h1></html>", b"upstream went away", b"[1, 2]"]
)
def test_non_openai_backend_error_body_is_wrapped(body: bytes) -> None:
    error = BackendHTTPError("fake", 502, body, "text/html")
    with pytest.raises(openai.InternalServerError) as excinfo:
        call(FakeBackend(error=error))
    assert excinfo.value.status_code == 502
    assert excinfo.value.type == "backend_error"
    assert "html" not in excinfo.value.message.lower()


def test_mid_stream_failure_raises_api_error_while_iterating() -> None:
    chunk = b'data: {"choices": [{"index": 0, "delta": {"content": "Hel"}}]}\n\n'
    failure = BackendStreamError("fake", "lost", timeout=False)
    with pytest.raises(openai.APIError) as excinfo:
        call(FakeBackend(stream=FakeStream([chunk], fail_with=failure)), stream=True)
    assert "connection lost mid-stream" in excinfo.value.message


class BuggyBackend(FakeBackend):
    async def chat(self, body: bytes):
        raise RuntimeError("secret internal detail")


def test_bug_in_relay_is_500_without_leaking_details() -> None:
    with pytest.raises(openai.InternalServerError) as excinfo:
        call(BuggyBackend())
    assert excinfo.value.status_code == 500
    assert excinfo.value.type == "server_error"
    assert "secret" not in str(excinfo.value.body)


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [("GET", "/v1/nope", 404), ("GET", "/v1/chat/completions", 405)],
)
def test_unknown_paths_and_methods_are_openai_errors(method: str, path: str, status: int) -> None:
    with sdk(FakeBackend()) as (_, http):
        resp = http.request(method, path)
    assert resp.status_code == status
    assert set(resp.json()["error"]) == {"message", "type", "param", "code"}
