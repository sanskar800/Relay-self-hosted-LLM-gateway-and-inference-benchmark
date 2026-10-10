"""The gateway against an in-memory fake backend: no HTTP below Relay at all."""

import json

from fastapi.testclient import TestClient

from relay.backends.base import (
    BackendHTTPError,
    BackendResponse,
    BackendStreamError,
    BackendTimeout,
    BackendUnavailable,
)
from tests.unit.fakes import FakeBackend, FakeStream, app_with_fake

REQUEST = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def post(backend: FakeBackend, **extra) -> tuple[int, bytes]:
    with TestClient(app_with_fake(backend)) as client:
        resp = client.post("/v1/chat/completions", json={**REQUEST, **extra})
        return resp.status_code, resp.content


def test_non_streaming_response_is_returned_as_is() -> None:
    backend = FakeBackend(response=BackendResponse(200, b'{"id":"abc"}', "application/json"))
    status, body = post(backend)
    assert (status, body) == (200, b'{"id":"abc"}')
    assert json.loads(backend.requests[0]) == REQUEST


def test_streaming_events_are_relayed_and_stream_closed() -> None:
    stream = FakeStream([b"data: 1\n\n", b"data: 2\n\n", b"data: [DONE]\n\n"])
    status, body = post(FakeBackend(stream=stream), stream=True)
    assert status == 200
    assert body == b"data: 1\n\ndata: 2\n\ndata: [DONE]\n\n"
    assert stream.closed


def test_stream_error_after_start_becomes_error_event() -> None:
    failure = BackendStreamError("fake", "lost", timeout=True)
    stream = FakeStream([b"data: 1\n\n"], fail_with=failure)
    status, body = post(FakeBackend(stream=stream), stream=True)
    assert status == 200
    first, error = body.split(b"\n\n")[:2]
    assert first == b"data: 1"
    assert json.loads(error.removeprefix(b"data: "))["error"]["type"] == "backend_timeout"
    assert stream.closed


def test_backend_errors_map_to_http_statuses() -> None:
    cases = [
        (BackendUnavailable("fake", "down"), 502),
        (BackendTimeout("fake", "slow"), 504),
        (BackendHTTPError("fake", 429, b'{"error":{}}', "application/json"), 429),
    ]
    for stream in (False, True):
        for error, status in cases:
            assert post(FakeBackend(error=error), stream=stream)[0] == status


def test_backend_is_closed_on_shutdown() -> None:
    backend = FakeBackend()
    with TestClient(app_with_fake(backend)):
        assert not backend.closed
    assert backend.closed
