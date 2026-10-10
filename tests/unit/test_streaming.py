import json
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi.testclient import TestClient

from relay.api.chat import _relay_events
from relay.backends.sse import iter_sse_events
from tests.unit.fakes import FakeStream, app_with_transport

REQUEST = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}


def event(delta: str) -> bytes:
    choice = {"index": 0, "delta": {"content": delta}}
    chunk = {"object": "chat.completion.chunk", "choices": [choice]}
    return f"data: {json.dumps(chunk)}\n\n".encode()


EVENTS = [event("Hel"), event("lo"), b"data: [DONE]\n\n"]
WIRE = b"".join(EVENTS)


def split_awkwardly(data: bytes) -> list[bytes]:
    """TCP-like chunks that cut events in the middle, including inside the JSON."""
    cuts = [7, 30, 31, len(EVENTS[0]) + 3, len(data) - 5]
    return [data[a:b] for a, b in zip([0, *cuts], [*cuts, len(data)], strict=True)]


async def chunks(parts: list[bytes], fail_after: int | None = None) -> AsyncIterator[bytes]:
    for i, part in enumerate(parts):
        if fail_after is not None and i == fail_after:
            raise httpx.ReadError("connection reset by peer")
        yield part


def sse_backend(parts: list[bytes], fail_after: int | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=chunks(parts, fail_after),
        )

    return handler


def post_stream(handler) -> tuple[httpx.Response, bytes]:
    app = app_with_transport(httpx.MockTransport(handler))
    with (
        TestClient(app) as client,
        client.stream("POST", "/v1/chat/completions", json=REQUEST) as resp,
    ):
        return resp, b"".join(resp.iter_bytes())


async def collect(it: AsyncIterator[bytes]) -> list[bytes]:
    return [x async for x in it]


async def test_events_are_reassembled_from_split_chunks() -> None:
    assert await collect(iter_sse_events(chunks(split_awkwardly(WIRE)))) == EVENTS


async def test_crlf_and_missing_final_blank_line_are_handled() -> None:
    raw = [b"data: a\r\n\r\ndata: b"]
    assert await collect(iter_sse_events(chunks(raw))) == [b"data: a\n\n", b"data: b\n\n"]


def test_stream_is_forwarded_as_sse() -> None:
    resp, body = post_stream(sse_backend(split_awkwardly(WIRE)))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    assert body == WIRE  # byte-identical events, ending with [DONE]


def test_mid_stream_failure_ends_with_a_clean_error_event() -> None:
    parts = split_awkwardly(WIRE)
    # Fail after 4 chunks: event 1 is complete, event 2 has only partly arrived.
    resp, body = post_stream(sse_backend(parts, fail_after=4))
    assert resp.status_code == 200  # headers were already sent; status cannot change
    events = body.split(b"\n\n")[:-1]
    assert events[0] + b"\n\n" == EVENTS[0]  # the complete event got through
    last = json.loads(events[-1].removeprefix(b"data: "))
    assert last["error"]["type"] == "backend_unavailable"  # no half event before it
    assert b"[DONE]" not in body


def test_backend_error_before_streaming_keeps_its_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "context too long"}})

    resp, body = post_stream(handler)
    assert resp.status_code == 400
    assert json.loads(body)["error"]["message"] == "context too long"


@pytest.mark.parametrize(
    ("exc", "status"),
    [(httpx.ConnectError("refused"), 502), (httpx.ConnectTimeout("slow"), 504)],
)
def test_backend_unreachable_before_streaming_is_a_json_error(exc: Exception, status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    resp, body = post_stream(handler)
    assert resp.status_code == status
    assert json.loads(body)["error"]["type"].startswith("backend_")


async def test_upstream_is_closed_when_client_stops_reading() -> None:
    # Simulates a client disconnect: the generator is closed after one event.
    upstream = FakeStream(list(EVENTS))
    relay = _relay_events(upstream)
    assert await anext(relay) == EVENTS[0]
    assert not upstream.closed
    await relay.aclose()  # what Starlette does when the client goes away
    assert upstream.closed  # -> the connection to the engine is released
