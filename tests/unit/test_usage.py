import json

import pytest
from fastapi.testclient import TestClient

from relay.accounting.usage import (
    StreamUsage,
    UsageRecord,
    request_stream_usage,
    usage_from_completion,
)
from relay.backends.base import BackendResponse, BackendStreamError
from relay.main import create_app
from tests.unit.fakes import FakeBackend, FakeStream, make_config


def sse(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def content_chunk(text: str) -> bytes:
    return sse({"choices": [{"index": 0, "delta": {"content": text}}]})


ROLE_CHUNK = sse({"choices": [{"index": 0, "delta": {"role": "assistant", "content": None}}]})
FINAL_CHUNK = sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
USAGE_CHUNK = sse({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2}})
DONE = b"data: [DONE]\n\n"
ENGINE_STREAM = [
    ROLE_CHUNK,
    content_chunk("Hi"),
    content_chunk("!"),
    FINAL_CHUNK,
    USAGE_CHUNK,
    DONE,
]


# --- pure functions -------------------------------------------------------------------


def test_usage_from_completion() -> None:
    body = json.dumps({"usage": {"prompt_tokens": 5, "completion_tokens": 3}}).encode()
    assert usage_from_completion(body) == (5, 3, "engine")
    assert usage_from_completion(b'{"choices": []}') == (None, None, "missing")
    assert usage_from_completion(b"not json") == (None, None, "missing")


def test_request_stream_usage_adds_option_and_keeps_everything_else() -> None:
    body = json.dumps(
        {"model": "m", "stream": True, "seed": 7, "stream_options": {"x_custom": 1}}
    ).encode()
    data = json.loads(request_stream_usage(body))
    assert data["stream_options"] == {"x_custom": 1, "include_usage": True}
    assert data["seed"] == 7 and data["model"] == "m" and data["stream"] is True


def test_record_total_only_when_both_counts_known() -> None:
    assert UsageRecord("m", "b", True, 200, 3, 4, "engine").total_tokens == 7
    assert UsageRecord("m", "b", True, 200, None, 4, "estimated").total_tokens is None


@pytest.mark.parametrize("forward", [False, True])
def test_stream_usage_reads_report_and_hides_it_unless_requested(forward: bool) -> None:
    usage = StreamUsage(forward_usage_chunk=forward)
    forwarded = [e for e in ENGINE_STREAM if usage.observe(e)]
    assert usage.result() == (12, 2, "engine")
    assert usage.content_chunks == 2  # role chunk and final chunk carry no content
    assert (USAGE_CHUNK in forwarded) is forward
    assert forwarded[-1] == DONE


def test_stream_usage_falls_back_to_estimate_then_missing() -> None:
    usage = StreamUsage(forward_usage_chunk=False)
    for e in [ROLE_CHUNK, content_chunk("a"), content_chunk("b"), content_chunk("c")]:
        usage.observe(e)
    assert usage.result() == (None, 3, "estimated")
    assert StreamUsage(forward_usage_chunk=False).result() == (None, None, "missing")


def test_non_json_events_are_forwarded_untouched() -> None:
    usage = StreamUsage(forward_usage_chunk=False)
    for event in [b": keep-alive comment\n\n", b"data: [DONE]\n\n", b"data: {broken\n\n"]:
        assert usage.observe(event)


# --- through the gateway --------------------------------------------------------------

REQUEST = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def run(backend: FakeBackend, raw: bytes | None = None, **extra) -> tuple[bytes, list[UsageRecord]]:
    records: list[UsageRecord] = []
    app = create_app(
        config=make_config("m", backend.name),
        backends={backend.name: backend},
        usage_sink=records.append,
    )
    with TestClient(app) as client:
        if raw is None:
            resp = client.post("/v1/chat/completions", json={**REQUEST, **extra})
        else:
            headers = {"Content-Type": "application/json"}
            resp = client.post("/v1/chat/completions", content=raw, headers=headers)
    return resp.content, records


def test_non_streaming_usage_is_recorded() -> None:
    body = json.dumps({"usage": {"prompt_tokens": 9, "completion_tokens": 4}}).encode()
    backend = FakeBackend(response=BackendResponse(200, body, "application/json"))
    _, records = run(backend)
    assert records == [UsageRecord("m", backend.name, False, 200, 9, 4, "engine")]


def test_stream_without_include_usage_asks_engine_and_hides_the_chunk() -> None:
    backend = FakeBackend(stream=FakeStream(list(ENGINE_STREAM)))
    out, records = run(backend, stream=True)
    sent_upstream = json.loads(backend.requests[0])
    assert sent_upstream["stream_options"] == {"include_usage": True}
    assert b'"usage"' not in out  # client gets exactly what it asked for
    assert out.endswith(DONE)
    assert records[0].source == "engine"
    assert (records[0].prompt_tokens, records[0].completion_tokens) == (12, 2)


def test_stream_with_include_usage_forwards_original_bytes_and_the_chunk() -> None:
    backend = FakeBackend(stream=FakeStream(list(ENGINE_STREAM)))
    # Unusual spacing on purpose: re-serialising would change these bytes.
    raw = (
        b'{ "model" : "m",  "messages":[{"role":"user","content":"hi"}],'
        b' "stream":true, "stream_options": {"include_usage": true} }'
    )
    out, records = run(backend, raw=raw)
    assert backend.requests[0] == raw  # forwarded byte-for-byte
    assert USAGE_CHUNK in out
    assert records[0].source == "engine"


def test_stream_broken_before_usage_is_recorded_as_estimate() -> None:
    failure = BackendStreamError("fake", "lost", timeout=False)
    stream = FakeStream([ROLE_CHUNK, content_chunk("a"), content_chunk("b")], fail_with=failure)
    _, records = run(FakeBackend(stream=stream), stream=True)
    assert (records[0].prompt_tokens, records[0].completion_tokens) == (None, 2)
    assert records[0].source == "estimated"
