import json

import httpx
import pytest
from fastapi.testclient import TestClient

from relay.config import Settings
from relay.main import create_app

COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    "timings": {"engine_specific": True},
}


def make_client(handler) -> TestClient:
    app = create_app(Settings(), transport=httpx.MockTransport(handler))
    return TestClient(app)


def test_healthz() -> None:
    with make_client(lambda r: httpx.Response(500)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}


def test_forwards_body_unchanged_and_returns_backend_response() -> None:
    seen: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=COMPLETION)

    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "seed": 7}
    with make_client(backend) as client:
        resp = client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 200
    assert resp.json() == COMPLETION  # engine-specific fields pass through
    assert seen[0].url.path == "/v1/chat/completions"
    assert json.loads(seen[0].content) == body  # unknown fields (seed) reach the backend


def test_backend_error_status_is_passed_through() -> None:
    with make_client(lambda r: httpx.Response(400, json={"error": {"message": "bad"}})) as client:
        resp = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert resp.status_code == 400
    assert resp.json()["error"]["message"] == "bad"


@pytest.mark.parametrize(
    ("exc", "status"),
    [(httpx.ConnectError("refused"), 502), (httpx.ReadTimeout("slow"), 504)],
)
def test_backend_failure_becomes_openai_error(exc: Exception, status: int) -> None:
    def backend(request: httpx.Request) -> httpx.Response:
        raise exc

    with make_client(backend) as client:
        resp = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert resp.status_code == status
    assert set(resp.json()["error"]) == {"message", "type", "param", "code"}


@pytest.mark.parametrize("raw", [b"not json", b"[1, 2]"])
def test_invalid_body_is_rejected_before_backend(raw: bytes) -> None:
    def backend(request: httpx.Request) -> httpx.Response:
        raise AssertionError("backend must not be called")

    with make_client(backend) as client:
        resp = client.post(
            "/v1/chat/completions", content=raw, headers={"Content-Type": "application/json"}
        )
    assert resp.status_code == 400


def test_streaming_is_rejected_until_implemented() -> None:
    def backend(request: httpx.Request) -> httpx.Response:
        raise AssertionError("backend must not be called")

    with make_client(backend) as client:
        resp = client.post(
            "/v1/chat/completions", json={"model": "m", "messages": [], "stream": True}
        )
    assert resp.status_code == 501
