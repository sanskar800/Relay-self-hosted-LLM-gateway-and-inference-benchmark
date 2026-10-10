import json
from collections.abc import Iterator

import openai
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from relay.config import LimitsConfig, RelayConfig
from relay.main import create_app
from tests.unit.fakes import FakeBackend

LIMITS = {"max_body_bytes": 300, "max_messages": 3, "max_tokens": 100}
MESSAGES = [{"role": "user", "content": "hi"}]


def client_with(backend: FakeBackend) -> TestClient:
    config = RelayConfig.model_validate(
        {
            "backends": {backend.name: {"engine": "llamacpp", "url": "http://x"}},
            "models": {"m": {"backends": [backend.name]}},
            "limits": LIMITS,
        }
    )
    return TestClient(create_app(config=config, backends={backend.name: backend}))


def post(backend: FakeBackend, **kwargs):
    with client_with(backend) as client:
        return client.post("/v1/chat/completions", **kwargs)


def assert_openai_error(resp, status: int, param: str | None = None) -> None:
    assert resp.status_code == status
    error = resp.json()["error"]
    assert set(error) == {"message", "type", "param", "code"}
    assert error["param"] == param


def test_request_within_limits_is_served() -> None:
    backend = FakeBackend()
    resp = post(backend, json={"model": "m", "messages": MESSAGES, "max_tokens": 100})
    assert resp.status_code == 200 and len(backend.requests) == 1


def test_declared_oversize_body_is_rejected_before_reading() -> None:
    backend = FakeBackend()
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "x" * 400}]})
    resp = post(backend, content=body, headers={"Content-Type": "application/json"})
    assert_openai_error(resp, 413)
    assert "300 bytes" in resp.json()["error"]["message"]
    assert backend.requests == []


def test_chunked_oversize_body_without_content_length_is_rejected() -> None:
    backend = FakeBackend()

    def chunks() -> Iterator[bytes]:  # a generator body is sent chunked, no Content-Length
        yield b'{"model": "m", "messages": [{"role": "user", "content": "'
        for _ in range(10):
            yield b"x" * 50
        yield b'"}]}'

    resp = post(backend, content=chunks(), headers={"Content-Type": "application/json"})
    assert_openai_error(resp, 413)
    assert backend.requests == []


def test_too_many_messages() -> None:
    backend = FakeBackend()
    resp = post(backend, json={"model": "m", "messages": MESSAGES * 4})
    assert_openai_error(resp, 400, "messages")
    assert backend.requests == []


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_token_ceiling(field: str) -> None:
    backend = FakeBackend()
    resp = post(backend, json={"model": "m", "messages": MESSAGES, field: 101})
    assert_openai_error(resp, 400, field)
    assert backend.requests == []


def test_limits_have_safe_defaults_and_reject_nonsense() -> None:
    assert LimitsConfig().max_body_bytes == 1_048_576
    with pytest.raises(ValidationError):
        LimitsConfig(max_tokens=0)
    with pytest.raises(ValidationError):
        LimitsConfig.model_validate({"max_body_size": 10})  # typo'd key


def test_sdk_sees_413_as_a_status_error() -> None:
    with client_with(FakeBackend()) as http:
        sdk = openai.OpenAI(
            base_url="http://testserver/v1", api_key="x", http_client=http, max_retries=0
        )
        with pytest.raises(openai.APIStatusError) as excinfo:
            sdk.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "x" * 400}]
            )
    assert excinfo.value.status_code == 413
