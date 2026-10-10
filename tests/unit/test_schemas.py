import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from relay.api.schemas import ChatCompletionRequest
from relay.config import Settings
from relay.main import create_app

VALID = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def test_minimal_request_is_valid() -> None:
    req = ChatCompletionRequest.model_validate(VALID)
    assert req.stream is False
    assert req.messages[0].role == "user"


def test_unknown_fields_are_kept_at_every_level() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            **VALID,
            "seed": 7,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "hi", "name": "alice"}],
        }
    )
    assert req.model_extra == {"seed": 7, "response_format": {"type": "json_object"}}
    assert req.messages[0].model_extra == {"name": "alice"}


def test_content_parts_and_null_content_are_accepted() -> None:
    ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": None, "tool_calls": []},
            ],
        }
    )


@pytest.mark.parametrize(
    ("override", "bad_field"),
    [
        ({"model": ""}, "model"),
        ({"messages": []}, "messages"),
        ({"messages": "hello"}, "messages"),
        ({"messages": [{"role": "robot", "content": "hi"}]}, "messages.0.role"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"temperature": 3}, "temperature"),
        ({"top_p": 1.5}, "top_p"),
        ({"n": 2}, "n"),
    ],
)
def test_invalid_requests_are_rejected(override: dict, bad_field: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        ChatCompletionRequest.model_validate({**VALID, **override})
    loc = ".".join(str(p) for p in excinfo.value.errors()[0]["loc"])
    assert loc == bad_field


def _client(backend_called: list[bool]) -> TestClient:
    def backend(request: httpx.Request) -> httpx.Response:
        backend_called.append(True)
        return httpx.Response(200, json={})

    return TestClient(create_app(Settings(), transport=httpx.MockTransport(backend)))


def test_endpoint_returns_openai_400_naming_the_bad_field() -> None:
    called: list[bool] = []
    with _client(called) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={**VALID, "messages": [{"role": "robot", "content": "hi"}]},
        )
    assert resp.status_code == 400  # not FastAPI's default 422
    error = resp.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "messages.0.role"
    assert set(error) == {"message", "type", "param", "code"}
    assert called == []  # rejected before reaching the backend


def test_original_bytes_are_forwarded_not_reserialised() -> None:
    seen: list[bytes] = []

    def backend(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        return httpx.Response(200, json={})

    raw = b'{"model":"m","messages":[{"role":"user","content":"hi"}],"seed":7,"top_k":40}'
    app = create_app(Settings(), transport=httpx.MockTransport(backend))
    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions", content=raw, headers={"Content-Type": "application/json"}
        )
    assert seen == [raw]


def test_openapi_documents_the_request_body() -> None:
    schema = create_app(Settings()).openapi()
    body = schema["paths"]["/v1/chat/completions"]["post"]["requestBody"]
    assert "ChatCompletionRequest" in str(body)
    responses = schema["paths"]["/v1/chat/completions"]["post"]["responses"]
    assert "422" not in responses  # Relay answers validation errors with 400
    assert {"400", "502", "504"} <= set(responses)
