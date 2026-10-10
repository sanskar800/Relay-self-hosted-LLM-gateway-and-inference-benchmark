"""Model routing and GET /v1/models, against in-memory fake backends."""

from fastapi.testclient import TestClient

from relay.config import RelayConfig
from relay.main import create_app
from tests.unit.fakes import FakeBackend

CONFIG = RelayConfig.model_validate(
    {
        "backends": {
            "gpu": {"engine": "llamacpp", "url": "http://gpu"},
            "cpu": {"engine": "llamacpp", "url": "http://cpu"},
        },
        "models": {
            "qwen-1.5b": {"backends": ["gpu", "cpu"], "owned_by": "qwen"},
            "qwen-0.5b": {"backends": ["cpu"]},
        },
    }
)


def make_client() -> tuple[TestClient, dict[str, FakeBackend]]:
    fakes = {"gpu": FakeBackend("gpu"), "cpu": FakeBackend("cpu")}
    return TestClient(create_app(config=CONFIG, backends=fakes)), fakes


def chat(client: TestClient, model: str):
    body = {"model": model, "messages": [{"role": "user", "content": "x"}]}
    return client.post("/v1/chat/completions", json=body)


def test_request_goes_to_first_backend_of_its_model() -> None:
    client, fakes = make_client()
    with client:
        assert chat(client, "qwen-1.5b").status_code == 200
        assert chat(client, "qwen-0.5b").status_code == 200
    assert len(fakes["gpu"].requests) == 1  # qwen-1.5b -> gpu (first in its list)
    assert len(fakes["cpu"].requests) == 1  # qwen-0.5b -> cpu


def test_unknown_model_is_404_and_reaches_no_backend() -> None:
    client, fakes = make_client()
    with client:
        resp = chat(client, "gpt-4o")
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "model_not_found"
    assert error["param"] == "model"
    assert "gpt-4o" in error["message"]
    assert not fakes["gpu"].requests and not fakes["cpu"].requests


def test_list_models_in_openai_format() -> None:
    client, _ = make_client()
    with client:
        body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["qwen-1.5b", "qwen-0.5b"]
    first = body["data"][0]
    assert first["object"] == "model" and first["owned_by"] == "qwen"
    assert isinstance(first["created"], int)


def test_retrieve_model_and_404() -> None:
    client, _ = make_client()
    with client:
        assert client.get("/v1/models/qwen-0.5b").json()["owned_by"] == "relay"
        missing = client.get("/v1/models/nope")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"


def test_model_ids_containing_slashes_can_be_retrieved() -> None:
    config = RelayConfig.model_validate(
        {
            "backends": {"b": {"engine": "vllm", "url": "http://b"}},
            "models": {"Qwen/Qwen2.5-1.5B-Instruct": {"backends": ["b"]}},
        }
    )
    with TestClient(create_app(config=config, backends={"b": FakeBackend("b")})) as client:
        assert client.get("/v1/models/Qwen/Qwen2.5-1.5B-Instruct").status_code == 200
