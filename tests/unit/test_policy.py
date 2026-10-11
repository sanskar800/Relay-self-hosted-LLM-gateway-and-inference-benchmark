import json
import logging

import pytest

from relay.api.upstream import upstream_body
from relay.config import RelayConfig
from tests.unit.fakes import (
    TEST_KEY,
    FakeBackend,
    StaticKeyStore,
    make_app,
    make_client,
    make_principal,
)

CONFIG = RelayConfig.model_validate(
    {
        "backends": {"b": {"engine": "llamacpp", "url": "http://b"}},
        "models": {"public": {"backends": ["b"]}, "secret": {"backends": ["b"]}},
    }
)
MESSAGES = [{"role": "user", "content": "hi"}]


def client_for(*, allowed=("public",), cap: int | None = None):
    backend = FakeBackend("b")
    principal = make_principal(allowed_models=frozenset(allowed), max_tokens_cap=cap)
    app = make_app(
        config=CONFIG, backends={"b": backend}, key_store=StaticKeyStore({TEST_KEY: principal})
    )
    return make_client(app), backend


def chat(client, **body):
    return client.post("/v1/chat/completions", json={"messages": MESSAGES, **body})


# --- model allow-list -----------------------------------------------------------------


def test_allowed_model_is_served() -> None:
    client, backend = client_for()
    with client:
        assert chat(client, model="public").status_code == 200
    assert len(backend.requests) == 1


def test_forbidden_model_looks_exactly_like_an_unknown_one(caplog) -> None:
    client, backend = client_for()
    with client, caplog.at_level(logging.INFO, logger="relay.policy"):
        forbidden = chat(client, model="secret")
        unknown = chat(client, model="nope")
    assert forbidden.status_code == unknown.status_code == 404
    f_err, u_err = forbidden.json()["error"], unknown.json()["error"]
    # Identical apart from the model name the client itself sent.
    assert f_err["message"].replace("secret", "X") == u_err["message"].replace("nope", "X")
    assert {k: v for k, v in f_err.items() if k != "message"} == {
        k: v for k, v in u_err.items() if k != "message"
    }
    assert backend.requests == []
    # ...but the operator can see why.
    assert "not in its allow-list" in caplog.text and "secret" in caplog.text


def test_models_list_and_retrieve_show_only_allowed_models() -> None:
    client, _ = client_for()
    with client:
        assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["public"]
        assert client.get("/v1/models/public").status_code == 200
        assert client.get("/v1/models/secret").status_code == 404


def test_allow_list_entries_not_in_config_are_ignored() -> None:
    client, _ = client_for(allowed=("public", "retired-model"))
    with client:
        assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["public"]


# --- max_tokens cap -------------------------------------------------------------------


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_over_cap_is_400_naming_the_field(field: str) -> None:
    client, backend = client_for(cap=50)
    with client:
        resp = chat(client, model="public", **{field: 51})
    assert resp.status_code == 400
    assert resp.json()["error"]["param"] == field
    assert "limit is 50" in resp.json()["error"]["message"]
    assert backend.requests == []


def test_at_cap_is_forwarded_unchanged() -> None:
    client, backend = client_for(cap=50)
    with client:
        assert chat(client, model="public", max_tokens=50).status_code == 200
    assert json.loads(backend.requests[0])["max_tokens"] == 50


def test_omitted_max_tokens_gets_the_cap_so_it_cannot_be_bypassed() -> None:
    client, backend = client_for(cap=50)
    with client:
        chat(client, model="public")
        chat(client, model="public", stream=True)
    plain, streamed = (json.loads(b) for b in backend.requests)
    assert plain["max_tokens"] == 50
    assert streamed["max_tokens"] == 50
    assert streamed["stream_options"] == {"include_usage": True}  # both additions at once


def test_without_a_cap_the_body_is_forwarded_byte_for_byte() -> None:
    client, backend = client_for(cap=None)
    raw = b'{"model": "public",  "messages": [{"role": "user", "content": "hi"}]}'
    with client:
        client.post(
            "/v1/chat/completions", content=raw, headers={"Content-Type": "application/json"}
        )
    assert backend.requests == [raw]


def test_upstream_body_returns_the_same_bytes_when_nothing_to_add() -> None:
    raw = b'{"a":  1}'
    assert upstream_body(raw, include_usage=False, default_max_tokens=None) is raw
