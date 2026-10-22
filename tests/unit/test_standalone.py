"""Standalone mode: Relay without Postgres or Redis (e.g. the Colab benchmark VM)."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from relay.config import RelayConfig, Settings
from relay.main import create_app
from tests.fake_engine import FakeEngine

KEY = "rk_standalone_test_key_0123456789"
CONFIG = RelayConfig.model_validate(
    {
        "backends": {"b": {"engine": "vllm", "url": "http://engine"}},
        "models": {"m": {"backends": ["b"]}},
    }
)


def standalone_client(engine: FakeEngine) -> TestClient:
    settings = Settings(
        _env_file=None,  # independent of the developer's .env (e.g. its tracing endpoint)
        standalone=True,
        standalone_api_key=KEY,
        # Unreachable on purpose: standalone mode must never try them.
        database_url="postgresql://nobody@127.0.0.1:1/none",
        redis_url="redis://127.0.0.1:1/0",
    )
    app = create_app(settings, config=CONFIG, transport=engine.transport())
    return TestClient(app)


def test_serves_with_its_one_key_and_no_database() -> None:
    engine = FakeEngine()
    with standalone_client(engine) as client:
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        ok = client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {KEY}"}
        )
        wrong = client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer rk_wrong"}
        )
        models = client.get("/v1/models", headers={"Authorization": f"Bearer {KEY}"})
    assert ok.status_code == 200 and ok.json()["choices"][0]["message"]["content"] == "Hello!"
    assert wrong.status_code == 401  # still authenticated, not open
    assert [m["id"] for m in models.json()["data"]] == ["m"]  # every configured model


def test_standalone_without_a_key_refuses_to_start() -> None:
    with pytest.raises(ValidationError, match="RELAY_STANDALONE_API_KEY"):
        Settings(_env_file=None, standalone=True, standalone_api_key=None)
