import re

import openai
import pytest
from fastapi.testclient import TestClient

from relay.auth.keys import (
    CachedKeyStore,
    KeyStoreUnavailable,
    Principal,
    display_prefix,
    generate_key,
    hash_key,
)
from tests.unit.fakes import (
    AUTH_HEADERS,
    TEST_KEY,
    FakeBackend,
    StaticKeyStore,
    make_app,
    make_config,
    make_principal,
)

REQUEST = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def client(key_store=None, backend: FakeBackend | None = None, **kwargs) -> TestClient:
    backend = backend or FakeBackend()
    app = make_app(
        config=make_config("m", backend.name),
        backends={backend.name: backend},
        key_store=key_store or StaticKeyStore(),
    )
    return TestClient(app, **kwargs)


# --- keys -----------------------------------------------------------------------------


def test_generated_keys_have_format_entropy_and_hash() -> None:
    keys = [generate_key() for _ in range(50)]
    for key in keys:
        assert re.fullmatch(r"rk_[A-Za-z0-9_-]{43}", key.raw)
        assert key.key_hash == hash_key(key.raw) and len(key.key_hash) == 32
        assert key.display_prefix == key.raw[:8]
    assert len({k.raw for k in keys}) == 50


# --- the dependency -------------------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Basic dXNlcjpwYXNz"}, {"Authorization": "Bearer "}],
)
def test_missing_key_is_401_and_reaches_no_backend(headers: dict) -> None:
    backend = FakeBackend()
    with client(backend=backend) as c:
        resp = c.post("/v1/chat/completions", json=REQUEST, headers=headers)
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    assert resp.json()["error"]["code"] == "missing_api_key"
    assert backend.requests == []


def test_unknown_key_is_401_and_only_its_prefix_is_echoed() -> None:
    wrong = "rk_this-key-does-not-exist-anywhere-at-all-0123"
    with client() as c:
        resp = c.post(
            "/v1/chat/completions", json=REQUEST, headers={"Authorization": f"Bearer {wrong}"}
        )
    assert resp.status_code == 401
    error = resp.json()["error"]
    assert error["code"] == "invalid_api_key"
    assert display_prefix(wrong) in error["message"]
    assert wrong not in resp.text  # never echo the full key


def test_valid_key_is_served_and_principal_attached() -> None:
    backend = FakeBackend()
    with client(backend=backend, headers=AUTH_HEADERS) as c:
        assert c.post("/v1/chat/completions", json=REQUEST).status_code == 200
        assert c.get("/v1/models").status_code == 200
    assert len(backend.requests) == 1


def test_authentication_runs_before_body_validation() -> None:
    with client() as c:
        resp = c.post("/v1/chat/completions", json={"model": "", "messages": "nonsense"})
    assert resp.status_code == 401  # not a 400 describing the API to an anonymous caller


def test_healthz_needs_no_key_but_models_does() -> None:
    with client() as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/v1/models").status_code == 401


class DownKeyStore:
    async def lookup(self, key_hash: bytes) -> Principal | None:
        raise KeyStoreUnavailable("connection refused")


def test_key_store_down_fails_closed_with_503() -> None:
    backend = FakeBackend()
    with client(key_store=DownKeyStore(), backend=backend, headers=AUTH_HEADERS) as c:
        resp = c.post("/v1/chat/completions", json=REQUEST)
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "service_unavailable"
    assert backend.requests == []


def test_sdk_raises_authentication_error() -> None:
    with client() as http:
        sdk = openai.OpenAI(
            base_url="http://testserver/v1", api_key="rk_wrong", http_client=http, max_retries=0
        )
        with pytest.raises(openai.AuthenticationError) as excinfo:
            sdk.chat.completions.create(model="m", messages=REQUEST["messages"])
    assert excinfo.value.code == "invalid_api_key"


def test_openapi_declares_bearer_auth_for_swagger() -> None:
    schema = make_app(config=make_config()).openapi()
    assert schema["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"


# --- the cache ------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


async def test_cache_hit_avoids_store_until_ttl_expires() -> None:
    inner, clock = StaticKeyStore(), FakeClock()
    cache = CachedKeyStore(inner, ttl_s=30, clock=clock)
    h = hash_key(TEST_KEY)
    assert (await cache.lookup(h)).tenant_id == "team-test"
    clock.now += 29
    await cache.lookup(h)
    assert inner.lookups == 1  # served from cache
    clock.now += 2  # 31 s after the first lookup
    await cache.lookup(h)
    assert inner.lookups == 2  # expired: a revoked key stops working after <= ttl_s


async def test_unknown_keys_are_cached_briefly() -> None:
    inner, clock = StaticKeyStore(), FakeClock()
    cache = CachedKeyStore(inner, ttl_s=30, negative_ttl_s=5, clock=clock)
    h = hash_key("rk_unknown")
    assert await cache.lookup(h) is None
    assert await cache.lookup(h) is None
    assert inner.lookups == 1
    clock.now += 6
    await cache.lookup(h)
    assert inner.lookups == 2


async def test_cache_size_is_bounded() -> None:
    cache = CachedKeyStore(StaticKeyStore(), max_entries=3, clock=FakeClock())
    for i in range(10):
        await cache.lookup(hash_key(f"rk_random-{i}"))
    assert len(cache._entries) == 3


async def test_store_failure_is_not_cached() -> None:
    cache = CachedKeyStore(DownKeyStore())
    for _ in range(2):
        with pytest.raises(KeyStoreUnavailable):
            await cache.lookup(hash_key(TEST_KEY))
    assert len(cache._entries) == 0


def test_principal_helper_is_a_frozen_value() -> None:
    p = make_principal(tenant_id="other")
    assert p.tenant_id == "other"
    with pytest.raises(AttributeError):
        p.tenant_id = "x"  # type: ignore[misc]
