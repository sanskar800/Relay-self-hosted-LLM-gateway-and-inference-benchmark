"""In-memory test doubles that satisfy the Backend protocol without any HTTP."""

from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from relay.auth.keys import Principal, hash_key
from relay.backends.base import BackendError, BackendResponse
from relay.config import RelayConfig
from relay.limits.token_bucket import InMemoryRateLimiter
from relay.main import create_app


class FakeStream:
    def __init__(self, events: list[bytes], fail_with: BackendError | None = None) -> None:
        self._events = events
        self._fail_with = fail_with
        self.closed = False

    async def events(self) -> AsyncIterator[bytes]:
        for event in self._events:
            yield event
        if self._fail_with:
            raise self._fail_with

    async def aclose(self) -> None:
        self.closed = True


class FakeBackend:
    """Records requests; returns a scripted response, stream or error."""

    engine = "fake"

    def __init__(
        self,
        name: str = "fake",
        *,
        response: BackendResponse | None = None,
        stream: FakeStream | None = None,
        error: BackendError | None = None,
        healthy: bool = True,
    ) -> None:
        self.name = name
        self._response = response or BackendResponse(200, b'{"ok": true}', "application/json")
        self._stream = stream or FakeStream([b"data: [DONE]\n\n"])
        self._error = error
        self._healthy = healthy
        self.requests: list[bytes] = []
        self.closed = False

    async def chat(self, body: bytes) -> BackendResponse:
        self.requests.append(body)
        if self._error:
            raise self._error
        return self._response

    async def stream(self, body: bytes) -> FakeStream:
        self.requests.append(body)
        if self._error:
            raise self._error
        return self._stream

    async def health(self) -> bool:
        return self._healthy

    async def aclose(self) -> None:
        self.closed = True


TEST_KEY = "rk_unit-test-key"
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_KEY}"}
TEST_MODELS = frozenset({"m", "qwen-1.5b", "qwen-0.5b", "Qwen/Qwen2.5-1.5B-Instruct"})


def make_principal(**overrides) -> Principal:
    fields = {
        "tenant_id": "team-test",
        "key_id": "00000000-0000-0000-0000-000000000001",
        "key_prefix": TEST_KEY[:8],
        "allowed_models": TEST_MODELS,
        "requests_per_minute": 1000,
        "tokens_per_minute": 1_000_000,
        "max_tokens_cap": None,
    }
    return Principal(**{**fields, **overrides})


class StaticKeyStore:
    """Key store with fixed keys, in memory. Counts lookups to observe caching."""

    def __init__(self, keys: dict[str, Principal] | None = None) -> None:
        keys = keys if keys is not None else {TEST_KEY: make_principal()}
        self._by_hash = {hash_key(raw): p for raw, p in keys.items()}
        self.lookups = 0

    async def lookup(self, key_hash: bytes) -> Principal | None:
        self.lookups += 1
        return self._by_hash.get(key_hash)


def make_app(**kwargs) -> FastAPI:
    """create_app with in-memory auth (accepts TEST_KEY) and rate limiting: no Postgres/Redis."""
    kwargs.setdefault("key_store", StaticKeyStore())
    kwargs.setdefault("rate_limiter", InMemoryRateLimiter())
    return create_app(**kwargs)


def make_client(app: FastAPI, **kwargs) -> TestClient:
    """TestClient that sends TEST_KEY on every request."""
    return TestClient(app, headers=AUTH_HEADERS, **kwargs)


def make_config(model: str = "m", backend: str = "b1", url: str = "http://engine") -> RelayConfig:
    """Smallest valid routing config: one model served by one llama.cpp backend."""
    return RelayConfig.model_validate(
        {
            "backends": {backend: {"engine": "llamacpp", "url": url}},
            "models": {model: {"backends": [backend]}},
        }
    )


def app_with_fake(backend: "FakeBackend", model: str = "m") -> FastAPI:
    """Gateway with an in-memory backend: no HTTP below Relay."""
    return make_app(config=make_config(model, backend.name), backends={backend.name: backend})


def app_with_transport(transport: httpx.AsyncBaseTransport, model: str = "m") -> FastAPI:
    """Gateway with the real HTTP backend, whose network is replaced by `transport`."""
    return make_app(config=make_config(model), transport=transport)
