import openai
import pytest

from relay.api.schemas import ChatCompletionRequest
from relay.config import RelayConfig
from relay.limits.estimate import estimate_prompt_tokens, reservation
from relay.limits.token_bucket import InMemoryRateLimiter
from tests.unit.fakes import (
    TEST_KEY,
    FakeBackend,
    StaticKeyStore,
    make_app,
    make_client,
    make_principal,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# --- bucket maths ---------------------------------------------------------------------


async def test_request_bucket_allows_burst_then_refills_continuously() -> None:
    clock = FakeClock()
    limiter = InMemoryRateLimiter(clock)
    p = make_principal(requests_per_minute=3, tokens_per_minute=10_000)
    assert [(await limiter.acquire(p, 1)).allowed for _ in range(3)] == [True] * 3
    denied = await limiter.acquire(p, 1)
    assert not denied.allowed
    assert denied.retry_after_s == pytest.approx(20)  # 1 request / (3 per 60 s)
    clock.now += 19.9
    assert not (await limiter.acquire(p, 1)).allowed
    clock.now += 0.2
    assert (await limiter.acquire(p, 1)).allowed  # refilled, not reset at a minute boundary


async def test_token_bucket_and_retry_after() -> None:
    clock = FakeClock()
    limiter = InMemoryRateLimiter(clock)
    p = make_principal(requests_per_minute=100, tokens_per_minute=120)  # 2 tokens/s
    assert (await limiter.acquire(p, 100)).allowed
    denied = await limiter.acquire(p, 50)  # 20 left, 30 short
    assert not denied.allowed and denied.retry_after_s == pytest.approx(15)
    clock.now += 15
    assert (await limiter.acquire(p, 50)).allowed


async def test_denied_request_takes_nothing_from_either_bucket() -> None:
    limiter = InMemoryRateLimiter(FakeClock())
    p = make_principal(requests_per_minute=10, tokens_per_minute=100)
    assert not (await limiter.acquire(p, 101)).allowed  # too large: no state change
    d = await limiter.acquire(p, 100)
    assert d.allowed and d.remaining_requests == pytest.approx(9)


async def test_request_larger_than_bucket_is_flagged_too_large() -> None:
    limiter = InMemoryRateLimiter(FakeClock())
    d = await limiter.acquire(make_principal(tokens_per_minute=100), 101)
    assert not d.allowed and d.too_large


async def test_settle_refunds_unused_reservation_and_can_charge_more() -> None:
    limiter = InMemoryRateLimiter(FakeClock())
    p = make_principal(requests_per_minute=100, tokens_per_minute=100)
    await limiter.acquire(p, 90)
    await limiter.settle(p, 80)  # reserved 90, used 10
    assert (await limiter.acquire(p, 85)).allowed
    await limiter.settle(p, -40)  # used 40 more than reserved: the bucket goes into debt
    assert not (await limiter.acquire(p, 1)).allowed


async def test_tenants_have_independent_buckets() -> None:
    limiter = InMemoryRateLimiter(FakeClock())
    a = make_principal(tenant_id="team-a", requests_per_minute=1)
    b = make_principal(tenant_id="team-b", requests_per_minute=1)
    assert (await limiter.acquire(a, 1)).allowed
    assert not (await limiter.acquire(a, 1)).allowed
    assert (await limiter.acquire(b, 1)).allowed


def test_reservation_is_estimated_prompt_plus_max_completion() -> None:
    payload = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "x" * 400}], "max_tokens": 50}
    )
    assert estimate_prompt_tokens(payload) == 100 + 4
    assert reservation(payload, max_tokens_default=999) == 104 + 50
    payload.max_tokens = None
    assert reservation(payload, max_tokens_default=999) == 104 + 999


# --- through the gateway --------------------------------------------------------------

CONFIG = RelayConfig.model_validate(
    {
        "backends": {"b": {"engine": "llamacpp", "url": "http://b"}},
        "models": {"m": {"backends": ["b"]}},
    }
)
BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
USAGE = b'{"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}'


def gateway(keys: dict, limiter: InMemoryRateLimiter | None = None):
    from relay.backends.base import BackendResponse

    backend = FakeBackend("b", response=BackendResponse(200, USAGE, "application/json"))
    app = make_app(
        config=CONFIG,
        backends={"b": backend},
        key_store=StaticKeyStore(keys),
        rate_limiter=limiter or InMemoryRateLimiter(FakeClock()),
    )
    return make_client(app), backend


def test_429_with_retry_after_and_openai_headers() -> None:
    client, backend = gateway({TEST_KEY: make_principal(requests_per_minute=2)})
    with client:
        ok = [client.post("/v1/chat/completions", json=BODY) for _ in range(2)]
        limited = client.post("/v1/chat/completions", json=BODY)
    assert [r.status_code for r in ok] == [200, 200]
    assert ok[0].headers["x-ratelimit-limit-requests"] == "2"
    assert ok[1].headers["x-ratelimit-remaining-requests"] == "0"
    assert limited.status_code == 429
    error = limited.json()["error"]
    assert (error["type"], error["code"]) == ("requests", "rate_limit_exceeded")
    assert limited.headers["retry-after"] == "30"  # 1 request / (2 per 60 s)
    assert limited.headers["retry-after-ms"] == "30000"
    assert len(backend.requests) == 2  # the limited request never reached the backend


def test_one_tenant_limited_the_other_unaffected() -> None:
    other_key = "rk_other-tenant-key"
    keys = {
        TEST_KEY: make_principal(tenant_id="noisy", requests_per_minute=1),
        other_key: make_principal(tenant_id="quiet", requests_per_minute=1),
    }
    client, _ = gateway(keys)
    with client:
        assert client.post("/v1/chat/completions", json=BODY).status_code == 200
        assert client.post("/v1/chat/completions", json=BODY).status_code == 429
        quiet = client.post(
            "/v1/chat/completions", json=BODY, headers={"Authorization": f"Bearer {other_key}"}
        )
    assert quiet.status_code == 200


def test_reservation_is_settled_against_real_usage() -> None:
    # 100 tokens/min; each request reserves ~5 (prompt) + 10 (max_tokens) but uses 8.
    client, _ = gateway({TEST_KEY: make_principal(tokens_per_minute=100)})
    with client:
        first = client.post("/v1/chat/completions", json=BODY)
        second = client.post("/v1/chat/completions", json=BODY)
    # After settling, only the real 8 tokens of the first request are gone.
    reserved = 0 + 4 + 10  # "hi": 2 chars // 4 = 0, + 4 per message, + max_tokens 10
    assert int(first.headers["x-ratelimit-remaining-tokens"]) == 100 - reserved
    assert int(second.headers["x-ratelimit-remaining-tokens"]) == 100 - 8 - reserved


def test_failed_backend_call_refunds_its_tokens() -> None:
    from relay.backends.base import BackendUnavailable

    backend = FakeBackend("b", error=BackendUnavailable("b", "down"))
    limiter = InMemoryRateLimiter(FakeClock())
    app = make_app(
        config=CONFIG,
        backends={"b": backend},
        key_store=StaticKeyStore({TEST_KEY: make_principal(tokens_per_minute=100)}),
        rate_limiter=limiter,
    )
    with make_client(app) as client:
        assert client.post("/v1/chat/completions", json=BODY).status_code == 502
        backend._error = None
        ok = client.post("/v1/chat/completions", json=BODY)
    assert int(ok.headers["x-ratelimit-remaining-tokens"]) == 100 - 14  # nothing lost to the 502


def test_request_too_large_for_the_bucket() -> None:
    client, backend = gateway({TEST_KEY: make_principal(tokens_per_minute=50)})
    with client:
        resp = client.post("/v1/chat/completions", json={**BODY, "max_tokens": 100})
    assert resp.status_code == 429
    assert "Request too large" in resp.json()["error"]["message"]
    assert "retry-after" not in resp.headers  # retrying cannot help
    assert backend.requests == []


def test_sdk_raises_rate_limit_error() -> None:
    client, _ = gateway({TEST_KEY: make_principal(requests_per_minute=1)})
    with client as http:
        sdk = openai.OpenAI(
            base_url="http://testserver/v1", api_key=TEST_KEY, http_client=http, max_retries=0
        )
        sdk.chat.completions.create(**BODY)
        with pytest.raises(openai.RateLimitError) as excinfo:
            sdk.chat.completions.create(**BODY)
    assert excinfo.value.code == "rate_limit_exceeded"
