"""Circuit breaker state machine and routing with fallback."""

import logging

import pytest

from relay.backends.base import (
    BackendHTTPError,
    BackendResponse,
    BackendStreamError,
    BackendTimeout,
    BackendUnavailable,
)
from relay.config import RelayConfig
from relay.routing.circuit_breaker import BreakerPolicy, CircuitBreaker, State
from relay.routing.retry import RetryPolicy
from relay.routing.router import NoBackendAvailable, Router
from tests.unit.fakes import FakeBackend, FakeStream, make_app, make_client


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def breaker(threshold: int = 3, open_s: float = 10.0) -> tuple[CircuitBreaker, FakeClock]:
    clock = FakeClock()
    return CircuitBreaker("b", BreakerPolicy(threshold, open_s), clock), clock


# --- the state machine ----------------------------------------------------------------


def test_opens_after_consecutive_failures_and_success_resets_the_count() -> None:
    b, _ = breaker(threshold=3)
    b.record_failure()
    b.record_failure()
    b.record_success()  # not consecutive any more
    b.record_failure()
    b.record_failure()
    assert b.state is State.CLOSED
    b.record_failure()
    assert b.state is State.OPEN and not b.allow_request()


def test_cool_down_then_exactly_one_probe() -> None:
    b, clock = breaker(threshold=1, open_s=10)
    b.record_failure()
    clock.now += 9.9
    assert not b.allow_request()  # still cooling down
    clock.now += 0.2
    assert b.allow_request()  # the probe
    assert b.state is State.HALF_OPEN
    assert not b.allow_request()  # only one probe at a time


def test_probe_success_closes_and_probe_failure_reopens() -> None:
    b, clock = breaker(threshold=1, open_s=10)
    b.record_failure()
    clock.now += 10
    assert b.allow_request()
    b.record_failure()  # probe failed
    assert b.state is State.OPEN and not b.allow_request()  # a fresh cool-down
    clock.now += 10
    assert b.allow_request()
    b.record_success()  # probe succeeded
    assert b.state is State.CLOSED and b.allow_request() and b.allow_request()


def test_a_probe_that_never_reports_is_replaced() -> None:
    b, clock = breaker(threshold=1, open_s=10)
    b.record_failure()
    clock.now += 10
    assert b.allow_request()  # probe sent, its result never recorded
    clock.now += 10
    assert b.allow_request()  # so another probe is allowed eventually


def test_transitions_are_logged(caplog) -> None:
    b, clock = breaker(threshold=1, open_s=1)
    with caplog.at_level(logging.INFO, logger="relay.routing"):
        b.record_failure()
        clock.now += 1
        b.allow_request()
        b.record_success()
    assert "closed -> open" in caplog.text
    assert "open -> half_open" in caplog.text
    assert "half_open -> closed" in caplog.text


# --- the router -----------------------------------------------------------------------

CONFIG = RelayConfig.model_validate(
    {
        "backends": {
            "primary": {"engine": "llamacpp", "url": "http://primary"},
            "secondary": {"engine": "llamacpp", "url": "http://secondary"},
        },
        "models": {"m": {"backends": ["primary", "secondary"]}},
    }
)
OK = BackendResponse(200, b'{"ok": true}', "application/json")
DOWN = BackendUnavailable("primary", "connection refused")


def router(primary: FakeBackend, secondary: FakeBackend, threshold: int = 3):
    clock = FakeClock()
    breakers = {
        name: CircuitBreaker(name, BreakerPolicy(threshold, 10), clock)
        for name in ("primary", "secondary")
    }

    async def no_sleep(_: float) -> None:
        pass

    r = Router(
        CONFIG,
        {"primary": primary, "secondary": secondary},
        breakers,
        RetryPolicy(max_attempts=3, base_delay_s=0),
        sleep=no_sleep,
    )
    return r, clock


async def test_unavailable_primary_falls_back_after_its_retries() -> None:
    primary, secondary = FakeBackend("primary", error=DOWN), FakeBackend("secondary")
    r, _ = router(primary, secondary)
    backend, result = await r.chat("m", b"{}")
    assert backend is secondary and result.status_code == 200
    assert primary.calls == 3  # retried (ADR-003), then gave up on it


async def test_open_breaker_skips_the_backend_without_calling_it() -> None:
    primary, secondary = FakeBackend("primary", error=DOWN), FakeBackend("secondary")
    r, clock = router(primary, secondary, threshold=3)
    await r.chat("m", b"{}")  # 3 failures: primary's breaker opens
    assert r.breakers["primary"].state is State.OPEN
    for _ in range(5):
        backend, _ = await r.chat("m", b"{}")
        assert backend is secondary
    assert primary.calls == 3  # never tried while open: no waiting on a dead backend
    primary._error = None  # it recovers...
    clock.now += 10  # ...and after the cool-down one probe finds out
    backend, _ = await r.chat("m", b"{}")
    assert backend is primary and r.breakers["primary"].state is State.CLOSED


async def test_timeout_is_not_retried_on_the_same_backend_but_falls_back() -> None:
    primary = FakeBackend("primary", error=BackendTimeout("primary", "slow"))
    secondary = FakeBackend("secondary")
    r, _ = router(primary, secondary)
    backend, _ = await r.chat("m", b"{}")
    assert backend is secondary and primary.calls == 1


async def test_client_errors_are_not_failures_and_do_not_fall_back() -> None:
    bad_request = BackendHTTPError("primary", 400, b"{}", "application/json")
    primary, secondary = FakeBackend("primary", error=bad_request), FakeBackend("secondary")
    r, _ = router(primary, secondary, threshold=1)
    with pytest.raises(BackendHTTPError):
        await r.chat("m", b"{}")
    assert secondary.calls == 0
    assert r.breakers["primary"].state is State.CLOSED  # a bad request is not a sick backend


async def test_server_errors_fall_back() -> None:
    oops = BackendHTTPError("primary", 500, b"{}", "application/json")
    primary, secondary = FakeBackend("primary", error=oops), FakeBackend("secondary")
    r, _ = router(primary, secondary)
    backend, _ = await r.chat("m", b"{}")
    assert backend is secondary and primary.calls == 1  # 500 is not retried, but falls back


async def test_every_backend_failing_raises_the_last_error() -> None:
    primary = FakeBackend("primary", error=DOWN)
    secondary = FakeBackend("secondary", error=BackendUnavailable("secondary", "down"))
    r, _ = router(primary, secondary)
    with pytest.raises(BackendUnavailable) as excinfo:
        await r.chat("m", b"{}")
    assert excinfo.value.backend == "secondary"


async def test_every_breaker_open_raises_no_backend_available() -> None:
    primary = FakeBackend("primary", error=DOWN)
    secondary = FakeBackend("secondary", error=BackendUnavailable("secondary", "down"))
    r, _ = router(primary, secondary, threshold=1)
    with pytest.raises(BackendUnavailable):
        await r.chat("m", b"{}")  # opens both breakers
    with pytest.raises(NoBackendAvailable):
        await r.chat("m", b"{}")
    assert primary.calls == 1 and secondary.calls == 1  # the second call tried nobody


# --- through the gateway --------------------------------------------------------------


def gateway(primary: FakeBackend, secondary: FakeBackend):
    config = RelayConfig.model_validate(
        {**CONFIG.model_dump(mode="json"), "retry": {"base_delay_s": 0}}
    )
    events = []
    app = make_app(
        config=config,
        backends={"primary": primary, "secondary": secondary},
        usage_sink=events.append,
    )
    return make_client(app), events


def test_fallback_is_visible_in_header_and_ledger() -> None:
    client, events = gateway(FakeBackend("primary", error=DOWN), FakeBackend("secondary"))
    with client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
        )
    assert resp.status_code == 200
    assert resp.headers["x-relay-backend"] == "secondary"
    assert events[0].usage.backend == "secondary"


def test_all_breakers_open_is_503() -> None:
    primary = FakeBackend("primary", error=DOWN)
    secondary = FakeBackend("secondary", error=BackendUnavailable("secondary", "down"))
    client, _ = gateway(primary, secondary)
    body = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    with client:
        assert client.post("/v1/chat/completions", json=body).status_code == 502
        resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "service_unavailable"


def test_mid_stream_failure_counts_against_the_breaker_without_fallback() -> None:
    failure = BackendStreamError("primary", "connection lost", timeout=False)
    primary = FakeBackend("primary", stream=FakeStream([b"data: {}\n\n"], fail_with=failure))
    secondary = FakeBackend("secondary")
    client, _ = gateway(primary, secondary)
    body = {"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": True}
    with client:
        resp = client.post("/v1/chat/completions", json=body)
        breaker = client.app.state.router.breakers["primary"]
        assert breaker.consecutive_failures == 1
    assert resp.status_code == 200 and b"backend_unavailable" in resp.content
    assert secondary.calls == 0  # tokens were already sent: no fallback mid-answer
