"""Timeouts and retries before the first token (ADR-003)."""

import json

import pytest

from relay.backends.base import (
    BackendHTTPError,
    BackendStreamError,
    BackendTimeout,
    BackendUnavailable,
)
from relay.config import RelayConfig
from relay.routing.retry import RetryPolicy, call_with_retry, is_retryable
from tests.fake_engine import FakeEngine
from tests.unit.fakes import FakeBackend, FakeStream, app_with_fake, make_app, make_client

BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
DOWN = BackendUnavailable("b", "connection refused")
BUSY = BackendHTTPError("b", 503, b'{"error": {"message": "loading"}}', "application/json")


# --- the retry policy -----------------------------------------------------------------


def test_backoff_is_exponential_capped_and_jittered() -> None:
    policy = RetryPolicy(max_attempts=6, base_delay_s=0.1, max_delay_s=1.0)
    ceilings = [policy.delay(n, rand=lambda: 1.0) for n in range(6)]
    assert ceilings == pytest.approx([0.1, 0.2, 0.4, 0.8, 1.0, 1.0])
    assert policy.delay(3, rand=lambda: 0.25) == pytest.approx(0.2)  # anywhere in [0, ceiling]


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (DOWN, True),
        (BUSY, True),
        (BackendHTTPError("b", 400, b"{}", "application/json"), False),  # the request is wrong
        (BackendHTTPError("b", 500, b"{}", "application/json"), False),
        (BackendTimeout("b", "slow"), False),  # may still be working: don't double its load
    ],
)
def test_what_is_retried(error, retryable: bool) -> None:
    assert is_retryable(error) is retryable


async def test_retries_until_success_with_backoff_between_attempts() -> None:
    errors, sleeps = [DOWN, BUSY], []

    async def attempt() -> str:
        if errors:
            raise errors.pop(0)
        return "ok"

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = await call_with_retry(attempt, RetryPolicy(), sleep=sleep, rand=lambda: 1.0)
    assert result == "ok"
    assert sleeps == pytest.approx([0.1, 0.2])


async def test_gives_up_after_max_attempts_and_never_retries_non_retryable() -> None:
    calls = []

    async def always(error):
        calls.append(error)
        raise error

    async def no_sleep(_: float) -> None:
        pass

    with pytest.raises(BackendUnavailable):
        await call_with_retry(lambda: always(DOWN), RetryPolicy(max_attempts=3), sleep=no_sleep)
    assert len(calls) == 3
    calls.clear()
    with pytest.raises(BackendTimeout):
        await call_with_retry(
            lambda: always(BackendTimeout("b", "x")), RetryPolicy(), sleep=no_sleep
        )
    assert len(calls) == 1


# --- through the gateway: the first-token boundary -------------------------------------


def post(backend: FakeBackend, **extra):
    with make_client(app_with_fake(backend)) as client:
        return client.post("/v1/chat/completions", json={**BODY, **extra})


@pytest.mark.parametrize("stream", [False, True])
def test_failures_before_the_first_token_are_retried(stream: bool) -> None:
    stream_obj = FakeStream([b'data: {"choices": []}\n\n', b"data: [DONE]\n\n"])
    backend = FakeBackend(fail_first=[DOWN, BUSY], stream=stream_obj)
    resp = post(backend, stream=stream)
    assert resp.status_code == 200
    assert backend.calls == 3  # two failures, then success: invisible to the client


def test_failure_after_tokens_reached_the_client_is_never_retried() -> None:
    tokens = [
        f'data: {{"choices": [{{"delta": {{"content": "t{i}"}}}}]}}\n\n'.encode() for i in range(3)
    ]
    failure = BackendStreamError("b", "connection lost", timeout=False)
    backend = FakeBackend(stream=FakeStream(tokens, fail_with=failure))
    resp = post(backend, stream=True)
    assert resp.status_code == 200  # already committed when the first token went out
    events = resp.content.split(b"\n\n")[:-1]
    assert len(events) == 4  # the 3 tokens, then one error event, no restarted answer
    assert json.loads(events[-1].removeprefix(b"data: "))["error"]["type"] == "backend_unavailable"
    assert backend.calls == 1


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (BackendHTTPError("b", 400, b'{"error": {}}', "application/json"), 400),
        (BackendTimeout("b", "slow"), 504),
    ],
)
def test_non_retryable_failures_are_returned_after_one_attempt(error, status: int) -> None:
    backend = FakeBackend(fail_first=[error])
    assert post(backend).status_code == status
    assert backend.calls == 1


def test_exhausted_retries_return_502() -> None:
    backend = FakeBackend(fail_first=[DOWN, DOWN, DOWN])
    assert post(backend).status_code == 502
    assert backend.calls == 3


# --- timeouts with a slow engine (real HTTP backend code, scripted engine) -----------


def engine_app(engine: FakeEngine, **timeouts):
    config = RelayConfig.model_validate(
        {
            "backends": {"b1": {"engine": "llamacpp", "url": "http://engine", **timeouts}},
            "models": {"m": {"backends": ["b1"]}},
            "retry": {"base_delay_s": 0},
        }
    )
    return make_client(make_app(config=config, transport=engine.transport()))


def test_first_token_timeout_is_504_and_not_retried_on_the_same_backend() -> None:
    engine = FakeEngine(first_event_delay_s=1.0)
    with engine_app(engine, first_token_timeout_s=0.1) as client:
        resp = client.post("/v1/chat/completions", json={**BODY, "stream": True})
    assert resp.status_code == 504
    assert resp.json()["error"]["type"] == "backend_timeout"
    assert len(engine.requests) == 1


def test_slow_but_steady_stream_is_fine() -> None:
    # 5 tokens, 0.05 s apart: well past the first-token timeout in total, but never stuck.
    engine = FakeEngine(tokens=list("abcde"), token_delay_s=0.05)
    with engine_app(engine, first_token_timeout_s=0.1) as client:
        resp = client.post("/v1/chat/completions", json={**BODY, "stream": True})
    assert resp.status_code == 200 and b"[DONE]" in resp.content


def test_total_timeout_ends_a_running_stream_with_an_error_event() -> None:
    engine = FakeEngine(tokens=list("abcdefgh"), stall_after=2, stall_s=2.0)
    with engine_app(
        engine,
        connect_timeout_s=0.1,
        first_token_timeout_s=0.2,
        idle_timeout_s=0.3,
        total_timeout_s=0.3,
    ) as client:
        resp = client.post("/v1/chat/completions", json={**BODY, "stream": True})
    assert resp.status_code == 200
    last = resp.content.split(b"\n\n")[-2]
    error = json.loads(last.removeprefix(b"data: "))["error"]
    assert error["type"] == "backend_timeout"
    assert b'"content": "a"' in resp.content and b"[DONE]" not in resp.content


def test_timeouts_must_fit_inside_the_total() -> None:
    with pytest.raises(ValueError, match="cannot exceed total_timeout_s"):
        RelayConfig.model_validate(
            {
                "backends": {"b": {"engine": "llamacpp", "url": "http://b",
                                   "first_token_timeout_s": 600, "total_timeout_s": 300}},
                "models": {"m": {"backends": ["b"]}},
            }
        )  # fmt: skip
