"""Readiness, draining and the graceful-shutdown server logic."""

import signal

import pytest
import uvicorn

import relay.serve as serve
from relay.lifecycle import Lifecycle, LifecycleMiddleware
from tests.unit.fakes import FakeBackend, app_with_fake, make_client

BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def test_ready_until_draining_then_503_but_still_alive() -> None:
    backend = FakeBackend()
    with make_client(app_with_fake(backend)) as client:
        ready = client.get("/readyz")
        assert ready.status_code == 200 and ready.json()["status"] == "ready"
        client.app.state.lifecycle.start_draining()
        draining = client.get("/readyz")
        alive = client.get("/healthz")
        refused = client.post("/v1/chat/completions", json=BODY)
    assert draining.status_code == 503 and draining.json()["status"] == "draining"
    assert alive.status_code == 200  # liveness stays green: do not restart a draining pod
    assert refused.status_code == 503
    assert refused.headers["retry-after"] == "1"
    assert refused.json()["error"]["type"] == "service_unavailable"
    assert backend.requests == []


def test_readyz_reports_breaker_states() -> None:
    with make_client(app_with_fake(FakeBackend("b1"))) as client:
        assert client.get("/readyz").json()["breakers"] == {"b1": "closed"}


async def test_in_flight_request_is_counted_and_finishes_while_draining() -> None:
    # Driven at the ASGI level: TestClient would run the app to completion before
    # returning any body, hiding what happens *during* a response.
    lifecycle = Lifecycle()
    seen: dict[str, object] = {}

    async def streaming_app(scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        seen["in_flight_while_streaming"] = lifecycle.in_flight
        lifecycle.start_draining()  # shutdown starts mid-response...
        for part in (b"a", b"b", b"c"):
            await send({"type": "http.response.body", "body": part, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {"type": "http", "path": "/v1/chat/completions", "method": "POST", "headers": []}
    await LifecycleMiddleware(streaming_app, lifecycle)(scope, receive, send)
    assert seen["in_flight_while_streaming"] == 1
    assert b"".join(m.get("body", b"") for m in sent) == b"abc"  # ...and still completes
    assert lifecycle.in_flight == 0


# --- the server's signal handling (no real signals) -----------------------------------


@pytest.fixture
def server(monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(serve.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr("relay.lifecycle.time.monotonic", lambda: clock["now"])
    lifecycle = Lifecycle()
    config = uvicorn.Config(app=lambda scope, receive, send: None)
    return serve.GracefulServer(config, lifecycle, grace_s=5), lifecycle, clock


async def test_first_signal_drains_and_shutdown_waits_for_the_grace_period(server) -> None:
    srv, lifecycle, clock = server
    srv.handle_exit(signal.SIGINT, None)
    assert lifecycle.draining and not srv.should_exit  # not ready, still accepting
    clock["now"] += 4.9
    assert await srv.on_tick(1) is False
    clock["now"] += 0.2
    await srv.on_tick(2)
    assert srv.should_exit  # grace period over: uvicorn stops accepting and drains


def test_second_signal_goes_straight_to_uvicorn(server) -> None:
    srv, lifecycle, _ = server
    srv.handle_exit(signal.SIGINT, None)
    srv.handle_exit(signal.SIGINT, None)
    assert srv.should_exit  # uvicorn's own handling: a second Ctrl+C escalates
