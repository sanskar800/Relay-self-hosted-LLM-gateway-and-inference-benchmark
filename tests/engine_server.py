"""A FakeEngine served over a real socket by uvicorn, for failure-scenario tests.

Real sockets make failures real: a crash mid-stream drops the TCP connection (httpx
raises a genuine RemoteProtocolError) and a stall triggers httpx's own read (idle)
timeout, which an in-memory transport cannot.
"""

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn

from tests.fake_engine import FakeEngine


class EngineApp:
    """ASGI app speaking the OpenAI chat API, scripted by a FakeEngine."""

    def __init__(self, engine: FakeEngine, crash_after_tokens: int | None = None) -> None:
        self.engine = engine
        self.crash_after_tokens = crash_after_tokens

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            while (message := await receive())["type"] != "lifespan.shutdown":
                await send({"type": "lifespan.startup.complete"})
            await send({"type": "lifespan.shutdown.complete"})
            return
        if scope["path"] == "/health":
            await self._json(send, 200, {"status": "ok"})
            return
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        request = json.loads(body)
        self.engine.requests.append(request)
        if not request.get("stream"):
            await self._json(send, 200, self.engine._completion(request))
            return
        include_usage = bool((request.get("stream_options") or {}).get("include_usage"))
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")]})  # fmt: skip
        tokens_sent = 0
        async for chunk in self.engine._events(request, include_usage):
            if b'"content": "' in chunk:
                tokens_sent += 1
                if self.crash_after_tokens is not None and tokens_sent > self.crash_after_tokens:
                    # The engine dies mid-answer: uvicorn drops the connection.
                    raise RuntimeError("engine crashed (scripted)")
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    @staticmethod
    async def _json(send, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})  # fmt: skip
        await send({"type": "http.response.body", "body": data})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def serve_engine(engine: FakeEngine, crash_after_tokens: int | None = None) -> Iterator[str]:
    """Run the scripted engine on a free local port; yields its base URL."""
    port = _free_port()
    config = uvicorn.Config(
        EngineApp(engine, crash_after_tokens), host="127.0.0.1", port=port, log_level="critical"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("engine server did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(5)
