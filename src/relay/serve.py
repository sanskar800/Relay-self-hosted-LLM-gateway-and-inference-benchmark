"""Production-style entry point with graceful shutdown: `python -m relay.serve`.

    signal (SIGTERM, Ctrl+C, Ctrl+Break on Windows)
      -> draining: /readyz 503, new /v1 requests 503, in-flight requests continue
      -> after RELAY_SHUTDOWN_GRACE_S: stop accepting connections
      -> wait up to RELAY_DRAIN_TIMEOUT_S for in-flight requests, then cancel the rest
      -> app shutdown (usage ledger flush, pools closed)
    A second Ctrl+C forces an immediate exit.

`make run` (uvicorn --reload) is for development and does not have the grace period.
"""

import time
from types import FrameType

import uvicorn

from relay.config import Settings
from relay.lifecycle import Lifecycle
from relay.main import create_app


class GracefulServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, lifecycle: Lifecycle, grace_s: float) -> None:
        super().__init__(config)
        self.lifecycle = lifecycle
        self.grace_s = grace_s

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        # Runs inside a signal handler: only record the request. The first signal starts
        # draining; a repeated signal falls through to uvicorn (Ctrl+C twice = force quit).
        if self.lifecycle.draining or self.grace_s <= 0:
            super().handle_exit(sig, frame)
            return
        self._captured_signals.append(sig)
        self.lifecycle.start_draining()

    async def on_tick(self, counter: int) -> bool:
        # Called about every 0.1 s by uvicorn's main loop: True = start shutting down.
        since = self.lifecycle.draining_since
        if since is not None and time.monotonic() - since >= self.grace_s:
            self.should_exit = True
        return await super().on_tick(counter)


def main() -> None:
    settings = Settings()  # RELAY_HOST, RELAY_PORT, RELAY_SHUTDOWN_GRACE_S, RELAY_DRAIN_TIMEOUT_S
    app = create_app(settings)
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        timeout_graceful_shutdown=settings.drain_timeout_s,
    )
    GracefulServer(config, app.state.lifecycle, settings.shutdown_grace_s).run()


if __name__ == "__main__":
    main()
