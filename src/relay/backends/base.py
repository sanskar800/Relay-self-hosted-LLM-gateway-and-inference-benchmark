"""What the gateway needs from an inference backend, and how backend failures look.

The gateway only sees these types: it never imports httpx. Each failure type tells the
caller what is safe to do next (e.g. retry only before streaming has started).
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


class BackendError(Exception):
    """Base class for every backend failure."""

    def __init__(self, backend: str, message: str) -> None:
        super().__init__(f"{backend}: {message}")
        self.backend = backend


class BackendUnavailable(BackendError):
    """No response at all (connection refused, reset, DNS). Safe to retry elsewhere."""


class BackendTimeout(BackendError):
    """No response within the timeout, before any data was sent to the client."""


class BackendHTTPError(BackendError):
    """The backend answered with an error status. Its body is passed to the client."""

    def __init__(self, backend: str, status_code: int, content: bytes, content_type: str) -> None:
        super().__init__(backend, f"HTTP {status_code}")
        self.status_code = status_code
        self.content = content
        self.content_type = content_type


class BackendStreamError(BackendError):
    """The stream broke after it started. Never retried: the client already has tokens."""

    def __init__(self, backend: str, message: str, *, timeout: bool) -> None:
        super().__init__(backend, message)
        self.timeout = timeout


@dataclass(frozen=True)
class BackendResponse:
    status_code: int
    content: bytes
    content_type: str


class BackendStream(Protocol):
    """An open streaming response from a backend."""

    def events(self) -> AsyncIterator[bytes]:
        """Whole SSE events, each yielded as soon as it is complete.

        Raises BackendStreamError if the stream breaks.
        """
        ...

    async def aclose(self) -> None:
        """Release the connection. Closing early makes the engine stop generating."""
        ...


class Backend(Protocol):
    """An OpenAI-compatible inference server. Structural: fakes need not inherit."""

    name: str
    engine: str

    async def chat(self, body: bytes) -> BackendResponse:
        """Non-streaming chat completion. Raises a BackendError subclass on failure."""
        ...

    async def stream(self, body: bytes) -> BackendStream:
        """Start a streaming chat completion; returns once the backend answered 200."""
        ...

    async def health(self) -> bool:
        """True if the backend can serve requests right now."""
        ...

    async def aclose(self) -> None:
        """Close connection pools on shutdown."""
        ...
