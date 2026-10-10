"""In-memory test doubles that satisfy the Backend protocol without any HTTP."""

from collections.abc import AsyncIterator

from relay.backends.base import BackendError, BackendResponse


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
