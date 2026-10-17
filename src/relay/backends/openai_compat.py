"""Shared HTTP logic for engines that speak the OpenAI chat-completions protocol."""

import asyncio
from collections.abc import AsyncIterator

import httpx

from relay.backends.base import (
    BackendHTTPError,
    BackendResponse,
    BackendStreamError,
    BackendTimeout,
    BackendUnavailable,
    Timeouts,
)
from relay.backends.sse import iter_sse_events

CHAT_PATH = "/v1/chat/completions"
JSON_HEADERS = {"Content-Type": "application/json"}


class OpenAICompatBackend:
    """One backend = one base URL with its own connection pool and timeouts."""

    engine = "openai-compatible"
    health_path = "/health"
    # Reuse connections between requests unless the engine is known to drop them.
    keep_alive_default = True

    def __init__(
        self,
        name: str,
        base_url: str,
        *,
        timeouts: Timeouts | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        keep_alive: bool | None = None,
    ) -> None:
        self.name = name
        self.timeouts = timeouts or Timeouts()
        self.keep_alive = self.keep_alive_default if keep_alive is None else keep_alive
        # httpx's read timeout applies to each socket read: for streams that is the idle
        # timeout (maximum silence between chunks).
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(self.timeouts.idle, connect=self.timeouts.connect),
            # Without keep-alive every request opens a new connection (well under 1 ms on
            # a local network) instead of reusing one the engine may have closed.
            # Otherwise httpx's defaults (100 connections, 20 kept alive).
            limits=httpx.Limits(
                max_connections=100, max_keepalive_connections=20 if self.keep_alive else 0
            ),
            transport=transport,
        )

    @property
    def http_client(self) -> httpx.AsyncClient:
        """The connection pool to this backend (e.g. for tracing instrumentation)."""
        return self._client

    async def chat(self, body: bytes) -> BackendResponse:
        # Non-streaming: the engine sends nothing until the answer is complete, so the
        # only meaningful bound is the total timeout.
        t = self.timeouts
        try:
            response = await self._client.post(
                CHAT_PATH,
                content=body,
                headers=JSON_HEADERS,
                timeout=httpx.Timeout(t.total, connect=t.connect),
            )
        except httpx.ConnectTimeout as exc:
            raise BackendUnavailable(self.name, "connect timeout") from exc
        except httpx.TimeoutException as exc:
            raise BackendTimeout(self.name, f"no answer within {t.total}s") from exc
        except httpx.RequestError as exc:
            raise BackendUnavailable(self.name, repr(exc)) from exc
        content_type = response.headers.get("content-type", "application/json")
        if response.status_code >= 400:
            raise BackendHTTPError(self.name, response.status_code, response.content, content_type)
        return BackendResponse(response.status_code, response.content, content_type)

    async def stream(self, body: bytes) -> "OpenAICompatStream":
        """Send the request and wait for the FIRST event before returning (see Backend)."""
        t = self.timeouts
        loop = asyncio.get_running_loop()
        deadline = loop.time() + t.total
        request = self._client.build_request("POST", CHAT_PATH, content=body, headers=JSON_HEADERS)
        response: httpx.Response | None = None
        try:
            async with asyncio.timeout(t.first_token):
                response = await self._client.send(request, stream=True)
                if response.status_code != 200:
                    content = await response.aread()
                    content_type = response.headers.get("content-type", "application/json")
                    raise BackendHTTPError(self.name, response.status_code, content, content_type)
                events = iter_sse_events(response.aiter_bytes())
                first = await anext(events)
        except BaseException as exc:
            if response is not None:
                await response.aclose()
            if isinstance(exc, BackendHTTPError):
                raise
            if isinstance(exc, httpx.ConnectTimeout):
                raise BackendUnavailable(self.name, "connect timeout") from exc
            if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
                raise BackendTimeout(self.name, f"no first event within {t.first_token}s") from exc
            if isinstance(exc, StopAsyncIteration):
                raise BackendUnavailable(self.name, "stream closed before any event") from exc
            if isinstance(exc, httpx.RequestError):
                raise BackendUnavailable(self.name, repr(exc)) from exc
            raise
        return OpenAICompatStream(self.name, response, events, first, deadline)

    async def health(self) -> bool:
        try:
            response = await self._client.get(self.health_path, timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenAICompatStream:
    """A stream whose first event has already arrived."""

    def __init__(
        self,
        backend: str,
        response: httpx.Response,
        events: AsyncIterator[bytes],
        first: bytes,
        deadline: float,
    ) -> None:
        self._backend = backend
        self._response = response
        self._events = events
        self._first = first
        self._deadline = deadline  # event-loop time by which the whole request must end

    async def events(self) -> AsyncIterator[bytes]:
        yield self._first
        loop = asyncio.get_running_loop()
        while True:
            remaining = self._deadline - loop.time()
            if remaining <= 0:
                raise BackendStreamError(self._backend, "total timeout", timeout=True)
            try:
                # The deadline wraps only this read, never a `yield`: a timeout spanning
                # a yield would fire while the caller (sending to the client) runs.
                async with asyncio.timeout(remaining):
                    event = await anext(self._events)
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                raise BackendStreamError(self._backend, "total timeout", timeout=True) from exc
            except httpx.TimeoutException as exc:
                raise BackendStreamError(self._backend, "idle timeout", timeout=True) from exc
            except httpx.TransportError as exc:
                raise BackendStreamError(self._backend, "connection lost", timeout=False) from exc
            yield event

    async def aclose(self) -> None:
        await self._events.aclose()  # type: ignore[attr-defined]  (an async generator)
        await self._response.aclose()
