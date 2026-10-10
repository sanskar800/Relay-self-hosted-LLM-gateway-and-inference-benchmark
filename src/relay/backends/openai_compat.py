"""Shared HTTP logic for engines that speak the OpenAI chat-completions protocol."""

from collections.abc import AsyncIterator

import httpx

from relay.backends.base import (
    BackendHTTPError,
    BackendResponse,
    BackendStreamError,
    BackendTimeout,
    BackendUnavailable,
)
from relay.backends.sse import iter_sse_events

CHAT_PATH = "/v1/chat/completions"
JSON_HEADERS = {"Content-Type": "application/json"}


class OpenAICompatBackend:
    """One backend = one base URL with its own connection pool and timeouts."""

    engine = "openai-compatible"
    health_path = "/health"

    def __init__(
        self,
        name: str,
        base_url: str,
        *,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        # For streams the read timeout applies to each read: it is the maximum silence
        # between chunks, not the total duration.
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(read_timeout_s, connect=connect_timeout_s),
            transport=transport,
        )

    async def chat(self, body: bytes) -> BackendResponse:
        try:
            response = await self._client.post(CHAT_PATH, content=body, headers=JSON_HEADERS)
        except httpx.TimeoutException as exc:
            raise BackendTimeout(self.name, "timed out") from exc
        except httpx.RequestError as exc:
            raise BackendUnavailable(self.name, repr(exc)) from exc
        content_type = response.headers.get("content-type", "application/json")
        if response.status_code >= 400:
            raise BackendHTTPError(self.name, response.status_code, response.content, content_type)
        return BackendResponse(response.status_code, response.content, content_type)

    async def stream(self, body: bytes) -> "OpenAICompatStream":
        request = self._client.build_request("POST", CHAT_PATH, content=body, headers=JSON_HEADERS)
        try:
            response = await self._client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise BackendTimeout(self.name, "timed out") from exc
        except httpx.RequestError as exc:
            raise BackendUnavailable(self.name, repr(exc)) from exc
        if response.status_code != 200:
            content = await response.aread()
            await response.aclose()
            content_type = response.headers.get("content-type", "application/json")
            raise BackendHTTPError(self.name, response.status_code, content, content_type)
        return OpenAICompatStream(self.name, response)

    async def health(self) -> bool:
        try:
            response = await self._client.get(self.health_path, timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenAICompatStream:
    def __init__(self, backend: str, response: httpx.Response) -> None:
        self._backend = backend
        self._response = response

    async def events(self) -> AsyncIterator[bytes]:
        try:
            async for event in iter_sse_events(self._response.aiter_bytes()):
                yield event
        except httpx.TimeoutException as exc:
            raise BackendStreamError(self._backend, "stopped sending", timeout=True) from exc
        except httpx.TransportError as exc:
            raise BackendStreamError(self._backend, "connection lost", timeout=False) from exc

    async def aclose(self) -> None:
        await self._response.aclose()
