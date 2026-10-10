"""Request size and parameter limits: cheap rejections before a request costs memory or GPU.

The body-size limit must be enforced while the body is being received: FastAPI reads the
whole body before an endpoint runs, so a check inside the endpoint would come too late.
"""

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from relay.api.errors import openai_error
from relay.api.schemas import ChatCompletionRequest
from relay.config import LimitsConfig


def _too_large(max_bytes: int) -> str:
    return f"Request body too large (limit {max_bytes} bytes)."


class BodySizeLimitMiddleware:
    """Reject bodies over `max_bytes` with 413, without ever holding them in memory."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Fast path: the declared size is already too big; read nothing.
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            response = openai_error(413, _too_large(self.max_bytes), "invalid_request_error")
            await response(scope, receive, send)
            return

        # Slow path: no (or a wrong) Content-Length, e.g. chunked uploads. Count bytes as
        # they arrive and stop once past the limit. FastAPI re-raises HTTPException from
        # body reading, so the app's handler turns it into an OpenAI-format 413.
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise HTTPException(status_code=413, detail=_too_large(self.max_bytes))
            return message

        await self.app(scope, limited_receive, send)


def check_request_limits(
    payload: ChatCompletionRequest, limits: LimitsConfig
) -> JSONResponse | None:
    """Parameter limits on a validated request; returns an error response or None."""
    if len(payload.messages) > limits.max_messages:
        message = f"Too many messages: {len(payload.messages)} (limit {limits.max_messages})."
        return openai_error(400, message, "invalid_request_error", "messages")
    for field in ("max_tokens", "max_completion_tokens"):
        value = getattr(payload, field)
        if value is not None and value > limits.max_tokens:
            message = f"{field} is {value}; the limit is {limits.max_tokens}."
            return openai_error(400, message, "invalid_request_error", field)
    return None
