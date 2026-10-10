"""Errors in the shape OpenAI clients parse: {"error": {"message", "type", "param", "code"}}.

Every error a client can receive from Relay has this shape, so SDKs raise typed
exceptions (BadRequestError, NotFoundError, RateLimitError, ...) with a useful message.
"""

import json
import logging

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from relay.api.schemas import ErrorDetail, ErrorResponse
from relay.backends.base import BackendHTTPError

log = logging.getLogger("relay.errors")


def model_not_found(model: str) -> JSONResponse:
    """OpenAI's answer for an unknown model: 404 with code model_not_found."""
    message = f"The model `{model}` does not exist or you do not have access to it."
    return openai_error(404, message, "invalid_request_error", "model", "model_not_found")


def openai_error(
    status_code: int,
    message: str,
    error_type: str,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    detail = ErrorDetail(message=message, type=error_type, param=param, code=code)
    body = ErrorResponse(error=detail)
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's default is a 422 with its own shape; OpenAI clients expect a 400."""
    first = exc.errors()[0]
    # loc looks like ("body", "messages", 0, "role"); OpenAI names it "messages.0.role".
    loc = [str(part) for part in first["loc"] if part != "body"]
    param = ".".join(loc) or None
    if first["type"] == "json_invalid":
        message = "Request body must be valid JSON."
    else:
        message = f"{param}: {first['msg']}" if param else first["msg"]
    return openai_error(400, message, "invalid_request_error", param)


def backend_error_response(exc: BackendHTTPError) -> Response:
    """Pass an OpenAI-shaped backend error through unchanged; wrap anything else.

    An engine's own errors are already OpenAI-shaped. An HTML or plain-text body (e.g.
    from a proxy in front of the engine) would confuse clients, so it is replaced by an
    OpenAI error with the same status. The original body goes to the server log only.
    """
    try:
        data = json.loads(exc.content)
    except ValueError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        return Response(exc.content, exc.status_code, media_type="application/json")
    log.warning(
        "non-OpenAI error body from backend %s (HTTP %s): %r",
        exc.backend,
        exc.status_code,
        exc.content[:200],
    )
    message = f"Backend returned HTTP {exc.status_code}."
    return openai_error(exc.status_code, message, "backend_error")


async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Unknown paths and wrong methods; FastAPI's default is {"detail": ...}."""
    if exc.status_code == 404:
        message = f"Invalid URL ({request.method} {request.url.path})."
    else:
        message = str(exc.detail)
    return openai_error(exc.status_code, message, "invalid_request_error")


async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    """A bug in Relay: full traceback in the server log, a generic message to the client."""
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return openai_error(500, "Internal error in Relay.", "server_error")


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unhandled_error)

    # FastAPI documents a 422 for every endpoint with a body, but the handler above
    # turns validation errors into 400s, so remove it from the OpenAPI schema.
    default_openapi = app.openapi

    def openapi_without_422() -> dict:
        schema = default_openapi()
        for path in schema.get("paths", {}).values():
            for operation in path.values():
                operation.get("responses", {}).pop("422", None)
        return schema

    app.openapi = openapi_without_422
