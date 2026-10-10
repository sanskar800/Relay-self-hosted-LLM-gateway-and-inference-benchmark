"""Errors in the shape OpenAI clients parse: {"error": {"message", "type", "param", "code"}}."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from relay.api.schemas import ErrorDetail, ErrorResponse


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


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation_error)

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
