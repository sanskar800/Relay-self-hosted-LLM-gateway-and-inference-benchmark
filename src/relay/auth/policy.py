"""Per-tenant policy: which models a tenant may use, and its max_tokens cap.

A model the tenant may not use gets the same 404 as a model that does not exist (as
OpenAI does), so tenants cannot probe which models exist. The real reason is logged.
"""

import logging

from relay.api.errors import APIError
from relay.api.schemas import ChatCompletionRequest
from relay.auth.keys import Principal
from relay.config import RelayConfig

log = logging.getLogger("relay.policy")


def model_not_found_error(model: str) -> APIError:
    return APIError(
        404,
        f"The model `{model}` does not exist or you do not have access to it.",
        "invalid_request_error",
        code="model_not_found",
        param="model",
    )


def visible_models(principal: Principal, config: RelayConfig) -> list[str]:
    """Models this tenant may call: configured AND allowed, in config order."""
    return [m for m in config.models if m in principal.allowed_models]


def check_model(model: str, principal: Principal, config: RelayConfig) -> None:
    if model in config.models and model in principal.allowed_models:
        return
    if model in config.models:
        log.info(
            "tenant %s requested model %r, which is not in its allow-list (answered 404)",
            principal.tenant_id,
            model,
        )
    raise model_not_found_error(model)


def check_max_tokens(payload: ChatCompletionRequest, principal: Principal) -> None:
    cap = principal.max_tokens_cap
    if cap is None:
        return
    for field in ("max_tokens", "max_completion_tokens"):
        value = getattr(payload, field)
        if value is not None and value > cap:
            raise APIError(
                400,
                f"{field} is {value}; this API key's limit is {cap}.",
                "invalid_request_error",
                param=field,
            )


def default_max_tokens(payload: ChatCompletionRequest, principal: Principal) -> int | None:
    """The cap to add when the client sent no max_tokens, so omitting it cannot bypass it."""
    if principal.max_tokens_cap is None:
        return None
    if payload.max_tokens is not None or payload.max_completion_tokens is not None:
        return None
    return principal.max_tokens_cap
