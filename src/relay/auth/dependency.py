"""FastAPI dependency: every API route requires `Authorization: Bearer rk_...`.

As a dependency it runs before body validation, so unauthenticated callers get 401
rather than details about an API they may not use. HTTPBearer also gives Swagger an
"Authorize" button.
"""

import logging
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from opentelemetry import trace

from relay.api.errors import APIError
from relay.auth.keys import KeyStore, KeyStoreUnavailable, Principal, display_prefix, hash_key

log = logging.getLogger("relay.auth")

_bearer = HTTPBearer(auto_error=False, description="Relay API key (rk_...)")
_CHALLENGE = {"WWW-Authenticate": "Bearer"}


async def require_api_key(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    if credentials is None or not credentials.credentials.strip():
        raise APIError(
            401,
            "Missing API key. Send it as 'Authorization: Bearer rk_...'.",
            "invalid_request_error",
            code="missing_api_key",
            headers=_CHALLENGE,
        )

    raw = credentials.credentials.strip()
    store: KeyStore = request.app.state.key_store
    tracer: trace.Tracer = getattr(request.app.state, "tracer", trace.NoOpTracer())
    try:
        with tracer.start_as_current_span("relay.auth", record_exception=False) as span:
            principal = await store.lookup(hash_key(raw))
            span.set_attribute("relay.auth.outcome", "ok" if principal else "invalid_key")
    except KeyStoreUnavailable as exc:
        # Fail closed: never let a request through because we could not check the key.
        log.error("key store unavailable: %s", exc)
        raise APIError(
            503, "Authentication is temporarily unavailable.", "service_unavailable"
        ) from exc

    if principal is None:
        # Same answer for unknown, revoked and disabled: callers learn nothing extra.
        # Only a short prefix is echoed (and logged), never the key itself.
        raise APIError(
            401,
            f"Incorrect API key provided: {display_prefix(raw)}...",
            "invalid_request_error",
            code="invalid_api_key",
            headers=_CHALLENGE,
        )

    request.state.principal = principal
    return principal
