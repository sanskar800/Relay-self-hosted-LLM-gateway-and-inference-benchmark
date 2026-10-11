"""Rate-limit headers and 429 errors in OpenAI's format.

The OpenAI SDK backs off using `retry-after-ms` / `retry-after`; the x-ratelimit-*
headers let clients pace themselves before hitting the limit.
"""

import math

from relay.api.errors import APIError
from relay.limits.token_bucket import Decision


def _duration(seconds: float) -> str:
    return f"{seconds:.3f}".rstrip("0").rstrip(".") + "s"


def ratelimit_headers(d: Decision) -> dict[str, str]:
    return {
        "x-ratelimit-limit-requests": str(int(d.buckets.req_capacity)),
        "x-ratelimit-limit-tokens": str(int(d.buckets.tok_capacity)),
        "x-ratelimit-remaining-requests": str(max(0, math.floor(d.remaining_requests))),
        "x-ratelimit-remaining-tokens": str(max(0, math.floor(d.remaining_tokens))),
        "x-ratelimit-reset-requests": _duration(d.reset_requests_s),
        "x-ratelimit-reset-tokens": _duration(d.reset_tokens_s),
    }


def rate_limited_error(d: Decision, reserved: int) -> APIError:
    if d.too_large:
        # Can never fit in the bucket: retrying will not help, so no Retry-After.
        return APIError(
            429,
            f"Request too large: it may use about {reserved} tokens, but this API key's"
            f" limit is {int(d.buckets.tok_capacity)} tokens per minute."
            " Lower max_tokens or shorten the prompt.",
            "tokens",
            code="rate_limit_exceeded",
            headers=ratelimit_headers(d),
        )
    kind, limit = (
        ("requests", d.buckets.req_capacity)
        if d.remaining_requests < 1
        else ("tokens", d.buckets.tok_capacity)
    )
    return APIError(
        429,
        f"Rate limit reached for {kind} per minute (limit {int(limit)})."
        f" Try again in {d.retry_after_s:.1f}s.",
        kind,
        code="rate_limit_exceeded",
        headers={
            **ratelimit_headers(d),
            "retry-after": str(max(1, math.ceil(d.retry_after_s))),
            "retry-after-ms": str(max(1, math.ceil(d.retry_after_s * 1000))),
        },
    )
