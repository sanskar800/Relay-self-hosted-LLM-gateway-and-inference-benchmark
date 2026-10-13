"""Retries before the first token (ADR-003).

A request is retried only while nothing has reached the client, and only for failures
where trying again can help without doubling the work: the backend could not be reached
(BackendUnavailable) or said it is temporarily overloaded (HTTP 503). Delays use
exponential backoff with full jitter, so many waiting requests do not retry in lockstep.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from relay.backends.base import BackendError, BackendHTTPError, BackendUnavailable

log = logging.getLogger("relay.routing")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3  # including the first try
    base_delay_s: float = 0.1
    max_delay_s: float = 1.0

    def delay(self, retry: int, rand: Callable[[], float] = random.random) -> float:
        """Full jitter: uniform in [0, min(max_delay, base * 2**retry)] for retry 0, 1, ..."""
        return rand() * min(self.max_delay_s, self.base_delay_s * 2**retry)


def is_retryable(exc: BackendError) -> bool:
    if isinstance(exc, BackendUnavailable):
        return True
    return isinstance(exc, BackendHTTPError) and exc.status_code == 503


async def call_with_retry[T](
    attempt: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rand: Callable[[], float] = random.random,
    on_retry: Callable[[BackendError], None] = lambda exc: None,
) -> T:
    """Run `attempt` until it succeeds, fails with a non-retryable error, or runs out."""
    for n in range(policy.max_attempts):
        try:
            return await attempt()
        except BackendError as exc:
            if not is_retryable(exc) or n == policy.max_attempts - 1:
                raise
            delay = policy.delay(n, rand)
            log.warning("retrying after %s (attempt %d/%d, in %.3fs)",
                        exc, n + 2, policy.max_attempts, delay)  # fmt: skip
            on_retry(exc)
            await sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
