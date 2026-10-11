"""Rate limiting that keeps working when Redis is down (ADR-002: fail open, locally).

If the shared limiter (Redis) is unreachable, each Relay process enforces the same
limits with an in-memory limiter. Traffic stays bounded instead of either stopping
(fail closed) or becoming unlimited (fail open without a limit). With N gateway
instances a tenant can get up to N x its limit during an outage: the accepted cost.

After a failure Redis is skipped for `retry_interval_s`, so requests do not each wait
for a connection timeout. State changes are logged once, not per request.
"""

import logging
import time
from collections.abc import Callable

from relay.auth.keys import Principal
from relay.limits.token_bucket import Decision, RateLimiter, RateLimiterUnavailable

log = logging.getLogger("relay.limits")


class FailoverRateLimiter:
    def __init__(
        self,
        primary: RateLimiter,
        fallback: RateLimiter,
        *,
        retry_interval_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._retry_interval_s = retry_interval_s
        self._clock = clock
        self._skip_primary_until = 0.0
        self.degraded = False  # True while limits are enforced locally (Day 4: a metric)
        self.fallback_decisions = 0

    def _use_primary(self) -> bool:
        return self._clock() >= self._skip_primary_until

    def _primary_failed(self, exc: RateLimiterUnavailable) -> None:
        self._skip_primary_until = self._clock() + self._retry_interval_s
        if not self.degraded:
            self.degraded = True
            log.warning(
                "rate limiter: Redis unavailable (%s); enforcing limits per instance "
                "(fail open to local buckets, ADR-002)",
                exc,
            )

    def _primary_ok(self) -> None:
        if self.degraded:
            self.degraded = False
            log.info("rate limiter: Redis reachable again; shared limits restored")

    async def acquire(self, principal: Principal, tokens: int) -> Decision:
        if self._use_primary():
            try:
                decision = await self._primary.acquire(principal, tokens)
            except RateLimiterUnavailable as exc:
                self._primary_failed(exc)
            else:
                self._primary_ok()
                return decision
        self.fallback_decisions += 1
        return await self._fallback.acquire(principal, tokens)

    async def settle(self, principal: Principal, refund: int) -> None:
        # Settle where limits are currently enforced. A request admitted just before a
        # switch may be settled on the other store: a small, temporary inaccuracy.
        if self._use_primary():
            try:
                await self._primary.settle(principal, refund)
            except RateLimiterUnavailable as exc:
                self._primary_failed(exc)
            else:
                return
        await self._fallback.settle(principal, refund)
