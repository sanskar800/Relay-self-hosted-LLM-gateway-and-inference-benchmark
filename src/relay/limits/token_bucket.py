"""Per-tenant token-bucket rate limiting: requests/min and LLM tokens/min.

Tokens are reserved before a request (estimated prompt + max_tokens) and settled after
it (refund `reserved - actual`), because the completion length is unknown up front.

RedisRateLimiter shares state across gateway instances; its Lua script makes
"refill, check both buckets, take from both" one atomic step. InMemoryRateLimiter
applies the same maths in-process (tests, fallback, single-instance setups).
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from relay.auth.keys import Principal

LUA = (Path(__file__).parent / "token_bucket.lua").read_text(encoding="utf-8")


@dataclass(frozen=True)
class Buckets:
    """Bucket sizes for one tenant: capacity per minute, refilled continuously."""

    req_capacity: float
    req_rate: float  # per second
    tok_capacity: float
    tok_rate: float  # per second

    @classmethod
    def for_principal(cls, p: Principal) -> "Buckets":
        return cls(
            p.requests_per_minute,
            p.requests_per_minute / 60,
            p.tokens_per_minute,
            p.tokens_per_minute / 60,
        )

    @property
    def ttl_seconds(self) -> int:
        # After a full refill an idle bucket equals a missing key, so it can expire.
        full = max(self.req_capacity / self.req_rate, self.tok_capacity / self.tok_rate)
        return math.ceil(full) + 60


@dataclass(frozen=True)
class Decision:
    allowed: bool
    buckets: Buckets
    remaining_requests: float
    remaining_tokens: float
    retry_after_s: float = 0.0
    # The request needs more tokens than the bucket can ever hold.
    too_large: bool = False

    @property
    def reset_requests_s(self) -> float:
        return max(0.0, self.buckets.req_capacity - self.remaining_requests) / self.buckets.req_rate

    @property
    def reset_tokens_s(self) -> float:
        return max(0.0, self.buckets.tok_capacity - self.remaining_tokens) / self.buckets.tok_rate


class RateLimiterUnavailable(Exception):
    """The shared limiter state (Redis) could not be reached."""


class RateLimiter(Protocol):
    async def acquire(self, principal: Principal, tokens: int) -> Decision:
        """Take 1 request and `tokens` tokens, or nothing. Raises RateLimiterUnavailable."""
        ...

    async def settle(self, principal: Principal, refund: int) -> None:
        """Return `refund` tokens (negative = charge more) once actual usage is known."""
        ...


def _too_large(b: Buckets, tokens: int) -> Decision | None:
    if tokens > b.tok_capacity:
        return Decision(False, b, b.req_capacity, b.tok_capacity, too_large=True)
    return None


class InMemoryRateLimiter:
    """The same bucket maths as the Lua script, in this process only."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._state: dict[tuple[str, str], tuple[float, float]] = {}

    def _current(self, key: tuple[str, str], capacity: float, rate: float, now: float) -> float:
        tokens, ts = self._state.get(key, (capacity, now))
        return min(capacity, tokens + max(0.0, now - ts) * rate)

    async def acquire(self, principal: Principal, tokens: int) -> Decision:
        b = Buckets.for_principal(principal)
        if (rejected := _too_large(b, tokens)) is not None:
            return rejected
        now = self._clock()
        rk, tk = (principal.tenant_id, "req"), (principal.tenant_id, "tok")
        req = self._current(rk, b.req_capacity, b.req_rate, now)
        tok = self._current(tk, b.tok_capacity, b.tok_rate, now)
        allowed = req >= 1 and tok >= tokens
        retry_after = 0.0
        if allowed:
            req, tok = req - 1, tok - tokens
        else:
            if req < 1:
                retry_after = (1 - req) / b.req_rate
            if tok < tokens:
                retry_after = max(retry_after, (tokens - tok) / b.tok_rate)
        self._state[rk], self._state[tk] = (req, now), (tok, now)
        return Decision(allowed, b, req, tok, retry_after)

    async def settle(self, principal: Principal, refund: int) -> None:
        b = Buckets.for_principal(principal)
        now = self._clock()
        tk = (principal.tenant_id, "tok")
        self._state[tk] = (
            min(b.tok_capacity, self._current(tk, b.tok_capacity, b.tok_rate, now) + refund),
            now,
        )


class RedisRateLimiter:
    def __init__(self, client: aioredis.Redis, prefix: str = "relay:rl") -> None:
        self._client = client
        self._script = client.register_script(LUA)
        self._prefix = prefix

    def _keys(self, tenant_id: str) -> list[str]:
        # {tenant} is a Redis Cluster hash tag: both keys map to the same slot, which a
        # multi-key script requires.
        return [f"{self._prefix}:{{{tenant_id}}}:req", f"{self._prefix}:{{{tenant_id}}}:tok"]

    async def _run(self, op: str, principal: Principal, amount: int) -> list[float]:
        b = Buckets.for_principal(principal)
        args = [op, b.req_capacity, b.req_rate, b.tok_capacity, b.tok_rate, amount, b.ttl_seconds]
        try:
            result = await self._script(keys=self._keys(principal.tenant_id), args=args)
        except (RedisError, OSError) as exc:
            raise RateLimiterUnavailable(repr(exc)) from exc
        return [float(x) for x in result]

    async def acquire(self, principal: Principal, tokens: int) -> Decision:
        b = Buckets.for_principal(principal)
        if (rejected := _too_large(b, tokens)) is not None:
            return rejected
        allowed, req, tok, retry_after = await self._run("acquire", principal, tokens)
        return Decision(bool(allowed), b, req, tok, retry_after)

    async def settle(self, principal: Principal, refund: int) -> None:
        await self._run("settle", principal, refund)
