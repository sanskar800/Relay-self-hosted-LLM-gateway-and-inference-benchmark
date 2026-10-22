"""API keys: generation, hashing, and lookup of the tenant (principal) behind a key.

Keys are `rk_` + 43 random URL-safe characters (256 bits). Only SHA-256(key) is stored.
A fast hash is right here: slow hashes (bcrypt, argon2) protect low-entropy passwords
against guessing, but a random 256-bit key cannot be guessed, and a slow hash would
cost ~100 ms of CPU on every request.
"""

import hashlib
import hmac
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import asyncpg

KEY_PREFIX = "rk_"
DISPLAY_CHARS = 8  # e.g. "rk_8f3aZ" identifies a key in lists and logs without exposing it


@dataclass(frozen=True)
class NewKey:
    raw: str  # shown once, never stored
    key_hash: bytes
    display_prefix: str


@dataclass(frozen=True)
class Principal:
    """Who is calling, and the policy that applies to them."""

    tenant_id: str
    key_id: str
    key_prefix: str
    allowed_models: frozenset[str]
    requests_per_minute: int
    tokens_per_minute: int
    max_tokens_cap: int | None


def hash_key(raw: str) -> bytes:
    return hashlib.sha256(raw.encode()).digest()


def display_prefix(raw: str) -> str:
    return raw[:DISPLAY_CHARS]


def generate_key() -> NewKey:
    raw = KEY_PREFIX + secrets.token_urlsafe(32)
    return NewKey(raw=raw, key_hash=hash_key(raw), display_prefix=display_prefix(raw))


class KeyStoreUnavailable(Exception):
    """The key store could not be queried (e.g. Postgres down)."""


class KeyStore(Protocol):
    async def lookup(self, key_hash: bytes) -> Principal | None:
        """The active principal for this key hash, or None if unknown/revoked/disabled."""
        ...


class SingleKeyStore:
    """Standalone mode (no Postgres): exactly one key, held as its hash like every key.

    The one tenant may use every configured model; its limits are given by the caller.
    Meant for a single-user deployment such as a benchmark VM, not for multi-tenancy.
    """

    def __init__(
        self,
        raw_key: str,
        *,
        tenant_id: str,
        allowed_models: frozenset[str],
        requests_per_minute: int,
        tokens_per_minute: int,
    ) -> None:
        self._hash = hash_key(raw_key)
        self._principal = Principal(
            tenant_id=tenant_id,
            key_id="standalone",
            key_prefix=display_prefix(raw_key),
            allowed_models=allowed_models,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            max_tokens_cap=None,
        )

    async def lookup(self, key_hash: bytes) -> Principal | None:
        # Constant-time comparison: the hash of a guessed key must not leak by timing.
        return self._principal if hmac.compare_digest(key_hash, self._hash) else None


LOOKUP_SQL = """
SELECT k.id::text AS key_id, k.key_prefix, t.id AS tenant_id, t.allowed_models,
       t.requests_per_minute, t.tokens_per_minute, t.max_tokens_cap
FROM api_keys k JOIN tenants t ON t.id = k.tenant_id
WHERE k.key_hash = $1 AND k.revoked_at IS NULL AND t.disabled_at IS NULL
"""


class PostgresKeyStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def lookup(self, key_hash: bytes) -> Principal | None:
        try:
            row = await self._pool.fetchrow(LOOKUP_SQL, key_hash)
        except (OSError, TimeoutError, asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
            raise KeyStoreUnavailable(repr(exc)) from exc
        if row is None:
            return None
        return Principal(
            tenant_id=row["tenant_id"],
            key_id=row["key_id"],
            key_prefix=row["key_prefix"],
            allowed_models=frozenset(row["allowed_models"]),
            requests_per_minute=row["requests_per_minute"],
            tokens_per_minute=row["tokens_per_minute"],
            max_tokens_cap=row["max_tokens_cap"],
        )


class CachedKeyStore:
    """In-memory TTL cache in front of a key store.

    - Valid keys are cached for `ttl_s`: a revoked key keeps working for up to that long.
    - Unknown keys are cached for `negative_ttl_s`, so a client retrying a bad key does
      not hit the database on every request.
    - At most `max_entries` (least recently used evicted), so random keys cannot grow
      memory without bound.
    """

    def __init__(
        self,
        inner: KeyStore,
        *,
        ttl_s: float = 30.0,
        negative_ttl_s: float = 5.0,
        max_entries: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._ttl_s = ttl_s
        self._negative_ttl_s = negative_ttl_s
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[bytes, tuple[float, Principal | None]] = OrderedDict()

    async def lookup(self, key_hash: bytes) -> Principal | None:
        now = self._clock()
        cached = self._entries.get(key_hash)
        if cached is not None and cached[0] > now:
            self._entries.move_to_end(key_hash)
            return cached[1]

        principal = await self._inner.lookup(key_hash)  # may raise KeyStoreUnavailable
        ttl = self._ttl_s if principal is not None else self._negative_ttl_s
        self._entries[key_hash] = (now + ttl, principal)
        self._entries.move_to_end(key_hash)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return principal
