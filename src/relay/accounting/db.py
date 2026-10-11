"""PostgreSQL connection pool (asyncpg), shared by API-key lookups and the usage ledger."""

import asyncpg


async def create_pool(dsn: str) -> asyncpg.Pool:
    """A small pool: Relay's queries are short (key lookups, batched ledger inserts).

    command_timeout bounds every query, so a stuck database cannot hold a request.
    """
    return await asyncpg.create_pool(dsn, min_size=1, max_size=10, command_timeout=5)
