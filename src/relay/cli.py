"""relay-admin: manage tenants and API keys in Relay's Postgres.

    relay-admin create-tenant support-bot --name "Support bot" \
        --models qwen2.5-1.5b-instruct --rpm 60 --tpm 20000 --max-tokens 512
    relay-admin create-key support-bot --name laptop     # prints the key ONCE
    relay-admin list-keys [support-bot]                   # prefixes only, never keys
    relay-admin revoke-key rk_8f3aZ                      # by prefix or key id
"""

import argparse
import asyncio
import sys

import asyncpg

from relay.auth.keys import generate_key
from relay.config import Settings


async def create_tenant(conn: asyncpg.Connection, args: argparse.Namespace) -> None:
    await conn.execute(
        "INSERT INTO tenants (id, name, allowed_models, requests_per_minute,"
        " tokens_per_minute, max_tokens_cap) VALUES ($1, $2, $3, $4, $5, $6)",
        args.tenant,
        args.name or args.tenant,
        args.models,
        args.rpm,
        args.tpm,
        args.max_tokens,
    )
    print(
        f"created tenant {args.tenant}: models={args.models} rpm={args.rpm} tpm={args.tpm}"
        f" max_tokens={args.max_tokens or 'global limit'}"
    )


async def create_key(conn: asyncpg.Connection, args: argparse.Namespace) -> None:
    key = generate_key()
    key_id = await conn.fetchval(
        "INSERT INTO api_keys (tenant_id, key_hash, key_prefix, name)"
        " VALUES ($1, $2, $3, $4) RETURNING id",
        args.tenant,
        key.key_hash,
        key.display_prefix,
        args.name,
    )
    print(f"created key {key_id} ({key.display_prefix}...) for tenant {args.tenant}")
    print("Store this key now; it cannot be shown again:")
    print(key.raw)


async def list_keys(conn: asyncpg.Connection, args: argparse.Namespace) -> None:
    rows = await conn.fetch(
        "SELECT k.id, k.tenant_id, k.key_prefix, k.name, k.created_at, k.revoked_at"
        " FROM api_keys k WHERE $1::text IS NULL OR k.tenant_id = $1"
        " ORDER BY k.tenant_id, k.created_at",
        args.tenant,
    )
    for r in rows:
        state = f"revoked {r['revoked_at']:%Y-%m-%d %H:%M}" if r["revoked_at"] else "active"
        print(
            f"{r['tenant_id']:<20} {r['key_prefix']}...  {state:<24} {r['name'] or ''}"
            f"  id={r['id']}"
        )
    if not rows:
        print("no keys")


async def revoke_key(conn: asyncpg.Connection, args: argparse.Namespace) -> None:
    # In a transaction: if the prefix matches more than one key, raising rolls the
    # UPDATE back, so an ambiguous prefix revokes nothing.
    async with conn.transaction():
        rows = await conn.fetch(
            "UPDATE api_keys SET revoked_at = now()"
            " WHERE revoked_at IS NULL AND (key_prefix = $1 OR id::text = $1)"
            " RETURNING id, tenant_id, key_prefix",
            args.key,
        )
        if len(rows) != 1:
            found = len(rows)
            raise SystemExit(
                f"expected exactly one active key matching {args.key!r}, found {found}"
            )
    r = rows[0]
    print(
        f"revoked key {r['key_prefix']}... (tenant {r['tenant_id']}, id {r['id']})."
        " Relay instances stop accepting it within the key cache TTL (30 s)."
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="relay-admin", description="Manage Relay tenants and keys.")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("create-tenant", help="create a tenant with its model list and limits")
    t.add_argument("tenant", help="id: lowercase letters, digits and dashes")
    t.add_argument("--name")
    t.add_argument("--models", nargs="+", required=True, help="public model names allowed")
    t.add_argument("--rpm", type=int, required=True, help="requests per minute")
    t.add_argument("--tpm", type=int, required=True, help="tokens per minute")
    t.add_argument("--max-tokens", type=int, help="per-request max_tokens cap")
    t.set_defaults(handler=create_tenant)

    k = sub.add_parser("create-key", help="create an API key (printed once)")
    k.add_argument("tenant")
    k.add_argument("--name", help="label, e.g. where the key is used")
    k.set_defaults(handler=create_key)

    lk = sub.add_parser("list-keys", help="list keys (prefixes only)")
    lk.add_argument("tenant", nargs="?")
    lk.set_defaults(handler=list_keys)

    r = sub.add_parser("revoke-key", help="revoke a key by prefix or id")
    r.add_argument("key")
    r.set_defaults(handler=revoke_key)
    return p


async def run(args: argparse.Namespace) -> None:
    conn = await asyncpg.connect(Settings().database_url)
    try:
        await args.handler(conn, args)
    finally:
        await conn.close()


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        asyncio.run(run(args))
    except asyncpg.UniqueViolationError as exc:
        sys.exit(f"already exists: {exc.detail}")
    except asyncpg.ForeignKeyViolationError:
        sys.exit("unknown tenant")
    except asyncpg.CheckViolationError as exc:
        sys.exit(f"invalid value ({exc.constraint_name})")


if __name__ == "__main__":
    main()
