-- Relay schema. Run automatically by the postgres image ONLY when its data volume is
-- empty (first start). After editing this file, recreate the volume: `make db-reset`.
-- Plain SQL on purpose: no migration framework for a one-week build with no production
-- data. A real deployment would use migrations.

-- A tenant is a team. Its policy (models, limits) lives here, next to its keys.
CREATE TABLE tenants (
    id                   text PRIMARY KEY CHECK (id ~ '^[a-z0-9][a-z0-9-]{1,62}$'),
    name                 text NOT NULL,
    -- Public model names this tenant may call. Empty = none.
    allowed_models       text[] NOT NULL DEFAULT '{}',
    requests_per_minute  integer NOT NULL CHECK (requests_per_minute > 0),
    tokens_per_minute    integer NOT NULL CHECK (tokens_per_minute > 0),
    -- Per-request cap on max_tokens; NULL = only the global limit applies.
    max_tokens_cap       integer CHECK (max_tokens_cap > 0),
    created_at           timestamptz NOT NULL DEFAULT now(),
    disabled_at          timestamptz
);

-- API keys look like rk_<random>. Only a SHA-256 hash is stored, so a database leak
-- does not leak usable keys. key_prefix lets humans tell keys apart in lists and logs.
CREATE TABLE api_keys (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   text NOT NULL REFERENCES tenants (id),
    key_hash    bytea NOT NULL UNIQUE CHECK (octet_length(key_hash) = 32),
    key_prefix  text NOT NULL,
    name        text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    revoked_at  timestamptz
);

CREATE INDEX api_keys_tenant ON api_keys (tenant_id);

-- One row per request: the cost-attribution ledger.
CREATE TABLE usage_events (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    -- Unique per request, so re-sending a batch after a failure cannot double-count.
    request_id         uuid NOT NULL UNIQUE,
    occurred_at        timestamptz NOT NULL,
    tenant_id          text NOT NULL REFERENCES tenants (id),
    api_key_id         uuid REFERENCES api_keys (id),
    model              text NOT NULL,
    backend            text NOT NULL,
    stream             boolean NOT NULL,
    status_code        smallint NOT NULL,
    prompt_tokens      integer CHECK (prompt_tokens >= 0),
    completion_tokens  integer CHECK (completion_tokens >= 0),
    -- engine = reported by the engine; estimated = counted by Relay; missing = unknown.
    usage_source       text NOT NULL CHECK (usage_source IN ('engine', 'estimated', 'missing')),
    ttft_ms            real,
    duration_ms        real NOT NULL,
    -- Estimated cost from config/pricing.yaml (an assumption, not a bill).
    cost_gbp           numeric(12, 6)
);

CREATE INDEX usage_events_tenant_time ON usage_events (tenant_id, occurred_at);
