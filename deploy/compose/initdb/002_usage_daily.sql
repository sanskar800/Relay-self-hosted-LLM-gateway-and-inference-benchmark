-- Daily usage per tenant and model, derived from the usage ledger.
-- CREATE OR REPLACE makes this safe to re-run on an existing database (`make db-apply`).
-- Days are UTC. Sums ignore unknown values, so the view also counts the requests whose
-- usage or cost is not exact: an aggregate should never look more complete than it is.

CREATE OR REPLACE VIEW usage_daily AS
SELECT
    tenant_id,
    (occurred_at AT TIME ZONE 'UTC')::date                  AS day,
    model,
    count(*)                                                AS requests,
    count(*) FILTER (WHERE status_code >= 400)              AS failed_requests,
    coalesce(sum(prompt_tokens), 0)                         AS prompt_tokens,
    coalesce(sum(completion_tokens), 0)                     AS completion_tokens,
    coalesce(sum(prompt_tokens), 0) + coalesce(sum(completion_tokens), 0) AS total_tokens,
    count(*) FILTER (WHERE usage_source <> 'engine')        AS requests_without_exact_usage,
    -- NULL when no request that day has a price; partial when some do (see below).
    sum(cost_gbp)                                           AS cost_gbp,
    count(*) FILTER (WHERE cost_gbp IS NULL)                AS unpriced_requests
FROM usage_events
GROUP BY tenant_id, (occurred_at AT TIME ZONE 'UTC')::date, model;
