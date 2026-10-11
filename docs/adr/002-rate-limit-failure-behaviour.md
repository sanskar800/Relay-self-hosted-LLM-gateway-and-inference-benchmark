# ADR-002: Rate limiting when Redis is unavailable

- **Status:** Accepted
- **Date:** 2026-10-11

## Context

Relay rate-limits each tenant with token buckets kept in Redis, so that every gateway instance shares one view of each tenant's usage (requests/min and LLM tokens/min). Redis is a separate process and can be unavailable: restarted, overloaded or unreachable over the network.

Before this decision, an unreachable Redis made every chat request fail with HTTP 500. The limiter then decided availability for the whole gateway.

Relay is an **internal platform**: tenants are teams in one organisation, and tokens cost GPU time on hardware we already run, not money per call to a third party. The requirements brief asks for reliability (a Redis or Postgres outage must not take requests down) and fair sharing of a scarce GPU.

Authentication is a different case: it **fails closed** (503) when it cannot verify a key, because admitting an unverified caller is a security failure. Rate limiting protects fairness and capacity, not identity.

## Options considered

1. **Fail closed:** reject every request (503) while Redis is down.
   - Pro: limits are never exceeded; right for a paid API where unmetered use costs real money.
   - Con: Redis becomes a single point of failure for the whole service; a fairness mechanism takes everything down.
2. **Fail open without limits:** admit everything while Redis is down.
   - Pro: maximum availability; simplest.
   - Con: during the outage one tenant can saturate the GPU and starve the others, which is exactly what the limiter exists to prevent.
3. **Fail open to a local in-memory limiter:** each Relay instance enforces the same per-tenant limits with in-process token buckets until Redis is back.
   - Pro: requests keep being served *and* stay bounded; reuses the same bucket maths (`InMemoryRateLimiter`).
   - Con: with N instances a tenant can get up to N × its limit, plus a small burst when switching back, because each instance (and Redis) has its own full bucket.

## Decision

**Option 3.** When the Redis limiter raises `RateLimiterUnavailable`, Relay enforces the same limits with a local `InMemoryRateLimiter` (`relay.limits.failover.FailoverRateLimiter`).

- After a failure Redis is **skipped for 5 s** before being retried, so requests do not each wait for a connection timeout (1 s).
- Relay logs one **warning** when it degrades and one **info** line when Redis is reachable again; per-request logging would flood the log during an outage.
- Settlements (refunds) go to whichever store is currently enforcing limits.
- Relay starts even if Redis is down (unlike Postgres, which it needs for authentication).

## Consequences

- A Redis outage no longer causes errors; traffic stays limited per instance.
- **Accepted inaccuracy:** up to N × the limit with N instances during an outage, and a request admitted just before a switch may be settled on the other store. Today N = 1.
- An outage must be *visible*, since nothing fails: the limiter exposes `degraded` and `fallback_decisions`, to become a metric and an alert with observability.
- Verified on 2026-10-11 by stopping the Redis container under a running Relay: a tenant limited to 3/min got 200, 200, 200, then 429; only the first request paid the ~1 s detection timeout; the shared bucket was used again after Redis restarted.

**When to revisit:** if Relay runs many instances, or if tokens start to cost money per call (e.g. a paid hosted fallback). Options then include dividing each tenant's limit by the instance count while degraded, or failing closed for paid backends only.
