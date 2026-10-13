# ADR-003: Retry only before the first token

- **Status:** Accepted
- **Date:** 2026-10-13

## Context

Relay streams answers token by token. Backends fail in different ways: refused connections, a server that is still loading its model (HTTP 503), a backend that accepts a request but never answers, or a connection that dies halfway through an answer.

Retrying can hide short failures from users, but a retry is only safe if the client cannot tell it happened. Once the first token of an answer has been delivered, the HTTP status (200) is committed and the client is already displaying text. A retry would restart generation: the user would see the beginning of the answer twice, or a different answer, because sampling is random.

Retries also cost something. A backend that is slow because it is overloaded gets slower if every request is sent twice. And the OpenAI SDK already retries 5xx responses twice by default, so client and gateway retries multiply.

## Options considered

1. **Never retry.** Simple, but every brief blip (a restarting backend, a dropped connection) becomes a user-visible error.
2. **Retry any failure, at any time.** Hides most failures, but duplicates or contradicts text already shown, and multiplies load on struggling backends.
3. **Retry only before the first token, only failures where trying again can help, with few jittered attempts.**

## Decision

**Option 3.**

- **The boundary is explicit:** `Backend.stream()` returns only after the backend has produced its first event. Every failure up to that point is an ordinary exception while nothing has reached the client. After it, failures become an SSE error event and the stream closes; they are never retried.
- **Retried:** `BackendUnavailable` (connection refused or reset, connect timeout, stream closed before any event) and HTTP **503**. The backend did no work for us, or said it is temporarily unable to.
- **Not retried:** 4xx (the request itself is wrong), other 5xx, and `BackendTimeout` (the backend accepted the request and may still be working on it; repeating it would double its load). Moving to *another* backend is the job of routing and fallback, not of retries.
- **How:** at most 3 attempts per backend, waiting `random(0, min(1 s, 0.1 s × 2^retry))` between them (exponential backoff with full jitter, so many requests do not retry in lockstep). Configurable under `retry:` in `config/relay.yaml`.
- **Non-streaming requests** follow the same rule: nothing is sent to the client before the full answer exists, so the same failures are retried.

## Consequences

- Short failures (a backend restarting, a dropped connection) are invisible to clients.
- An answer is never duplicated or restarted after the client has seen part of it; a mid-stream failure ends with an OpenAI-format error event instead.
- Waiting for the first event before committing the response adds no latency: the client could not receive anything earlier anyway.
- **Retries cost time against a backend that is really down:** each attempt waits for the connection to fail. Observed on Windows (2026-10-13): a refused connection to a stopped container takes about 2 s, so 3 attempts took about 6.3 s before the 502. Routing should stop sending requests to a backend that keeps failing (a circuit breaker), so that only brief failures pay this cost.
- Client retries (the OpenAI SDK retries 5xx twice) still multiply with these. Keeping the gateway's attempts few and jittered limits the effect; clients that need tight latency can set `max_retries=0`.
