# Architecture Decision Records

An ADR records **one** significant decision: the context, the options considered, the choice and its consequences. One is written only when a decision is actually made, not in advance.

## Index

| # | Decision | Status |
|---|---|---|
| [001](001-inference-engines-and-hardware.md) | Inference engines, hardware and the benchmark matrix: llama.cpp for development, vLLM on a free Colab T4 for the benchmark; comparisons only within one machine | Accepted (2026-10-16); amended 2026-10-25: Colab T4 only, local RTX 4050 runs dropped |
| [002](002-rate-limit-failure-behaviour.md) | Rate limiting when Redis is unavailable: fail open to the same limits enforced locally, per instance | Accepted (2026-10-11) |
| [003](003-streaming-retry-policy.md) | Retry only before the first streamed token: connection failures and 503 only, few attempts with jittered backoff | Accepted (2026-10-13); follow-up 2026-10-16: retries had hidden a llama.cpp connection fault |

Smaller decisions and accepted deviations from the original plan are listed in [architecture.md §6](../architecture.md#6-deviations-from-the-original-specification); what the build and the benchmark revealed is in [findings.md](../findings.md).

## Template

Copy this to `NNN-short-title.md`:

```markdown
# ADR-NNN: <title>

- **Status:** Proposed | Accepted | Superseded by ADR-XXX
- **Date:** YYYY-MM-DD

## Context
What problem forces a decision? What constraints apply?

## Options considered
1. Option A: pros / cons
2. Option B: pros / cons

## Decision
What we chose, in one or two sentences.

## Consequences
What becomes easier, what becomes harder, what we must monitor.
How we would know this decision was wrong.
```
