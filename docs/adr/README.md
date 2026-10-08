# Architecture Decision Records

An ADR records **one** significant decision: the context, the options considered, the choice and its consequences. We write one only when we actually make a decision, not in advance.

## Index

| # | Title | Status |
|---|---|---|
| none yet | | |

Expected during the build (created only when decided):

- ADR-001: Inference engines and benchmark hardware (vLLM + llama.cpp; free Colab GPU instead of rented GPU)
- ADR-002: Rate-limiter behaviour when Redis is unavailable (fail-open vs fail-closed)
- ADR-003: Streaming retry policy (no retries after the first token)

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
