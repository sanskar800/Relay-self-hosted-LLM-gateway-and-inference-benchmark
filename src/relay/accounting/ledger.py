"""Usage ledger: one durable row per request, written without ever slowing a request.

Requests hand events to `submit()`, which only puts them on a bounded in-memory queue
(never waits). A background task writes them to Postgres in batches. If Postgres is
down, the batch is retried with backoff while the queue absorbs new events; if the
queue is full, new events are dropped and counted. A batch that fails because of bad
data is retried row by row so one bad row cannot block the ledger.
"""

import asyncio
import logging
from collections.abc import Sequence

import asyncpg

from relay.accounting.usage import UsageEvent

log = logging.getLogger("relay.ledger")

INSERT_SQL = """
INSERT INTO usage_events (
    request_id, occurred_at, tenant_id, api_key_id, model, backend, stream, status_code,
    prompt_tokens, completion_tokens, usage_source, ttft_ms, duration_ms, cost_gbp
) VALUES ($1, $2, $3, $4::uuid, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
ON CONFLICT (request_id) DO NOTHING
"""

# Errors caused by the data itself: retrying the same row will never succeed.
PERMANENT_ERRORS = (asyncpg.IntegrityConstraintViolationError, asyncpg.DataError)


def _row(e: UsageEvent) -> tuple:
    u = e.usage
    return (
        e.request_id, e.occurred_at, e.tenant_id, e.api_key_id, u.model, u.backend, u.stream,
        u.status_code, u.prompt_tokens, u.completion_tokens, u.source, e.ttft_ms,
        e.duration_ms, e.cost_gbp,
    )  # fmt: skip


class UsageLedger:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        max_queue: int = 10_000,
        batch_size: int = 500,
        flush_interval_s: float = 1.0,
        initial_backoff_s: float = 1.0,
        max_backoff_s: float = 30.0,
    ) -> None:
        self._pool = pool
        self._queue: asyncio.Queue[UsageEvent] = asyncio.Queue(maxsize=max_queue)
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._initial_backoff_s = initial_backoff_s
        self._max_backoff_s = max_backoff_s
        self._task: asyncio.Task | None = None
        # The batch the writer holds (e.g. while waiting to retry); written at shutdown.
        self._inflight: list[UsageEvent] = []
        # Counters (become metrics with observability).
        self.written = 0
        self.dropped = 0  # queue full
        self.rejected = 0  # rows the database refused as invalid
        self.write_failures = 0  # batches that failed and were retried

    # --- request side -----------------------------------------------------------------

    def submit(self, event: UsageEvent) -> None:
        """Never blocks: a full queue drops the event and counts it."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1
            if self.dropped == 1 or self.dropped % 1000 == 0:
                log.warning("usage ledger queue full: %d events dropped so far", self.dropped)

    @property
    def queued(self) -> int:
        return self._queue.qsize()

    # --- background writer ------------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="usage-ledger")

    async def close(self, timeout_s: float = 5.0) -> None:
        """Stop the writer, then try to write what is still queued (bounded in time)."""
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        try:
            async with asyncio.timeout(timeout_s):
                await self._write(self._inflight)
                self._inflight = []
                while not self._queue.empty():
                    await self._write(self._take_batch_nowait())
        except (TimeoutError, OSError, asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
            lost = self.queued + len(self._inflight)
            log.warning("usage ledger: %d events not written at shutdown (%r)", lost, exc)

    def _take_batch_nowait(self) -> list[UsageEvent]:
        batch = []
        while len(batch) < self._batch_size and not self._queue.empty():
            batch.append(self._queue.get_nowait())
        return batch

    async def _next_batch(self) -> list[UsageEvent]:
        """Wait for the first event, then collect more for up to flush_interval_s."""
        batch = [await self._queue.get()]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._flush_interval_s
        while len(batch) < self._batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), remaining))
            except TimeoutError:
                break
        return batch

    async def _run(self) -> None:
        while True:
            batch = self._inflight = await self._next_batch()
            backoff = self._initial_backoff_s
            while True:
                try:
                    await self._write(batch)
                    self._inflight = []
                    break
                except (OSError, asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
                    # Postgres unreachable or overloaded: keep the batch and retry.
                    self.write_failures += 1
                    log.warning(
                        "usage ledger: write of %d events failed (%r); retrying in %.0fs,"
                        " %d more queued",
                        len(batch), exc, backoff, self.queued,
                    )  # fmt: skip
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self._max_backoff_s)

    async def _write(self, batch: Sequence[UsageEvent]) -> None:
        if not batch:
            return
        rows = [_row(e) for e in batch]
        try:
            async with self._pool.acquire() as conn, conn.transaction():
                await conn.executemany(INSERT_SQL, rows)
        except PERMANENT_ERRORS:
            # Some row is invalid. Write row by row so only the bad ones are lost.
            await self._write_one_by_one(rows)
            return
        self.written += len(rows)

    async def _write_one_by_one(self, rows: list[tuple]) -> None:
        async with self._pool.acquire() as conn:
            for row in rows:
                try:
                    await conn.execute(INSERT_SQL, *row)
                    self.written += 1
                except PERMANENT_ERRORS as exc:
                    self.rejected += 1
                    log.error("usage ledger: rejected event %s (%r)", row[0], exc)
