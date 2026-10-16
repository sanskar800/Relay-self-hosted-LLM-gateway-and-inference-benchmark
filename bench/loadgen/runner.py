"""Open-loop load generator: requests arrive as a Poisson process, whatever the server does.

Each request is started as its own task at its scheduled time and never waits for earlier
requests, so a slow server builds queues instead of slowing the arrivals (a closed loop
would hide that). Every request records when it was scheduled, sent, each content event,
and when it ended; the gap between scheduled and sent (scheduling lag) shows whether the
generator itself kept up.
"""

import asyncio
import json
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import httpx

from bench.loadgen.config import Scenario
from bench.loadgen.workload import RequestSpec, request_body


@dataclass
class RequestResult:
    level: float  # arrival rate (open loop) or concurrency (closed loop)
    index: int
    scheduled_s: float  # relative to the start of this level's run
    sent_s: float | None = None
    first_token_s: float | None = None
    end_s: float | None = None
    token_times_s: list[float] = field(default_factory=list)  # each content event
    status: int = 0  # HTTP status; 0 = no response (connection/timeout)
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    target_prompt_tokens: int = 0
    max_tokens: int = 0
    in_warmup: bool = False
    backend: str | None = None  # x-relay-backend, when going through Relay
    request_id: str | None = None
    engine_fingerprint: str | None = None  # system_fingerprint (engine build)

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None and self.first_token_s is not None


def arrival_schedule(scenario: Scenario, rate_rps: float) -> list[float]:
    """This level's arrival times. Seeded per level, so levels are independent and every
    run of the same scenario sends at the same moments."""
    seed = scenario.workload.seed + int(rate_rps * 1000)
    return poisson_schedule(rate_rps, scenario.load.duration_s, seed)


def poisson_schedule(rate_rps: float, duration_s: float, seed: int) -> list[float]:
    """Arrival times in [0, duration): exponential gaps with mean 1/rate."""
    rng = random.Random(seed)
    times, t = [], 0.0
    while True:
        t += rng.expovariate(rate_rps)
        if t >= duration_s:
            return times
        times.append(t)


def make_client(scenario: Scenario, transport: httpx.AsyncBaseTransport | None = None):
    # No connection cap: a capped pool would queue requests in the client, quietly
    # turning the open loop back into a closed one.
    return httpx.AsyncClient(
        base_url=str(scenario.target.base_url),
        timeout=httpx.Timeout(scenario.load.request_timeout_s, connect=10),
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        transport=transport,
    )


async def send_one(
    client: httpx.AsyncClient,
    spec: RequestSpec,
    result: RequestResult,
    scenario: Scenario,
    headers: dict[str, str],
    clock: Callable[[], float],
    start: float,
) -> None:
    body = request_body(spec, scenario.target.model, scenario.workload)
    result.sent_s = clock() - start
    try:
        async with client.stream("POST", "chat/completions", json=body, headers=headers) as resp:
            result.status = resp.status_code
            result.backend = resp.headers.get("x-relay-backend")
            result.request_id = resp.headers.get("x-request-id")
            if resp.status_code != 200:
                result.error = (await resp.aread()).decode(errors="replace")[:300]
                return
            async for line in resp.aiter_lines():
                now = clock() - start
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                result.engine_fingerprint = (
                    chunk.get("system_fingerprint") or result.engine_fingerprint
                )
                if "error" in chunk:
                    result.error = json.dumps(chunk["error"])[:300]
                    continue
                if any((c.get("delta") or {}).get("content") for c in chunk.get("choices") or []):
                    if result.first_token_s is None:
                        result.first_token_s = now
                    result.token_times_s.append(now)
                if usage := chunk.get("usage"):
                    result.prompt_tokens = usage.get("prompt_tokens")
                    result.completion_tokens = usage.get("completion_tokens")
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        result.end_s = clock() - start


async def run_open_loop(
    scenario: Scenario,
    rate_rps: float,
    specs: list[RequestSpec],
    *,
    api_key: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> list[RequestResult]:
    load = scenario.load
    schedule = arrival_schedule(scenario, rate_rps)
    if len(schedule) > len(specs):
        raise ValueError(f"need {len(schedule)} request specs, got {len(specs)}")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    results: list[RequestResult] = []
    tasks: list[asyncio.Task] = []
    async with make_client(scenario, transport) as client:
        start = clock()
        for t, spec in zip(schedule, specs, strict=False):
            delay = start + t - clock()
            if delay > 0:
                await asyncio.sleep(delay)
            result = RequestResult(
                level=rate_rps,
                index=spec.index,
                scheduled_s=t,
                target_prompt_tokens=spec.target_prompt_tokens,
                max_tokens=spec.max_tokens,
                in_warmup=t < load.warmup_s,
            )
            results.append(result)
            # Fire and move on: the next arrival does not wait for this one.
            tasks.append(
                asyncio.create_task(send_one(client, spec, result, scenario, headers, clock, start))
            )
        await asyncio.gather(*tasks)
    return results


async def run_closed_loop(
    scenario: Scenario,
    concurrency: int,
    specs: Iterator[RequestSpec],
    *,
    api_key: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> list[RequestResult]:
    """N workers, each sending its next request only when the previous one finished.

    The offered load is whatever the server allows: when it slows down, the clients slow
    down with it, so queueing never shows up as latency. That is the effect the report
    contrasts with the open loop; real users do not wait for each other.
    """
    load = scenario.load
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    results: list[RequestResult] = []

    async def worker(client: httpx.AsyncClient, start: float) -> None:
        # No await between the check and next(): workers take specs one at a time.
        while (now := clock() - start) < load.duration_s:
            spec = next(specs)
            result = RequestResult(
                level=concurrency,
                index=spec.index,
                scheduled_s=now,  # ready when the previous request ended: no lag by design
                target_prompt_tokens=spec.target_prompt_tokens,
                max_tokens=spec.max_tokens,
                in_warmup=now < load.warmup_s,
            )
            results.append(result)
            await send_one(client, spec, result, scenario, headers, clock, start)

    async with make_client(scenario, transport) as client:
        start = clock()
        await asyncio.gather(*(worker(client, start) for _ in range(concurrency)))
    return results
