"""Seeded synthetic workload: prompt and output lengths from documented distributions.

Lengths are drawn from clipped log-normal distributions (see Scenario.workload). Prompts
are built from words; ~1.3 tokens per English word is the planning factor, and the real
token count is taken from the engine's usage report for every request.
"""

import itertools
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass

from bench.loadgen.config import LengthDistribution, Workload

TOKENS_PER_WORD = 1.3  # planning factor only; actual counts come from `usage`

# A neutral vocabulary for filler text.
WORDS = [
    "river",
    "mountain",
    "city",
    "library",
    "garden",
    "market",
    "bridge",
    "harbor",
    "forest",
    "valley",
    "museum",
    "station",
    "window",
    "kitchen",
    "teacher",
    "engine",
    "signal",
    "planet",
    "winter",
    "summer",
    "island",
    "village",
    "meadow",
    "theater",
    "circuit",
    "compass",
    "lantern",
    "orchard",
    "pottery",
    "quarry",
    "ribbon",
    "saddle",
    "timber",
    "violin",
    "wagon",
    "yacht",
    "zephyr",
    "anchor",
    "beacon",
    "canyon",
    "desert",
    "ember",
    "falcon",
    "glacier",
]

INSTRUCTION = "Continue the following notes in the same style."


@dataclass(frozen=True)
class RequestSpec:
    index: int
    prompt: str
    target_prompt_tokens: int
    max_tokens: int


def sample_length(dist: LengthDistribution, rng: random.Random) -> int:
    value = rng.lognormvariate(math.log(dist.median), dist.sigma)
    return int(min(dist.max, max(dist.min, round(value))))


def build_prompt(
    target_tokens: int, rng: random.Random, unique_prefix: bool, openings: random.Random
) -> str:
    n_words = max(1, round(target_tokens / TOKENS_PER_WORD))
    words = [rng.choice(WORDS) for _ in range(n_words)]
    if unique_prefix:
        # A random opening means no two prompts share a cacheable prefix beyond the
        # chat template, as with real users. Drawn from its own RNG so the salt changes
        # only the openings, never the lengths or words.
        opening = f"[note {openings.getrandbits(48):012x}]"
        return f"{opening} {INSTRUCTION}\n{' '.join(words)}"
    return f"{INSTRUCTION}\n{' '.join(words)}"


def stream(workload: Workload, salt: str = "") -> Iterator[RequestSpec]:
    """This workload's requests, endlessly. The seed fixes lengths and words; the salt
    fixes the openings. Engines cache prompt prefixes ACROSS requests (llama.cpp keeps
    earlier prompts and reuses the longest common prefix), so a prompt sent again by
    another level or run would skip its prefill and flatter TTFT. Callers pass a salt
    unique to each run and level: same workload shape, never the same prompt twice."""
    rng = random.Random(workload.seed)
    openings = random.Random(f"{workload.seed}:{salt}")
    for i in itertools.count():
        prompt_tokens = sample_length(workload.prompt_tokens, rng)
        output_tokens = sample_length(workload.output_tokens, rng)
        yield RequestSpec(
            index=i,
            prompt=build_prompt(prompt_tokens, rng, workload.unique_prefix, openings),
            target_prompt_tokens=prompt_tokens,
            max_tokens=output_tokens,
        )


def generate(workload: Workload, n: int, salt: str = "") -> list[RequestSpec]:
    """The first `n` requests of this workload (open loop: the count is known up front)."""
    return list(itertools.islice(stream(workload, salt), n))


def request_body(spec: RequestSpec, model: str, workload: Workload) -> dict:
    """An OpenAI chat-completions body; streamed, with usage, fixed output length."""
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": spec.prompt}],
        "max_tokens": spec.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if workload.ignore_eos:
        body["ignore_eos"] = True  # llama.cpp and vLLM: generate exactly max_tokens
    return body
