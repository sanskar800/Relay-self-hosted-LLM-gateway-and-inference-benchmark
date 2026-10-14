"""Seeded synthetic workload: prompt and output lengths from documented distributions.

Lengths are drawn from clipped log-normal distributions (see Scenario.workload). Prompts
are built from words; ~1.3 tokens per English word is the planning factor, and the real
token count is taken from the engine's usage report for every request.
"""

import math
import random
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


def build_prompt(target_tokens: int, rng: random.Random, unique_prefix: bool) -> str:
    n_words = max(1, round(target_tokens / TOKENS_PER_WORD))
    words = [rng.choice(WORDS) for _ in range(n_words)]
    if unique_prefix:
        # A random opening means no two prompts share a cacheable prefix beyond the
        # chat template, as with real users.
        opening = f"[note {rng.getrandbits(48):012x}]"
        return f"{opening} {INSTRUCTION}\n{' '.join(words)}"
    return f"{INSTRUCTION}\n{' '.join(words)}"


def generate(workload: Workload, n: int) -> list[RequestSpec]:
    """The first `n` requests of this workload; the same seed gives the same requests."""
    rng = random.Random(workload.seed)
    specs = []
    for i in range(n):
        prompt_tokens = sample_length(workload.prompt_tokens, rng)
        output_tokens = sample_length(workload.output_tokens, rng)
        specs.append(
            RequestSpec(
                index=i,
                prompt=build_prompt(prompt_tokens, rng, workload.unique_prefix),
                target_prompt_tokens=prompt_tokens,
                max_tokens=output_tokens,
            )
        )
    return specs


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
