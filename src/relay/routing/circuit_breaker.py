"""Circuit breaker per backend: stop sending requests to a backend that keeps failing.

    CLOSED --N consecutive failures--> OPEN --cool-down--> HALF_OPEN --probe ok--> CLOSED
                                         ^                     |
                                         +----probe fails------+

OPEN skips the backend without trying it, so requests stop paying for its failures.
HALF_OPEN lets exactly one probe request through to test recovery.
"""

import enum
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger("relay.routing")


class State(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class BreakerPolicy:
    failure_threshold: int = 3  # consecutive failures that open the breaker
    open_duration_s: float = 10.0  # cool-down before a probe is allowed


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        policy: BreakerPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.policy = policy or BreakerPolicy()
        self._clock = clock
        self.state = State.CLOSED
        self.consecutive_failures = 0
        self._opened_at = 0.0
        self._probe_started_at: float | None = None

    def allow_request(self) -> bool:
        """May a request be sent now? In HALF_OPEN, only one probe at a time."""
        now = self._clock()
        if self.state is State.CLOSED:
            return True
        if self.state is State.OPEN:
            if now - self._opened_at < self.policy.open_duration_s:
                return False
            self._transition(State.HALF_OPEN)
        # HALF_OPEN: one probe; a probe that never reported back is replaced after a while.
        if self._probe_started_at is None or (
            now - self._probe_started_at >= self.policy.open_duration_s
        ):
            self._probe_started_at = now
            return True
        return False

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self._probe_started_at = None
        if self.state is not State.CLOSED:
            self._transition(State.CLOSED)

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self._probe_started_at = None
        if self.state is State.HALF_OPEN or (
            self.state is State.CLOSED
            and self.consecutive_failures >= self.policy.failure_threshold
        ):
            self._opened_at = self._clock()
            self._transition(State.OPEN)

    def _transition(self, new: State) -> None:
        old, self.state = self.state, new
        level = logging.WARNING if new is State.OPEN else logging.INFO
        log.log(level, "circuit breaker %s: %s -> %s (consecutive failures: %d)",
                self.name, old.value, new.value, self.consecutive_failures)  # fmt: skip
