"""Host memory guard (docs/findings.md F8).

On the 16 GB dev laptop, a full stack paged to disk under load and every stream stalled:
the run measured the laptop's memory, not the GPU. So a run checks free RAM before it
starts, samples it throughout, and records the result in env.json; a run whose host ran
short is flagged as invalid instead of being reported.
"""

import threading
from dataclasses import dataclass, field

import psutil

GiB = 2**30
# Refuse to start below this (unless overridden) ...
MIN_AVAILABLE_TO_START_GIB = 2.0
# ... and flag the run if available memory ever fell below this while it ran.
MIN_AVAILABLE_DURING_RUN_GIB = 1.0


def available_gib() -> float:
    return psutil.virtual_memory().available / GiB


@dataclass
class MemoryMonitor:
    """Samples available RAM every `interval_s` on a background thread."""

    interval_s: float = 1.0
    samples_gib: list[float] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __enter__(self) -> "MemoryMonitor":
        self.samples_gib.append(available_gib())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(self.interval_s * 2)
        self.samples_gib.append(available_gib())

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.samples_gib.append(available_gib())

    @property
    def low(self) -> bool:
        return min(self.samples_gib) < MIN_AVAILABLE_DURING_RUN_GIB

    def report(self) -> dict:
        return {
            "total_gib": round(psutil.virtual_memory().total / GiB, 2),
            "available_gib_start": round(self.samples_gib[0], 2),
            "available_gib_min": round(min(self.samples_gib), 2),
            "available_gib_end": round(self.samples_gib[-1], 2),
            "samples": len(self.samples_gib),
            "flagged_low": self.low,
            "threshold_gib": MIN_AVAILABLE_DURING_RUN_GIB,
        }
