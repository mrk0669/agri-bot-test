"""Loop timing helpers.

The runtime is a set of fixed-rate loops (perception, control, telemetry,
heartbeat) sharing one thread. These primitives let each one run at its own
rate off a single monotonic clock, and - importantly for tests - off an
injectable clock so that timing behaviour is verifiable without sleeping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional
from collections import deque

__all__ = ["RateLimiter", "LoopTimer", "Stopwatch", "monotonic"]

monotonic = time.monotonic


class RateLimiter:
    """Fires at most ``hz`` times per second.

    Used for *sub-rate* work inside a faster loop (e.g. push the OLED at 4 Hz
    from a 30 Hz main loop) - it never sleeps, it just answers "is it due?".
    """

    def __init__(self, hz: float, clock: Callable[[], float] = monotonic,
                 fire_immediately: bool = True):
        if hz <= 0:
            raise ValueError(f"rate must be positive, got {hz}")
        self.period = 1.0 / float(hz)
        self._clock = clock
        self._next = -float("inf") if fire_immediately else clock() + self.period

    def due(self) -> bool:
        """True if the interval has elapsed; advances the schedule when it does."""
        now = self._clock()
        if now < self._next:
            return False
        # Re-base rather than accumulate, so a long stall does not produce a
        # burst of catch-up firings.
        self._next = now + self.period
        return True

    def reset(self) -> None:
        self._next = -float("inf")


class LoopTimer:
    """Paces a loop to a target rate and records how well it kept up.

    ``sleep()`` blocks for the remainder of the period. The rolling statistics
    are what the preflight check and the telemetry log use to prove the control
    loop is actually meeting its deadline on the Jetson under inference load.
    """

    def __init__(self, hz: float, clock: Callable[[], float] = monotonic,
                 sleeper: Callable[[float], None] = time.sleep,
                 window: int = 200):
        if hz <= 0:
            raise ValueError(f"rate must be positive, got {hz}")
        self.hz = float(hz)
        self.period = 1.0 / self.hz
        self._clock = clock
        self._sleep = sleeper
        self._last: Optional[float] = None
        self._durations: Deque[float] = deque(maxlen=window)
        self.overruns = 0
        self.iterations = 0

    def sleep(self) -> float:
        """Sleep the remainder of this period. Returns the achieved period (s)."""
        now = self._clock()
        if self._last is None:
            self._last = now
            self.iterations += 1
            return self.period

        elapsed = now - self._last
        remaining = self.period - elapsed
        if remaining > 0:
            self._sleep(remaining)
            achieved = self.period
        else:
            self.overruns += 1
            achieved = elapsed

        self._last = self._clock()
        self._durations.append(achieved)
        self.iterations += 1
        return achieved

    @property
    def mean_period(self) -> float:
        return sum(self._durations) / len(self._durations) if self._durations else 0.0

    @property
    def max_period(self) -> float:
        return max(self._durations) if self._durations else 0.0

    @property
    def mean_hz(self) -> float:
        mp = self.mean_period
        return 1.0 / mp if mp > 0 else 0.0

    @property
    def overrun_ratio(self) -> float:
        return self.overruns / self.iterations if self.iterations else 0.0

    def stats(self) -> dict:
        return {
            "target_hz": round(self.hz, 2),
            "mean_hz": round(self.mean_hz, 2),
            "max_period_ms": round(self.max_period * 1e3, 2),
            "overruns": self.overruns,
            "iterations": self.iterations,
            "overrun_ratio": round(self.overrun_ratio, 4),
        }


@dataclass
class Stopwatch:
    """Context-manager timer for profiling a block.

    >>> with Stopwatch() as sw:      # doctest: +SKIP
    ...     detector.infer(frame)
    >>> sw.ms                        # doctest: +SKIP
    12.4
    """

    clock: Callable[[], float] = monotonic
    start: float = field(default=0.0, init=False)
    elapsed: float = field(default=0.0, init=False)

    def __enter__(self) -> "Stopwatch":
        self.start = self.clock()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = self.clock() - self.start

    @property
    def ms(self) -> float:
        return self.elapsed * 1e3
