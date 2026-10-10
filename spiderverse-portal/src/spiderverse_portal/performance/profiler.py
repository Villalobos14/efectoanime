"""Measured rates and latencies. Every number shown in the HUD comes from here."""

from __future__ import annotations

import threading
import time
from collections import deque


class RateMeter:
    """Events per second over a sliding time window."""

    def __init__(self, window_s: float = 1.5) -> None:
        self.window_s = window_s
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def tick(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._times.append(now)
            self._trim(now)

    def rate(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._trim(now)
            if len(self._times) < 2:
                return 0.0
            span = self._times[-1] - self._times[0]
            return (len(self._times) - 1) / span if span > 0 else 0.0

    def _trim(self, now: float) -> None:
        while self._times and now - self._times[0] > self.window_s:
            self._times.popleft()


class LatencyStat:
    """Recent samples in milliseconds with mean, median and p90."""

    def __init__(self, size: int = 60) -> None:
        self._samples: deque[float] = deque(maxlen=size)
        self._lock = threading.Lock()

    def add(self, ms: float) -> None:
        with self._lock:
            self._samples.append(float(ms))

    def _sorted(self) -> list[float]:
        with self._lock:
            return sorted(self._samples)

    @property
    def count(self) -> int:
        return len(self._samples)

    def median(self) -> float:
        s = self._sorted()
        return s[len(s) // 2] if s else 0.0

    def p90(self) -> float:
        s = self._sorted()
        return s[min(len(s) - 1, int(len(s) * 0.9))] if s else 0.0

    def mean(self) -> float:
        s = self._sorted()
        return sum(s) / len(s) if s else 0.0


class SectionTimer:
    """Accumulate named section timings for one unit of work."""

    def __init__(self) -> None:
        self.sections: dict[str, float] = {}
        self._start = time.perf_counter()
        self._last = self._start

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.sections[name] = self.sections.get(name, 0.0) + (now - self._last) * 1000.0
        self._last = now

    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0
