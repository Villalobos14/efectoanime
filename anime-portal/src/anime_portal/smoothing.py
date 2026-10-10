"""Motion smoothing.

The One Euro Filter follows Casiez, Roussel, and Vogel, "1€ Filter: A Simple
Speed-based Low-pass Filter for Noisy Input in Interactive Systems" (CHI 2012).
It stays smooth when the hands are still and stays responsive when they move.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Scalar One Euro Filter using seconds as the time unit."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoff frequencies must be positive")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._hat_x: float | None = None
        self._hat_dx = 0.0
        self._prev_raw: float | None = None
        self._prev_t: float | None = None

    def reset(self, value: float | None = None, timestamp: float | None = None) -> None:
        self._hat_x = None if value is None else float(value)
        self._hat_dx = 0.0
        self._prev_raw = None if value is None else float(value)
        self._prev_t = timestamp

    def __call__(self, value: float, timestamp: float) -> float:
        value = float(value)
        timestamp = float(timestamp)
        if self._hat_x is None or self._prev_raw is None or self._prev_t is None:
            self._hat_x = value
            self._prev_raw = value
            self._hat_dx = 0.0
            self._prev_t = timestamp
            return value
        dt = timestamp - self._prev_t
        if dt <= 1e-6:
            return self._hat_x
        dx = (value - self._prev_raw) / dt
        a_d = _alpha(dt, self.d_cutoff)
        self._hat_dx = a_d * dx + (1.0 - a_d) * self._hat_dx
        cutoff = self.min_cutoff + self.beta * abs(self._hat_dx)
        a = _alpha(dt, cutoff)
        self._hat_x = a * value + (1.0 - a) * self._hat_x
        self._prev_raw = value
        self._prev_t = timestamp
        return self._hat_x


class MultiPointSmoother:
    """Independent One Euro filters for each coordinate of N 2D points.

    A sample that jumps farther than ``max_jump_px`` is ignored until it
    persists for ``jump_accept_frames`` consecutive updates, then the filters
    snap to that new pose. This rejects single-frame landmark swaps without
    freezing a hand that really moved.
    """

    def __init__(
        self,
        n_points: int,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
        max_jump_px: float = 160.0,
        jump_accept_frames: int = 3,
    ) -> None:
        if n_points < 1:
            raise ValueError("n_points must be positive")
        self.n_points = n_points
        self.max_jump_px = float(max_jump_px)
        self.jump_accept_frames = max(1, int(jump_accept_frames))
        self._x = [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(n_points)]
        self._y = [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(n_points)]
        self._last: np.ndarray | None = None
        self._pending = 0

    def reset(self, points: np.ndarray, timestamp: float) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(self.n_points, 2)
        for i in range(self.n_points):
            self._x[i].reset(pts[i, 0], timestamp)
            self._y[i].reset(pts[i, 1], timestamp)
        self._last = pts.copy()
        self._pending = 0
        return self._last.copy()

    def clear(self) -> None:
        for filt in (*self._x, *self._y):
            filt.reset()
        self._last = None
        self._pending = 0

    def smooth(self, points: np.ndarray, timestamp: float) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(self.n_points, 2)
        if self._last is None:
            return self.reset(pts, timestamp)
        jump = float(np.max(np.linalg.norm(pts - self._last, axis=1)))
        if jump > self.max_jump_px:
            self._pending += 1
            if self._pending < self.jump_accept_frames:
                return self._last.copy()
            return self.reset(pts, timestamp)
        self._pending = 0
        out = np.empty_like(pts)
        for i in range(self.n_points):
            out[i, 0] = self._x[i](pts[i, 0], timestamp)
            out[i, 1] = self._y[i](pts[i, 1], timestamp)
        self._last = out.copy()
        return out


class TemporalFade:
    """Exponential fade toward 1 when active and toward 0 when released."""

    def __init__(self, attack: float = 16.0, release: float = 8.0) -> None:
        self.attack = float(attack)
        self.release = float(release)
        self.alpha = 0.0

    def reset(self) -> None:
        self.alpha = 0.0

    def update(self, active: bool, dt: float) -> float:
        dt = max(0.0, min(float(dt), 0.1))
        target = 1.0 if active else 0.0
        speed = self.attack if target > self.alpha else self.release
        if speed <= 0.0 or dt == 0.0:
            return self.alpha
        k = 1.0 - math.exp(-speed * dt)
        self.alpha += (target - self.alpha) * k
        if abs(self.alpha - target) < 1e-3:
            self.alpha = target
        return self.alpha


class SampleHold:
    """Keep the last good sample through brief detection dropouts."""

    def __init__(self, hold_sec: float) -> None:
        self.hold_sec = float(hold_sec)
        self._value: np.ndarray | None = None
        self._time: float | None = None

    def clear(self) -> None:
        self._value = None
        self._time = None

    def update(self, value: np.ndarray | None, timestamp: float) -> np.ndarray | None:
        if value is not None:
            self._value = np.array(value, dtype=np.float64, copy=True)
            self._time = float(timestamp)
            return self._value.copy()
        if self._value is not None and self._time is not None and (timestamp - self._time) <= self.hold_sec:
            return self._value.copy()
        return None
