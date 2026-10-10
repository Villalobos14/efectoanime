"""Temporal smoothing for tracked geometry (not for images).

The One Euro filter (Casiez et al., CHI 2012) adapts its cutoff to speed:
slow, deliberate movement is smoothed heavily, fast movement passes with little
lag. Coordinates are expected in normalised units (fraction of the frame
diagonal) so `beta` does not depend on camera resolution.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Vector One Euro filter. Confidence < 1 lowers the cutoff (more smoothing)."""

    def __init__(self, min_cutoff: float = 1.4, beta: float = 6.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = None
        self._t = None

    @property
    def value(self) -> np.ndarray | None:
        return None if self._x is None else self._x.copy()

    def __call__(self, x: np.ndarray, t: float, confidence: float = 1.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        if dt <= 1e-6:
            return self._x.copy()
        self._t = t
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        speed = float(np.linalg.norm(self._dx))
        cutoff = self.min_cutoff + self.beta * speed
        cutoff *= 0.35 + 0.65 * float(np.clip(confidence, 0.0, 1.0))
        a = _alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x.copy()


class ExponentialMovingAverage:
    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._x: np.ndarray | None = None

    def reset(self) -> None:
        self._x = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        self._x = x.copy() if self._x is None else self.alpha * x + (1.0 - self.alpha) * self._x
        return self._x.copy()


class KeyedOneEuro:
    """One filter per identity key (e.g. (hand_track_id, 'index_tip'))."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.params = (min_cutoff, beta, d_cutoff)
        self._filters: dict[object, OneEuroFilter] = {}

    def __call__(self, key: object, x: np.ndarray, t: float, confidence: float = 1.0) -> np.ndarray:
        f = self._filters.get(key)
        if f is None:
            f = self._filters[key] = OneEuroFilter(*self.params)
        return f(x, t, confidence)

    def retain(self, keys: set) -> None:
        for key in list(self._filters):
            if key not in keys:
                del self._filters[key]

    def reset(self) -> None:
        self._filters.clear()
