"""Auto-exposure ahead of stylization.

Webcams in dim rooms deliver frames whose luminance sits in the bottom tenth of
the range. AnimeGANv2 and the cel bands then produce near-black output. This
stage stretches levels between low/high luminance percentiles and lifts
midtones with a gamma, using parameters smoothed over time so brightness never
pumps or flickers between frames. It is a single 256-entry LUT per frame.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class ExposureNormalizer:
    def __init__(self, enabled: bool = True, max_gain: float = 4.0, target_mid: float = 0.40,
                 smoothing: float = 0.12) -> None:
        self.enabled = enabled
        self.max_gain = max_gain
        self.target_mid = target_mid
        self.smoothing = smoothing
        self.reset()

    def reset(self) -> None:
        self._state: np.ndarray | None = None  # (low, high, gamma)
        self.gain = 1.0
        self.gamma = 1.0

    def _measure(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (80, 60), interpolation=cv2.INTER_AREA)
        lum = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        lo, mid, hi = np.percentile(lum, (1.0, 50.0, 99.5))
        hi = max(hi, lo + 1.0 / self.max_gain)
        stretched_mid = float(np.clip((mid - lo) / (hi - lo), 1e-3, 0.999))
        gamma = math.log(self.target_mid) / math.log(stretched_mid)
        # Gamma below ~0.7 washed faces out on well-lit cameras; dim rooms rely on the gain instead.
        return np.array([lo, hi, float(np.clip(gamma, 0.7, 1.0))])

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame
        target = self._measure(frame)
        if self._state is None:
            self._state = target
        else:
            self._state += self.smoothing * (target - self._state)
        lo, hi, gamma = self._state
        self.gain, self.gamma = float(1.0 / (hi - lo)), float(gamma)
        x = np.linspace(0.0, 1.0, 256)
        y = np.clip((x - lo) / (hi - lo), 0.0, 1.0) ** gamma
        lut = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return cv2.LUT(frame, lut)
