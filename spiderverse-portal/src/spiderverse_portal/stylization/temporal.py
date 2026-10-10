"""Motion-aware temporal stabilisation of neural stylization.

Two cheap stages at inference resolution:

1. Input denoise: where the camera image barely changed, the network input is
   pulled toward the previous (already stabilised) input. Sensor noise is the
   main source of neural flicker; this removes it without touching motion.
2. Output blend: where the scene is static, the new neural output is averaged
   with the previous output; where anything moved, the new output is used as is.

Blinks, mouth movement and head turns change the input well above the noise
threshold, so they register as motion and pass through immediately. Nothing
is frozen; history only accumulates in static regions.
"""

from __future__ import annotations

import cv2
import numpy as np


class TemporalStabilizer:
    def __init__(self, input_denoise: float = 0.6, output_blend: float = 0.45, motion_threshold: float = 10.0):
        self.input_denoise = input_denoise
        self.output_blend = output_blend
        self.motion_threshold = motion_threshold
        self.noise_gain = 1.0  # set to the auto-exposure gain: amplified sensor noise is not motion
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.reset()

    def reset(self) -> None:
        self._prev_in: np.ndarray | None = None
        self._prev_out: np.ndarray | None = None
        self._motion: np.ndarray | None = None
        self.static_fraction = 0.0

    def stabilize_input(self, rgb: np.ndarray) -> np.ndarray:
        if self._prev_in is None or self._prev_in.shape != rgb.shape:
            self._prev_in = rgb.astype(np.float32)
            self._prev_out = None
            self._motion = np.ones(rgb.shape[:2], np.float32)
            return rgb
        cur = rgb.astype(np.float32)
        diff = cv2.absdiff(cur, self._prev_in).max(axis=2)
        diff = cv2.GaussianBlur(diff, (0, 0), 1.0)
        thr = float(self.motion_threshold * max(1.0, float(self.noise_gain)))
        t = np.clip((diff - 0.5 * thr) / thr, 0.0, 1.0)
        motion = t * t * (3.0 - 2.0 * t)
        motion = cv2.dilate(motion, self._kernel)
        self._motion = motion
        self.static_fraction = float((motion < 0.05).mean())
        w = (self.input_denoise * (1.0 - motion))[..., None]
        stabilized = cur * (1.0 - w) + self._prev_in * w
        self._prev_in = stabilized.astype(np.float32, copy=False)
        return np.clip(stabilized + 0.5, 0, 255).astype(np.uint8)

    def stabilize_output(self, out: np.ndarray) -> np.ndarray:
        if self._prev_out is None or self._prev_out.shape != out.shape or self._motion is None:
            self._prev_out = out.astype(np.float32)
            return out
        w = (self.output_blend * (1.0 - self._motion))[..., None]
        blended = out.astype(np.float32) * (1.0 - w) + self._prev_out * w
        self._prev_out = blended.astype(np.float32, copy=False)
        return np.clip(blended + 0.5, 0, 255).astype(np.uint8)
