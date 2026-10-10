"""Ink contours from a guide image.

Lines come from a difference of Gaussians on the luminance, taken on the dark
side of edges (where a pixel is darker than its surroundings), which is where
an illustrator's outline sits. Two scales are combined: a fine one for
features (eyes, brows, mouth, fingers) and a coarse one for silhouettes. Soft
thresholds instead of hard ones keep weak edges from popping on and off
between frames; a small opening removes isolated specks so the result reads as
deliberate strokes, not Canny noise.

Work happens at half resolution; lines are upsampled and re-sharpened with a
LUT, which gives them a natural anti-aliased weight.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .shading import smoothstep

_SHARPEN = (smoothstep(0.2, 0.7, np.linspace(0.0, 1.0, 256)) * 255.0 + 0.5).astype(np.uint8)


class InkRenderer:
    def __init__(self, threshold: float = 0.018, scale: float = 1.0, silhouette: float = 0.6) -> None:
        self.threshold = threshold
        self.scale = scale
        self.silhouette = silhouette
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    def lines_u8(self, guide_bgr: np.ndarray, protect: np.ndarray | None = None) -> np.ndarray:
        """Ink coverage as uint8 (0..255) at the guide's full resolution.

        Inside `protect` (eyes, brows, mouth) only the fine scale draws, so features
        keep their lid and lip lines instead of filling in as solid blobs.
        """
        h, w = guide_bgr.shape[:2]
        small = cv2.resize(guide_bgr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) * (1.0 / 255.0)
        gray = cv2.GaussianBlur(gray, (0, 0), 0.6)
        s = max(self.scale, 0.3)
        thr = self.threshold
        b1 = cv2.GaussianBlur(gray, (0, 0), 0.9 * s)
        b2 = cv2.GaussianBlur(gray, (0, 0), 1.6 * s)
        b3 = cv2.GaussianBlur(gray, (0, 0), 3.0 * s)
        fine = smoothstep(thr, thr * 2.4, b2 - b1)
        coarse = smoothstep(thr * 1.3, thr * 3.0, b3 - b2) * (self.silhouette * 1.25)
        if protect is not None:
            small_protect = cv2.resize(protect, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            coarse *= 1.0 - small_protect
        ink = np.clip(np.maximum(fine, coarse) * 255.0, 0, 255).astype(np.uint8)
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, self._kernel)
        up = cv2.resize(ink, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.LUT(up, _SHARPEN)

    def lines(self, guide_bgr: np.ndarray) -> np.ndarray:
        """Float32 ink coverage in [0, 1]."""
        return self.lines_u8(guide_bgr).astype(np.float32) * (1.0 / 255.0)


@lru_cache(maxsize=4)
def _solid(h: int, w: int, color: tuple[int, int, int]) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:] = color
    img.setflags(write=False)
    return img


def apply_ink(bgr: np.ndarray, ink_u8: np.ndarray, strength: float, ghost: float = 0.0,
              ink_color_bgr=(28, 8, 18), ghost_color_bgr=(200, 40, 235), ghost_offset=(2, 1)) -> np.ndarray:
    """Composite an offset ghost plate (multiply-tinted), then the aligned key plate."""
    h, w = ink_u8.shape
    out = bgr
    if ghost > 0:
        dx, dy = ghost_offset
        shifted = np.zeros_like(ink_u8)
        shifted[dy:, dx:] = ink_u8[: h - dy, : w - dx]
        g = shifted.astype(np.float32) * (ghost / 255.0)
        tinted = cv2.multiply(out, np.array(ghost_color_bgr + (0,), np.float64) / 255.0)
        out = cv2.blendLinear(out, tinted, 1.0 - g, g)
    a = ink_u8.astype(np.float32) * (strength / 255.0)
    return cv2.blendLinear(out, _solid(h, w, tuple(int(c) for c in ink_color_bgr)), 1.0 - a, a)
