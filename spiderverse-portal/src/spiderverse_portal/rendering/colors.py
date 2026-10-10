"""Spider-Verse-inspired colour grading.

Everything is a lookup table on uint8 channels, so grading costs a few
milliseconds per frame and is perfectly deterministic (no temporal flicker).

  * contrast: an S-curve with a crushed black point for rich blacks,
  * hue snapping: hues are pulled part of the way toward a comic palette
    (red, warm skin, gold, teal, cyan, electric blue, violet, magenta); the warm
    skin anchor keeps faces warm instead of drifting to pink or orange,
  * saturation: boosted, with a gentle roll-off so it does not clip.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import ComicConfig

# Palette anchors in degrees.
PALETTE_HUES_DEG = (355.0, 22.0, 46.0, 165.0, 190.0, 218.0, 268.0, 318.0)


def contrast_lut(contrast: float, black_point: float = 0.05, white_point: float = 0.98) -> np.ndarray:
    x = np.linspace(0.0, 1.0, 256)
    x = np.clip((x - black_point) / (white_point - black_point), 0.0, 1.0)
    k = 4.0 * max(contrast, 0.05)
    s = 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
    lo, hi = 1.0 / (1.0 + np.exp(k * 0.5)), 1.0 / (1.0 + np.exp(-k * 0.5))
    s = (s - lo) / (hi - lo)
    # Blend so contrast=1 is a mild S-curve, not a hard one.
    y = x + (s - x) * min(1.0, max(contrast - 0.6, 0.0) / 0.8)
    return np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)


def hue_snap_lut(strength: float, anchors_deg=PALETTE_HUES_DEG) -> np.ndarray:
    """OpenCV hue (0..179, degrees/2) -> snapped hue, as a 256-entry table (cv2.LUT needs 256)."""
    hues = np.arange(180, dtype=np.float64) * 2.0
    anchors = np.asarray(anchors_deg, dtype=np.float64)
    diff = (anchors[None, :] - hues[:, None] + 180.0) % 360.0 - 180.0
    nearest = diff[np.arange(180), np.argmin(np.abs(diff), axis=1)]
    out = (hues + strength * nearest) % 360.0
    table = np.zeros(256, np.uint8)
    table[:180] = (np.round(out / 2.0).astype(np.int32) % 180).astype(np.uint8)
    return table


def saturation_lut(saturation: float) -> np.ndarray:
    s = np.linspace(0.0, 1.0, 256)
    boosted = 1.0 - np.power(1.0 - s, max(saturation, 0.01))  # soft boost, never clips abruptly
    return np.clip(boosted * 255.0 + 0.5, 0, 255).astype(np.uint8)


class ColorGrader:
    def __init__(self, cfg: ComicConfig) -> None:
        self.update(cfg)

    def update(self, cfg: ComicConfig) -> None:
        self._contrast = contrast_lut(cfg.contrast)
        self._hue = hue_snap_lut(cfg.hue_snap)
        self._sat = saturation_lut(cfg.saturation)

    def apply(self, bgr: np.ndarray) -> np.ndarray:
        img = cv2.LUT(bgr, self._contrast)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h = cv2.LUT(h, self._hue)
        s = cv2.LUT(s, self._sat)
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def misregister(bgr: np.ndarray, offset: int) -> np.ndarray:
    """Offset the cyan and magenta plates by a pixel or two, like print registration drift.

    Only colour plates move; the black ink plate is applied afterwards, aligned.
    """
    if offset <= 0:
        return bgr
    out = bgr.copy()
    o = int(offset)
    # Red channel (cyan plate) shifts right/down, blue channel (yellow-ish plate) shifts left.
    out[o:, o:, 2] = bgr[:-o, :-o, 2]
    out[:, :-o, 0] = bgr[:, o:, 0]
    return out
