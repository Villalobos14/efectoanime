"""Print textures: amplitude-modulated halftone, Ben-Day dots, crosshatching.

Patterns live in screen space and are computed once per frame size. Only the
dot radius (driven by a smoothed tone map) changes from frame to frame, so the
texture never crawls or shimmers as the subject moves.

`PrintTexture.coverage` returns ink coverage maps instead of an image so the
comic renderer can fold them, the cel shading and the paper grain into a single
per-channel multiply.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from .shading import smoothstep


@lru_cache(maxsize=8)
def dot_distance_field(h: int, w: int, cell: float, angle_deg: float) -> np.ndarray:
    """Distance to the nearest dot centre of a rotated grid, in pixels."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(angle_deg)
    u = (x * math.cos(a) + y * math.sin(a)) / cell
    v = (-x * math.sin(a) + y * math.cos(a)) / cell
    fu = u - np.round(u)
    fv = v - np.round(v)
    field = (np.sqrt(fu * fu + fv * fv) * cell).astype(np.float32)
    field.setflags(write=False)
    return field


@lru_cache(maxsize=8)
def hatch_fields(h: int, w: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Anti-aliased diagonal line coverage (0..1) in two directions."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    d1 = np.abs(((x + y) / spacing) % 1.0 - 0.5) * spacing
    d2 = np.abs(((x - y) / spacing) % 1.0 - 0.5) * spacing
    l1 = np.clip(1.2 - d1, 0.0, 1.0).astype(np.float32)
    l2 = np.clip(1.2 - d2, 0.0, 1.0).astype(np.float32)
    l1.setflags(write=False)
    l2.setflags(write=False)
    return l1, l2


@lru_cache(maxsize=8)
def benday_dots(h: int, w: int, cell: float, angle_deg: float, coverage: float) -> np.ndarray:
    field = dot_distance_field(h, w, cell, angle_deg)
    radius = math.sqrt(coverage / math.pi) * cell
    dots = np.clip(radius - field + 0.5, 0.0, 1.0).astype(np.float32)
    dots.setflags(write=False)
    return dots


def _lut(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values.astype(np.float32).reshape(256, 1))


def dot_coverage(field_px: np.ndarray, coverage: np.ndarray, cell: float) -> np.ndarray:
    """Anti-aliased dot mask for per-pixel area coverage in [0, 1] (reference implementation)."""
    radius = np.sqrt(np.clip(coverage, 0.0, 1.0) / math.pi) * cell
    return np.clip(radius - field_px + 0.5, 0.0, 1.0)


class PrintTexture:
    """Computes halftone (shadows), Ben-Day (midtones) and hatching (deep shadows) coverage."""

    def __init__(self, halftone_strength: float, cell: float, angle_deg: float, benday_strength: float,
                 crosshatch_strength: float) -> None:
        self.halftone_strength = halftone_strength
        self.cell = max(cell, 2.0)
        self.angle = angle_deg
        self.benday_strength = benday_strength
        self.crosshatch_strength = crosshatch_strength
        dark = 1.0 - np.linspace(0.0, 1.0, 256)  # tone index -> darkness
        coverage = smoothstep(0.35, 0.95, dark) * 0.85
        self._radius = _lut(np.sqrt(coverage / math.pi) * self.cell + 0.5)
        self._ht_weight = _lut(halftone_strength * smoothstep(0.32, 0.68, dark))
        mid = smoothstep(0.08, 0.30, dark) * (1.0 - smoothstep(0.45, 0.65, dark))
        self._bd_weight = _lut(benday_strength * mid)
        self._hatch1 = _lut(crosshatch_strength * smoothstep(0.62, 0.85, dark))
        self._hatch2 = _lut(crosshatch_strength * smoothstep(0.78, 0.95, dark))

    def coverage(self, tone_u8: np.ndarray, keep: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Returns (tinted halftone coverage, neutral darkening), float32, or None when off.

        `keep` is 1 where textures are allowed (0 over protected eyes and mouth).
        """
        h, w = tone_u8.shape
        tinted = neutral = None
        if self.halftone_strength > 0:
            field = dot_distance_field(h, w, self.cell, self.angle)
            dots = cv2.subtract(cv2.LUT(tone_u8, self._radius), field)
            np.clip(dots, 0.0, 1.0, out=dots)
            tinted = cv2.multiply(dots, cv2.LUT(tone_u8, self._ht_weight))
        parts = []
        if self.benday_strength > 0:
            dots_b = benday_dots(h, w, self.cell * 0.7, self.angle - 30.0, 0.16)
            parts.append(cv2.multiply(dots_b, cv2.LUT(tone_u8, self._bd_weight)))
        if self.crosshatch_strength > 0:
            l1, l2 = hatch_fields(h, w, max(self.cell * 0.9, 4.0))
            parts.append(cv2.max(cv2.multiply(l1, cv2.LUT(tone_u8, self._hatch1)),
                                 cv2.multiply(l2, cv2.LUT(tone_u8, self._hatch2))))
        if parts:
            neutral = parts[0] if len(parts) == 1 else cv2.add(parts[0], parts[1])
            np.clip(neutral, 0.0, 0.9, out=neutral)
        if keep is not None:
            if tinted is not None:
                tinted = cv2.multiply(tinted, keep)
            if neutral is not None:
                neutral = cv2.multiply(neutral, keep)
        return tinted, neutral


def smooth_tone(value_u8: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Low-pass uint8 tone map so dot sizes vary smoothly (blurred at half resolution)."""
    h, w = value_u8.shape
    small = cv2.resize(value_u8, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
