"""Cel shading and split-tone tinting as one fused per-pixel LUT pass.

The value channel V = max(R, G, B) is quantised into a few bands with
anti-aliased steps (so band edges do not crawl or shimmer). Each channel is
scaled by V'/V, which keeps hue and saturation while flattening tone. In the
same pass, shadows get a multiply tint (violet/blue) and highlights a screen
tint (warm pink), both as functions of V'. All of it collapses into
`out_c = in_c * mul_c[V] + add_c[V]`.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import ComicConfig


def smoothstep(e0: float | np.ndarray, e1: float | np.ndarray, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / np.maximum(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def band_curve(levels: int, softness: float) -> np.ndarray:
    """V (0..1, 256 samples) -> quantised V' with smooth steps."""
    v = np.linspace(0.0, 1.0, 256)
    n = int(levels)
    thresholds = np.array([(i / n) ** 0.9 for i in range(1, n)])
    outputs = 0.15 + (0.97 - 0.15) * (np.arange(n) / max(n - 1, 1)) ** 0.85
    q = np.full_like(v, outputs[0])
    for i, t in enumerate(thresholds):
        q += (outputs[i + 1] - outputs[i]) * smoothstep(t - softness, t + softness, v)
    return q


class CelShader:
    def __init__(self, cfg: ComicConfig) -> None:
        self.update(cfg)

    def update(self, cfg: ComicConfig) -> None:
        v = np.linspace(0.0, 1.0, 256)
        if cfg.cel_shading:
            q = band_curve(cfg.color_levels, cfg.cel_softness)
            vq = v + (q - v) * cfg.cel_strength
        else:
            vq = v.copy()
        scale = vq / np.maximum(v, 1.0 / 255.0)
        shadow_w = cfg.tint_strength * (1.0 - vq) ** 2
        high_w = cfg.tint_strength * 0.45 * smoothstep(0.72, 1.0, vq)
        shadow = np.array(cfg.shadow_tint[::-1], dtype=np.float64)  # RGB config -> BGR
        high = np.array(cfg.highlight_tint[::-1], dtype=np.float64)
        mul = scale[:, None] * (1.0 - shadow_w[:, None] + shadow_w[:, None] * shadow[None, :])
        mul = mul * (1.0 - high_w[:, None] * high[None, :])
        add = 255.0 * high_w[:, None] * high[None, :]
        self._mul = mul.astype(np.float32).reshape(256, 1, 3)
        self._add = add.astype(np.float32).reshape(256, 1, 3)
        self._value = (np.clip(vq, 0, 1) * 255.0 + 0.5).astype(np.uint8)

    def factors(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-pixel (mul, add) float32 3-channel maps and the shaded value (uint8)."""
        b, g, r = cv2.split(bgr)
        v = cv2.max(cv2.max(b, g), r)
        v3 = cv2.merge((v, v, v))
        return cv2.LUT(v3, self._mul), cv2.LUT(v3, self._add), cv2.LUT(v, self._value)

    def apply(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (shaded BGR uint8, shaded value uint8)."""
        mul, add, value = self.factors(bgr)
        out = cv2.add(cv2.multiply(bgr.astype(np.float32), mul), add)
        return np.clip(out, 0, 255).astype(np.uint8), value
