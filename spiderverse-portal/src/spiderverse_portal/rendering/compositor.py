"""Portal compositing.

    final = original * (1 - alpha) + stylized * alpha

Both layers are in the same image coordinates: the stylized frame is a
pixel-aligned transform of the camera frame, so the portal reveals the scene
exactly where it is. The quad only shapes the mask; nothing is warped.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from ..geometry.masks import polygon_alpha_roi
from .occlusion import HandOccluder


class PortalAnimator:
    """Time-based fade toward 1 while the gesture is visible, toward 0 otherwise."""

    def __init__(self, fade_in_ms: float, fade_out_ms: float) -> None:
        self.fade_in_ms = fade_in_ms
        self.fade_out_ms = fade_out_ms
        self.alpha = 0.0
        self.quad: np.ndarray | None = None
        self._t: float | None = None

    def update(self, visible: bool, quad: np.ndarray | None, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        dt_ms = 0.0 if self._t is None else (now - self._t) * 1000.0
        self._t = now
        if visible and quad is not None:
            self.quad = quad
            self.alpha = min(1.0, self.alpha + dt_ms / max(self.fade_in_ms, 1.0))
        else:
            self.alpha = max(0.0, self.alpha - dt_ms / max(self.fade_out_ms, 1.0))
            if self.alpha <= 0.0:
                self.quad = None
        return self.alpha

    def reset(self) -> None:
        self.alpha = 0.0
        self.quad = None
        self._t = None


def blend_roi(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray, roi: tuple[int, int, int, int]) -> None:
    """dst[roi] = dst * (1 - alpha) + src * alpha, in place."""
    x0, y0, x1, y1 = roi
    a = np.ascontiguousarray(alpha, dtype=np.float32)
    dst[y0:y1, x0:x1] = cv2.blendLinear(src[y0:y1, x0:x1], dst[y0:y1, x0:x1], a, 1.0 - a)


def draw_panel_border(img: np.ndarray, quad: np.ndarray, width: int, offset: int, opacity: float = 1.0) -> None:
    """Comic panel border: misregistered magenta/cyan plates under a black key line."""
    if width <= 0 or opacity <= 0.01:
        return
    pts = np.round(np.asarray(quad) * 16).astype(np.int32).reshape(-1, 1, 2)
    layer = img.copy()
    if offset > 0:
        o = np.array([[offset * 16, offset * 16]], np.int32)
        cv2.polylines(layer, [pts + o], True, (235, 60, 255), max(1, width - 1), cv2.LINE_AA, shift=4)
        cv2.polylines(layer, [pts - o], True, (255, 230, 40), max(1, width - 1), cv2.LINE_AA, shift=4)
    cv2.polylines(layer, [pts], True, (20, 8, 16), width, cv2.LINE_AA, shift=4)
    if opacity >= 0.999:
        img[:] = layer
    else:
        cv2.addWeighted(layer, opacity, img, 1.0 - opacity, 0.0, dst=img)


class PortalCompositor:
    def __init__(self, occluder: HandOccluder, feather_px: float, border_px: int, border_offset_px: int) -> None:
        self.occluder = occluder
        self.feather_px = feather_px
        self.border_px = border_px
        self.border_offset_px = border_offset_px
        self.last_ms = 0.0

    def compose(self, original: np.ndarray, stylized: np.ndarray, quad: np.ndarray, alpha_scale: float,
                hands=None, strength: float = 1.0) -> np.ndarray:
        t0 = time.perf_counter()
        h, w = original.shape[:2]
        out = original.copy()
        res = polygon_alpha_roi(quad, (w, h), self.feather_px)
        if res is None or alpha_scale <= 0.0:
            self.last_ms = (time.perf_counter() - t0) * 1000
            return out
        portal, roi = res
        alpha = portal * (alpha_scale * strength)
        blend_roi(out, stylized, alpha, roi)
        draw_panel_border(out, quad, self.border_px, self.border_offset_px, alpha_scale)
        if hands:
            # Occlusion ROI: portal box grown to cover the border and fingers crossing it.
            pad = int(max(hd.palm_length for hd in hands) * 0.8) + self.border_px + self.border_offset_px
            ox0, oy0 = max(0, roi[0] - pad), max(0, roi[1] - pad)
            ox1, oy1 = min(w, roi[2] + pad), min(h, roi[3] + pad)
            big = np.zeros((oy1 - oy0, ox1 - ox0), np.float32)
            big[roi[1] - oy0: roi[3] - oy0, roi[0] - ox0: roi[2] - ox0] = portal
            occ = self.occluder.mask(original, hands, big, (ox0, oy0, ox1, oy1))
            if occ is not None:
                blend_roi(out, original, occ, (ox0, oy0, ox1, oy1))
        self.last_ms = (time.perf_counter() - t0) * 1000
        return out
