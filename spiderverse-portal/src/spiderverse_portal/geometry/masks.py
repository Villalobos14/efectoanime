"""Polygon alpha masks in image coordinates.

Masks are rasterised with OpenCV's anti-aliased polygon fill at 1/16 pixel
precision, then optionally feathered. Work is limited to the polygon's
bounding box so cost scales with portal size, not frame size.
"""

from __future__ import annotations

import cv2
import numpy as np

_SHIFT = 4
_SCALE = 1 << _SHIFT


def bounding_roi(points: np.ndarray, frame_size: tuple[int, int], pad: int) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) clipped to the frame, or None if fully outside."""
    w, h = frame_size
    p = np.asarray(points, dtype=np.float64)
    x0 = max(0, int(np.floor(p[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(p[:, 1].min())) - pad)
    x1 = min(w, int(np.ceil(p[:, 0].max())) + pad + 1)
    y1 = min(h, int(np.ceil(p[:, 1].max())) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def polygon_alpha_roi(
    points: np.ndarray, frame_size: tuple[int, int], feather_px: float = 0.0
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Float32 alpha in [0, 1] for the polygon's ROI, plus that ROI."""
    pad = int(np.ceil(feather_px * 2)) + 2
    roi = bounding_roi(points, frame_size, pad)
    if roi is None:
        return None
    x0, y0, x1, y1 = roi
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    local = np.asarray(points, dtype=np.float64) - np.array([x0, y0], dtype=np.float64)
    pts = np.round(local * _SCALE).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_AA, shift=_SHIFT)
    alpha = mask.astype(np.float32) * (1.0 / 255.0)
    if feather_px > 0.25:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=float(feather_px))
    return alpha, roi


def polygon_alpha(points: np.ndarray, frame_size: tuple[int, int], feather_px: float = 0.0) -> np.ndarray:
    """Full-frame float32 alpha mask (convenience for tests and debug views)."""
    w, h = frame_size
    full = np.zeros((h, w), dtype=np.float32)
    result = polygon_alpha_roi(points, frame_size, feather_px)
    if result is not None:
        alpha, (x0, y0, x1, y1) = result
        full[y0:y1, x0:x1] = alpha
    return full
