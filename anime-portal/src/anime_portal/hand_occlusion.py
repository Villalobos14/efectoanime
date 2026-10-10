"""Approximate hand occlusion from landmarks.

Fingers and the palm are painted as thick strokes. This is a geometric stand-in
for a segmentation mask and can be replaced later without changing the renderer.
"""

from __future__ import annotations

import cv2
import numpy as np

from anime_portal.config import OcclusionSettings

FINGER_CHAINS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
PALM_INDICES: tuple[int, ...] = (0, 5, 9, 13, 17)
TIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)


class HandOccluder:
    def __init__(self, settings: OcclusionSettings | None = None) -> None:
        self.settings = settings or OcclusionSettings()
        self._kernel: np.ndarray | None = None
        self._kernel_px = -1

    def mask(
        self,
        shape_hw: tuple[int, int],
        hands: list[np.ndarray],
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Return a float32 mask in ``[0, 1]`` with the same height and width.

        ``origin`` is the top-left of this mask in full-frame pixel coordinates.
        Landmark points are shifted into the mask before drawing.
        """
        height, width = shape_hw
        canvas = np.zeros((height, width), dtype=np.uint8)
        origin_x, origin_y = origin
        for landmarks in hands:
            points = np.asarray(landmarks, dtype=np.float64)
            if points.shape[0] < 21:
                continue
            local = points.copy()
            local[:, 0] -= origin_x
            local[:, 1] -= origin_y
            palm = float(np.hypot(local[0, 0] - local[9, 0], local[0, 1] - local[9, 1]))
            if palm < 1.0:
                palm = float(np.hypot(local[0, 0] - local[5, 0], local[0, 1] - local[5, 1]))
            thickness = int(
                np.clip(
                    palm * self.settings.thickness_scale,
                    self.settings.min_thickness,
                    self.settings.max_thickness,
                )
            )
            integer = np.round(local).astype(np.int32)
            for chain in FINGER_CHAINS:
                cv2.polylines(canvas, [integer[list(chain)]], False, 255, thickness, cv2.LINE_AA)
            cv2.fillPoly(canvas, [integer[list(PALM_INDICES)]], 255)
            radius = max(3, thickness // 2)
            for tip in TIP_INDICES:
                center = (int(integer[tip, 0]), int(integer[tip, 1]))
                cv2.circle(canvas, center, radius, 255, -1, cv2.LINE_AA)
        dilate = self.settings.dilate_px
        if dilate > 0:
            if self._kernel is None or self._kernel_px != dilate:
                size = dilate * 2 + 1
                self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                self._kernel_px = dilate
            canvas = cv2.dilate(canvas, self._kernel)
        return canvas.astype(np.float32) * (1.0 / 255.0)
