"""Miles-inspired suit: procedural, region-aware, driven by the live frame.

Nothing here is a stored image. Per frame:
  * material: the suit region is repainted near-black, but its shading is the
    live image's own luminance (normalised inside the region), so folds,
    lighting and every movement of the real clothing stay visible; bright
    areas pick up cool blue highlights like the films' black suits,
  * web: red web lines radiating from the chest, drawn in the body frame
    (they follow shoulder position, rotation, scale and torso yaw), strongest
    on the shoulders and upper chest and fading down the torso,
  * emblem: a red spider with ink outline, anchored to the chest in the body
    frame, so it scales, rotates and foreshortens with the torso.
The comic renderer runs afterwards, so the suit gets the same ink, cel bands
and halftone as the rest of the image.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..config import CostumeConfig
from .body import BodyFrame

# Spider emblem in unit space (x right, y down; about one unit tall with legs).
_BODY_PARTS = (
    ((0.0, -0.30), (0.055, 0.060)),  # head
    ((0.0, -0.17), (0.078, 0.100)),  # thorax
    ((0.0, 0.07), (0.088, 0.200)),  # abdomen
)
_LEGS = (  # right side; mirrored for the left
    ((0.05, -0.21), (0.25, -0.33), (0.37, -0.53)),
    ((0.06, -0.17), (0.33, -0.21), (0.53, -0.37)),
    ((0.06, -0.12), (0.33, -0.04), (0.50, 0.13)),
    ((0.05, -0.08), (0.26, 0.11), (0.31, 0.38)),
)


def _bgr(rgb: tuple[float, float, float]) -> np.ndarray:
    return np.array(rgb[::-1], np.float32) * 255.0


def _ellipse_poly(center, radii, n: int = 28) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([center[0] + radii[0] * np.cos(t), center[1] + radii[1] * np.sin(t)], axis=1)


def emblem_shapes() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(filled body polygons, leg polylines) in unit space."""
    bodies = [_ellipse_poly(c, r) for c, r in _BODY_PARTS]
    legs = []
    for leg in _LEGS:
        pts = np.array(leg, np.float64)
        legs.append(pts)
        legs.append(pts * np.array([-1.0, 1.0]))
    return bodies, legs


class MilesSuitRenderer:
    def __init__(self, cfg: CostumeConfig) -> None:
        self.update(cfg)
        self._stats: np.ndarray | None = None  # smoothed (lo, hi) luminance inside the suit
        self._bodies, self._legs = emblem_shapes()

    def update(self, cfg: CostumeConfig) -> None:
        self.cfg = cfg
        self._black = _bgr(cfg.suit_color)
        self._highlight = _bgr(cfg.highlight_color)
        self._red = _bgr(cfg.accent_color)

    def reset(self) -> None:
        self._stats = None

    # material ------------------------------------------------------------------
    def material(self, base: np.ndarray, region: np.ndarray) -> np.ndarray:
        """Near-black suit (uint8 BGR) whose shading is the live luminance of the region.

        Shading depends only on luminance, so the whole material is one 256-entry colour
        table, re-fitted each frame to the region's own luminance range.
        """
        lum = cv2.GaussianBlur(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
        sample = lum[::4, ::4][region[::4, ::4] > 0.5]
        if sample.size > 50:
            lo, hi = np.percentile(sample, (5, 97)) / 255.0
            target = np.array([lo, max(hi, lo + 0.08)], np.float32)
            self._stats = target if self._stats is None else self._stats + 0.25 * (target - self._stats)
        lo, hi = (self._stats if self._stats is not None else np.array([0.0, 1.0], np.float32))
        x = np.linspace(0.0, 1.0, 256)
        shade = np.clip((x - lo) / max(hi - lo, 1e-3), 0.0, 1.0) ** 1.7 * self.cfg.highlight_strength
        table = self._black[None, :] + shade[:, None] * (self._highlight - self._black)[None, :]
        lut = np.clip(table, 0, 255).astype(np.uint8).reshape(256, 1, 3)
        return cv2.LUT(cv2.merge((lum, lum, lum)), lut)

    # web and emblem -------------------------------------------------------------
    def _line_px(self, body: BodyFrame, w: int) -> int:
        return max(1, int(round(self.cfg.line_px * body.scale / 220.0 * max(w, 320) / 640.0 * 1.6)))

    def _web_geometry(self) -> list[tuple[int, list[np.ndarray]]]:
        """Web strands in body space, grouped by drawing intensity (built once per config)."""
        key = (int(self.cfg.web_density), int(self.cfg.web_rings))
        cached = self.__dict__.get("_web_cache")
        if cached and cached[0] == key:
            return cached[1]
        n = max(4, key[0])
        angles = np.linspace(-math.pi, math.pi, n, endpoint=False) + math.pi / n

        def weight(y_body: float) -> float:
            # Strong across shoulders and upper chest, gone by the stomach.
            return float(np.clip(1.0 - max(0.0, y_body - 0.05) / 0.85, 0.0, 1.0))

        groups: dict[int, list[np.ndarray]] = {}

        def add(pts: np.ndarray, inten: float) -> None:
            level = int(round(inten * 8))  # 8 intensity buckets -> 8 draw calls
            if level > 0:
                groups.setdefault(level, []).append(pts)

        for a in angles:
            if abs(a + math.pi / 2) < math.radians(35):
                continue  # no strands straight up the neck
            radial = np.array([[math.cos(a) * r, math.sin(a) * r] for r in np.linspace(0.30, 2.0, 12)])
            for i in range(len(radial) - 1):
                add(radial[i:i + 2], weight(radial[i, 1]))
        for r in np.linspace(0.38, 1.7, max(1, key[1])):
            for a0 in angles:
                t = np.linspace(0.0, 1.0, 7)
                ang = a0 + t * (2 * math.pi / n)
                rad = r * (1.0 - 0.13 * np.sin(math.pi * t))  # web strands sag toward the centre
                pts = np.stack([np.cos(ang) * rad, np.sin(ang) * rad], axis=1)
                add(pts, weight(float(pts[:, 1].mean())))
        result = sorted(groups.items())
        self._web_cache = (key, result)
        return result

    def web_alpha(self, body: BodyFrame, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        canvas = np.zeros((h, w), np.uint8)
        origin = body.shoulder_mid + body.v * (0.40 * body.scale)
        thick = self._line_px(body, w)
        for level, strands in self._web_geometry():
            pix = [np.round(body.to_image(st, origin)).astype(np.int32) for st in strands]
            cv2.polylines(canvas, pix, False, int(255 * level / 8), thick, cv2.LINE_AA)
        return canvas.astype(np.float32) * (1.0 / 255.0)

    def emblem(self, body: BodyFrame, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """(red alpha, ink alpha) for the chest emblem."""
        h, w = shape
        red = np.zeros((h, w), np.uint8)
        ink = np.zeros((h, w), np.uint8)
        size = self.cfg.emblem_scale * body.scale
        origin = body.shoulder_mid + body.v * (0.43 * body.scale)
        leg_w = max(2, int(round(0.042 * size)))
        outline = max(2, int(round(0.028 * size)))
        bodies = [np.round(body.to_image(p, origin, size)).astype(np.int32) for p in self._bodies]
        legs = [np.round(body.to_image(p, origin, size)).astype(np.int32) for p in self._legs]
        for poly in bodies:
            cv2.fillPoly(ink, [poly], 255, cv2.LINE_AA)
            cv2.polylines(ink, [poly], True, 255, outline * 2, cv2.LINE_AA)
        for leg in legs:
            cv2.polylines(ink, [leg], False, 255, leg_w + outline * 2, cv2.LINE_AA)
        for poly in bodies:
            cv2.fillPoly(red, [poly], 255, cv2.LINE_AA)
        for leg in legs:
            cv2.polylines(red, [leg[:2]], False, 255, leg_w, cv2.LINE_AA)
            cv2.polylines(red, [leg[1:]], False, 255, max(1, int(leg_w * 0.7)), cv2.LINE_AA)
        return red.astype(np.float32) * (1.0 / 255.0), ink.astype(np.float32) * (1.0 / 255.0)

    # composition ------------------------------------------------------------------
    def _solid(self, shape: tuple[int, int], color: np.ndarray) -> np.ndarray:
        key = (shape, tuple(int(c) for c in color))
        cache = self.__dict__.setdefault("_solids", {})
        if key not in cache:
            img = np.empty(shape + (3,), np.uint8)
            img[:] = np.clip(color, 0, 255).astype(np.uint8)
            cache.clear()
            cache[key] = img
        return cache[key]

    @staticmethod
    def blend(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(alpha, np.float32)
        return cv2.blendLinear(src, dst, a, 1.0 - a)

    def render(self, base: np.ndarray, region: np.ndarray, body: BodyFrame | None,
               material: np.ndarray | None = None) -> np.ndarray:
        """Paint the suit onto `base` inside `region` (float mask). Work is limited to the region's box."""
        cfg = self.cfg
        out = base.copy()
        box = cv2.boundingRect((region > 0.02).astype(np.uint8))
        x, y, bw, bh = box
        if bw == 0 or bh == 0 or cfg.suit_strength <= 0:
            return out
        h, w = base.shape[:2]
        if material is None:
            material = self.material(base, region)
        sl = (slice(y, y + bh), slice(x, x + bw))
        reg = region[sl]
        roi = self.blend(out[sl], material[sl], reg * cfg.suit_strength)
        if body is not None:
            if cfg.red_balance > 0:
                web = self.web_alpha(body, (h, w))[sl] * reg * min(1.0, cfg.red_balance * 1.25)
                roi = self.blend(roi, self._solid((bh, bw), self._red), web)
            if cfg.emblem_scale > 0:
                red, ink = self.emblem(body, (h, w))
                keep = np.clip(reg * 1.6, 0.0, 1.0) * cfg.suit_strength
                roi = self.blend(roi, self._solid((bh, bw), self._black * 0.4), ink[sl] * keep)
                roi = self.blend(roi, self._solid((bh, bw), self._red), red[sl] * keep)
        out[sl] = roi
        return out
