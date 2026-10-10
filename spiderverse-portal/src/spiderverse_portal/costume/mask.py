"""MILES_FULL_MASK (experimental): a face-following spider mask with comic eyes.

Geometry comes from the face landmarks of the frame being stylized:
  * the mask covers the head region (face oval plus hair cap) with the same
    live-shaded black material as the suit, so the head keeps its 3D shading
    and every turn and tilt,
  * red web strands radiate from the nose bridge through the face-oval points,
    with rings interpolated between them, so the web deforms with the head,
  * the white comic eyes are built per eye from its corners and lids: lid
    opening scales their height (blinks and squints read through the mask) and
    brow height lifts or slants their top edge (surprise, anger).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import CostumeConfig
from ..tracking.face import FACE_OVAL, FaceObservation
from .regions import face_axes
from .suit import MilesSuitRenderer, _bgr

NOSE_BRIDGE = 168
EYES = (  # (outer corner, inner corner, upper lid, lower lid, brow middle)
    (33, 133, 159, 145, 105),
    (263, 362, 386, 374, 334),
)
# Comic eye outline in eye space: x from inner corner (0) toward outer corner (1),
# y toward the chin, units of eye width. Top points come first.
_EYE_TOP = np.array([(-0.30, 0.12), (-0.22, -0.42), (0.20, -0.62), (0.68, -0.48), (0.98, -0.10)])
_EYE_BOTTOM = np.array([(0.74, 0.30), (0.28, 0.44), (-0.10, 0.36)])


def chaikin(poly: np.ndarray, iterations: int = 2) -> np.ndarray:
    p = poly
    for _ in range(iterations):
        q = np.roll(p, -1, axis=0)
        p = np.stack([0.75 * p + 0.25 * q, 0.25 * p + 0.75 * q], axis=1).reshape(-1, 2)
    return p


def eye_state(face: FaceObservation, idx: tuple[int, int, int, int, int], face_h: float) -> tuple[float, float]:
    """(openness 0..1, brow raise -1..1) measured from landmarks (no left/right label ambiguity)."""
    lm = face.landmarks
    outer, inner, up, low, brow = idx
    width = float(np.linalg.norm(lm[outer] - lm[inner])) + 1e-6
    opening = float(np.linalg.norm(lm[up] - lm[low])) / width
    openness = float(np.clip((opening - 0.08) / (0.30 - 0.08), 0.0, 1.0))
    center = (lm[outer] + lm[inner]) / 2.0
    brow_h = float(np.linalg.norm(lm[brow] - center)) / max(face_h, 1e-6)
    raise_ = float(np.clip((brow_h - 0.115) / 0.035, -1.0, 1.0))
    return openness, raise_


def eye_polygon(face: FaceObservation, idx, up: np.ndarray, face_h: float, scale: float) -> np.ndarray:
    lm = face.landmarks
    outer, inner = lm[idx[0]], lm[idx[1]]
    axis = outer - inner
    ew = float(np.linalg.norm(axis))
    axis = axis / max(ew, 1e-6)
    down = -up
    openness, raise_ = eye_state(face, idx, face_h)
    top = _EYE_TOP.copy()
    bottom = _EYE_BOTTOM.copy()
    top[:, 1] *= 0.22 + 0.78 * openness
    bottom[:, 1] *= 0.45 + 0.55 * openness
    top[1:4, 1] -= 0.16 * max(raise_, 0.0)
    top[1, 1] += 0.28 * max(-raise_, 0.0)  # frown: inner top edge drops into an angry slant
    local = np.vstack([top, bottom])
    local = (local - np.array([0.35, 0.0])) * scale + np.array([0.35, -0.12])  # grow away from the nose
    pts = inner + np.outer(local[:, 0] * ew, axis) + np.outer(local[:, 1] * ew, down)
    return chaikin(pts, 2)


class SpiderMaskRenderer:
    def __init__(self, cfg: CostumeConfig, suit: MilesSuitRenderer) -> None:
        self.cfg = cfg
        self.suit = suit

    def update(self, cfg: CostumeConfig) -> None:
        self.cfg = cfg

    def web_alpha(self, face: FaceObservation, shape: tuple[int, int], fh: float) -> np.ndarray:
        h, w = shape
        canvas = np.zeros((h, w), np.uint8)
        lm = face.landmarks
        center = lm[NOSE_BRIDGE]
        oval = lm[list(FACE_OVAL)]
        rim = center + (oval - center) * 1.18  # past the face edge, over the hair cap
        thick = max(1, int(round(self.cfg.line_px * fh / 160.0)))
        for p in rim[::3]:
            cv2.line(canvas, tuple(np.round(center).astype(int)), tuple(np.round(p).astype(int)), 255, thick, cv2.LINE_AA)
        for f in (0.32, 0.55, 0.78, 1.0):
            ring = center + (rim - center) * f
            cv2.polylines(canvas, [np.round(ring).astype(np.int32)], True, 255, thick, cv2.LINE_AA)
        return canvas.astype(np.float32) * (1.0 / 255.0)

    def render(self, img: np.ndarray, base: np.ndarray, head: np.ndarray, face: FaceObservation | None,
               material: np.ndarray | None = None) -> np.ndarray:
        if face is None:
            return img
        h, w = img.shape[:2]
        x, y, bw, bh = cv2.boundingRect((head > 0.02).astype(np.uint8))
        if bw == 0 or bh == 0:
            return img
        up, _, fh, _ = face_axes(face)
        if material is None:
            material = self.suit.material(base, head)
        sl = (slice(y, y + bh), slice(x, x + bw))
        blend = self.suit.blend
        reg = head[sl]
        roi = blend(img[sl], material[sl], reg)
        web = self.web_alpha(face, (h, w), fh)[sl] * reg * min(1.0, self.cfg.red_balance * 1.25)
        roi = blend(roi, self.suit._solid((bh, bw), _bgr(self.cfg.accent_color)), web)
        ink = np.zeros((h, w), np.uint8)
        white = np.zeros((h, w), np.uint8)
        outline = max(2, int(round(fh * 0.035)))
        for idx in EYES:
            poly = np.round(eye_polygon(face, idx, up, fh, self.cfg.mask_eye_scale)).astype(np.int32)
            cv2.fillPoly(ink, [poly], 255, cv2.LINE_AA)
            cv2.polylines(ink, [poly], True, 255, outline * 2, cv2.LINE_AA)
            cv2.fillPoly(white, [poly], 255, cv2.LINE_AA)
        out = img.copy()
        out[sl] = roi
        # Eyes may extend past the head box when the head turns; composite them on the full frame.
        ex, ey, ew, eh = cv2.boundingRect(ink)
        if ew and eh:
            esl = (slice(ey, ey + eh), slice(ex, ex + ew))
            ink_a = ink[esl].astype(np.float32) * (1.0 / 255.0)
            white_a = white[esl].astype(np.float32) * (1.0 / 255.0)
            e = blend(out[esl], self.suit._solid((eh, ew), np.array([14, 8, 12], np.float32)), ink_a)
            out[esl] = blend(e, self.suit._solid((eh, ew), np.array([246, 244, 240], np.float32)), white_a)
        return out
