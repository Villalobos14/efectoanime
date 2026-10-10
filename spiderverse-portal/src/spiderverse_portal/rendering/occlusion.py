"""Keep the real hands in front of the portal edge.

The fingers that frame the portal overlap its border. Without occlusion the
stylized layer (and the border ink) would be drawn over them and the portal
would look pasted on top of the hands instead of held between them.

Mask construction, per hand, inside a small ROI:
  1. capsules along every finger bone plus the palm polygon, sized from the
     hand's own palm length (scale-invariant),
  2. optional skin refinement: a Cr/Cb Gaussian sampled from the palm this
     frame decides which pixels in a slightly dilated capsule band are skin,
     so edges follow the real silhouette instead of the capsule geometry,
  3. a band weight: occlusion applies near the portal boundary and fades out
     deeper inside, so a hand fully inside the portal is stylized like the
     rest of the scene.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import OcclusionConfig
from ..tracking.hands import HAND_CONNECTIONS, INDEX_MCP, MIDDLE_MCP, PINKY_MCP, RING_MCP, WRIST, TrackedHand

PALM = (WRIST, 1, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
TIPS = (4, 8, 12, 16, 20)


def hand_capsule_mask(hand: TrackedHand, shape: tuple[int, int], offset: tuple[int, int],
                      width_ratio: float) -> np.ndarray:
    """uint8 mask (255 = hand) of finger capsules and palm, in ROI coordinates."""
    mask = np.zeros(shape, np.uint8)
    pts = hand.landmarks - np.array(offset, np.float64)
    palm = max(hand.palm_length, 1.0)
    thick = max(2, int(round(palm * width_ratio)))
    # Fingertip landmarks sit on the pad, not the end of the nail: extend the last bone a little.
    ext = pts.copy()
    for tip in TIPS:
        d = pts[tip] - pts[tip - 1]
        n = np.linalg.norm(d)
        if n > 1e-6:
            ext[tip] = pts[tip] + d / n * thick * 0.35
    cv2.fillConvexPoly(mask, np.round(ext[list(PALM)]).astype(np.int32), 255, lineType=cv2.LINE_AA)
    for a, b in HAND_CONNECTIONS:
        pa = tuple(int(round(v)) for v in ext[a])
        pb = tuple(int(round(v)) for v in ext[b])
        cv2.line(mask, pa, pb, 255, thick, lineType=cv2.LINE_AA)
    return mask


def skin_refine(roi_bgr: np.ndarray, capsule: np.ndarray, hand: TrackedHand, offset: tuple[int, int],
                max_mahalanobis: float = 3.0) -> np.ndarray:
    """Replace capsule edges with the skin silhouette in a band around them."""
    ycc = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    core = cv2.erode(capsule, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    samples = ycc[core > 200][:, 1:].astype(np.float64)
    if len(samples) < 40:
        return capsule
    mean = samples.mean(axis=0)
    cov = np.cov(samples.T) + np.eye(2) * 4.0
    inv = np.linalg.inv(cov)
    band = cv2.dilate(capsule, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    d = ycc[..., 1:].astype(np.float32) - mean.astype(np.float32)
    m2 = d[..., 0] ** 2 * inv[0, 0] + 2 * d[..., 0] * d[..., 1] * inv[0, 1] + d[..., 1] ** 2 * inv[1, 1]
    skin = ((m2 < max_mahalanobis ** 2) & (band > 0)).astype(np.uint8) * 255
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return cv2.max(core, cv2.min(skin, band))


class HandOccluder:
    def __init__(self, cfg: OcclusionConfig) -> None:
        self.cfg = cfg

    def mask(self, frame: np.ndarray, hands: list[TrackedHand], portal_alpha: np.ndarray,
             roi: tuple[int, int, int, int]) -> np.ndarray | None:
        """Float32 occlusion alpha over `roi` (1 = show the real camera pixel)."""
        if not self.cfg.enabled or not hands:
            return None
        x0, y0, x1, y1 = roi
        shape = (y1 - y0, x1 - x0)
        roi_img = frame[y0:y1, x0:x1]
        combined = np.zeros(shape, np.uint8)
        band_px = 1.0
        for hand in hands:
            capsule = hand_capsule_mask(hand, shape, (x0, y0), self.cfg.finger_width)
            if not capsule.any():
                continue
            if self.cfg.skin_refine:
                capsule = skin_refine(roi_img, capsule, hand, (x0, y0))
            combined = cv2.max(combined, capsule)
            band_px = max(band_px, hand.palm_length * self.cfg.edge_band)
        if not combined.any():
            return None
        occ = combined.astype(np.float32) * (1.0 / 255.0)
        if self.cfg.softness_px > 0:
            occ = cv2.GaussianBlur(occ, (0, 0), self.cfg.softness_px)
        # Distance from the portal boundary, measured inside the portal.
        inside = (portal_alpha > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
        near_edge = 1.0 - np.clip((dist - band_px * 0.6) / max(band_px * 0.4, 1.0), 0.0, 1.0)
        return occ * near_edge
