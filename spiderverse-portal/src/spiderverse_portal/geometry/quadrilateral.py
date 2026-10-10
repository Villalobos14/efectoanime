"""Quadrilateral ordering and validation for the hand-framed portal.

All thresholds are relative (area as a fraction of the frame, lengths as a
fraction of the frame diagonal or of the quad itself) so they hold at any
camera resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def polygon_area(points: np.ndarray) -> float:
    """Signed shoelace area; positive for counter-clockwise in y-up coordinates."""
    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def order_corners(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order 4 points clockwise on screen (image y down), starting top-left.

    Returns (ordered_points, permutation) where ordered = points[permutation].
    Sorting by angle around the centroid yields a simple polygon whenever the
    points are in convex position; convexity is checked separately.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (4, 2):
        raise ValueError("expected 4x2 points")
    center = p.mean(axis=0)
    angles = np.arctan2(p[:, 1] - center[1], p[:, 0] - center[0])
    perm = np.argsort(angles)  # increasing angle = clockwise on screen because y points down
    # Rotate so the first corner is the one closest to the top-left.
    start = int(np.argmin((p[perm, 0] + p[perm, 1])))
    perm = np.roll(perm, -start)
    return p[perm], perm


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def is_self_intersecting(quad: np.ndarray) -> bool:
    q = np.asarray(quad, dtype=np.float64)
    return _segments_intersect(q[0], q[1], q[2], q[3]) or _segments_intersect(q[1], q[2], q[3], q[0])


def is_convex(quad: np.ndarray) -> bool:
    q = np.asarray(quad, dtype=np.float64)
    signs = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross)
    signs = np.array(signs)
    return bool(np.all(signs > 0) or np.all(signs < 0))


def interior_angles_deg(quad: np.ndarray) -> np.ndarray:
    q = np.asarray(quad, dtype=np.float64)
    out = []
    for i in range(4):
        prev, cur, nxt = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = prev - cur, nxt - cur
        n = np.linalg.norm(v1) * np.linalg.norm(v2)
        cosang = float(np.dot(v1, v2) / n) if n > 1e-9 else 1.0
        out.append(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
    return np.array(out)


def edge_lengths(quad: np.ndarray) -> np.ndarray:
    q = np.asarray(quad, dtype=np.float64)
    return np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)


@dataclass
class QuadCheck:
    valid: bool
    score: float  # 0..1 geometric plausibility
    ordered: np.ndarray | None
    permutation: np.ndarray | None
    area_ratio: float = 0.0
    reasons: list[str] = field(default_factory=list)


def validate_quad(
    points: np.ndarray,
    frame_size: tuple[int, int],
    *,
    min_area_ratio: float = 0.025,
    max_area_ratio: float = 0.95,
    min_angle_deg: float = 25.0,
    min_edge_ratio: float = 0.12,
) -> QuadCheck:
    """Validate 4 candidate corners. `frame_size` is (width, height)."""
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (4, 2) or not np.all(np.isfinite(p)):
        return QuadCheck(False, 0.0, None, None, reasons=["bad points"])
    ordered, perm = order_corners(p)
    reasons: list[str] = []
    if is_self_intersecting(ordered):
        reasons.append("self-intersecting")
    if not is_convex(ordered):
        reasons.append("not convex")
    w, h = frame_size
    area_ratio = abs(polygon_area(ordered)) / float(w * h)
    if area_ratio < min_area_ratio:
        reasons.append("too small")
    if area_ratio > max_area_ratio:
        reasons.append("too large")
    angles = interior_angles_deg(ordered)
    if angles.min() < min_angle_deg:
        reasons.append("collapsed angle")
    edges = edge_lengths(ordered)
    edge_ratio = float(edges.min() / max(edges.max(), 1e-9))
    if edge_ratio < min_edge_ratio:
        reasons.append("degenerate edge")

    # Soft score: how comfortably the quad clears each threshold.
    def ramp(value: float, lo: float, hi: float) -> float:
        return float(np.clip((value - lo) / max(hi - lo, 1e-9), 0.0, 1.0))

    score = (
        ramp(area_ratio, min_area_ratio, min_area_ratio * 2.5)
        * ramp(float(angles.min()), min_angle_deg, min_angle_deg + 25.0)
        * ramp(edge_ratio, min_edge_ratio, min_edge_ratio + 0.2)
    ) ** (1.0 / 3.0)
    valid = not reasons
    return QuadCheck(valid, score if valid else 0.0, ordered, perm, area_ratio, reasons)
