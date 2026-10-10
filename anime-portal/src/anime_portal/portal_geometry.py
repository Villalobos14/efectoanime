"""Quadrilateral geometry for the hand-controlled portal.

Corners stay in a stable order across frames. A quad is accepted only when it
is convex, simple, and inside the configured area, angle, and aspect limits.
"""

from __future__ import annotations

import itertools

import numpy as np


def signed_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(points: np.ndarray) -> float:
    return abs(signed_area(points))


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1e-8) -> bool:
    """True when open segments AB and CD properly cross.

    Shared endpoints and grazing touches are not treated as intersections, so a
    normal polygon edge chain is not flagged.
    """
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    if abs(o1) <= eps or abs(o2) <= eps or abs(o3) <= eps or abs(o4) <= eps:
        return False
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def is_self_intersecting(points: np.ndarray) -> bool:
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    for i in range(n):
        a1 = pts[i]
        a2 = pts[(i + 1) % n]
        for j in range(i + 1, n):
            shares_vertex = (j == (i + 1) % n) or (i == (j + 1) % n)
            if shares_vertex:
                continue
            if segments_intersect(a1, a2, pts[j], pts[(j + 1) % n]):
                return True
    return False


def is_convex(points: np.ndarray, eps: float = 1e-6) -> bool:
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    sign = 0
    for i in range(n):
        cross = _orient(pts[i], pts[(i + 1) % n], pts[(i + 2) % n])
        if abs(cross) <= eps:
            continue
        current = 1 if cross > 0 else -1
        if sign == 0:
            sign = current
        elif current != sign:
            return False
    return sign != 0


def edge_lengths(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    nxt = np.roll(pts, -1, axis=0)
    return np.linalg.norm(nxt - pts, axis=1)


def interior_angles_deg(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    angles = np.empty(n, dtype=np.float64)
    for i in range(n):
        prev_pt = pts[(i - 1) % n]
        cur = pts[i]
        nxt = pts[(i + 1) % n]
        v1 = prev_pt - cur
        v2 = nxt - cur
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom <= 1e-8:
            angles[i] = 0.0
            continue
        cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        angles[i] = float(np.degrees(np.arccos(cosine)))
    return angles


def canonical_order(points: np.ndarray) -> np.ndarray:
    """Order 4 points around their centroid, starting at the top-most point.

    Image y grows downward. The winding is chosen so the shoelace signed area
    is positive.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) != 4:
        raise ValueError("a portal quad needs 4 corners")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    if signed_area(ordered) < 0:
        ordered = ordered[::-1]
    start = int(np.lexsort((ordered[:, 0], ordered[:, 1]))[0])
    return np.roll(ordered, -start, axis=0).copy()


def match_corners(previous: np.ndarray | None, current: np.ndarray) -> np.ndarray:
    """Permute ``current`` so each corner stays with its previous identity.

    Four points have 24 permutations. The assignment with the smallest squared
    travel is used, which prevents adjacent fingertips from swapping places.
    """
    cur = np.asarray(current, dtype=np.float64).reshape(4, 2)
    if previous is None:
        return canonical_order(cur)
    prev = np.asarray(previous, dtype=np.float64).reshape(4, 2)
    best: np.ndarray | None = None
    best_cost = float("inf")
    for perm in itertools.permutations(range(4)):
        candidate = cur[list(perm)]
        cost = float(np.sum((candidate - prev) ** 2))
        if cost < best_cost:
            best_cost = cost
            best = candidate
    assert best is not None
    return best.copy()


def validate_quad(
    points: np.ndarray,
    image_area: float,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    min_corner_angle_deg: float,
    max_side_ratio: float,
) -> tuple[bool, str]:
    """Return whether ``points`` can be used as a stable portal."""
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return False, "invalid corners"
    if is_self_intersecting(pts):
        return False, "self-intersection"
    if not is_convex(pts):
        return False, "not convex"
    if image_area <= 0:
        return False, "invalid image"
    area_ratio = polygon_area(pts) / float(image_area)
    if area_ratio < min_area_ratio:
        return False, "too small"
    if area_ratio > max_area_ratio:
        return False, "too large"
    angles = interior_angles_deg(pts)
    if float(np.min(angles)) < min_corner_angle_deg:
        return False, "sharp corner"
    sides = edge_lengths(pts)
    shortest = float(np.min(sides))
    if shortest < 1.0:
        return False, "collapsed"
    if float(np.max(sides)) / shortest > max_side_ratio:
        return False, "extreme aspect"
    return True, "ok"
