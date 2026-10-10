"""PROTOTYPE Miles-inspired character, built procedurally and exported as a rigged glTF 2.0 (GLB).

This is placeholder geometry, not a production character. It exists so the
whole runtime path (GLB loading, skinning, morph targets, toon materials,
tracking-driven rig) is exercised with an honest asset while a properly
modelled one is obtained (see docs/CHARACTER_ASSET_SPEC.md). Any GLB that
follows that spec replaces it without code changes.

What is built:
  * head / mask: the MediaPipe canonical face, smoothed into a fabric mask
    (nose softened, eye sockets partly filled, lips flattened), merged into a
    skull by a radial field with a harmonic blend band. Because it is built in
    canonical-face coordinates, the head joint's frame IS the frame of the face
    tracker's transformation matrix, which is what registers it to the face.
  * lenses: angular Spider-Man lenses projected onto the mask surface, each a
    separate mesh with six morph targets per side (blink, squint, wide,
    angry, sad, happy) and a red contour band in the same mesh.
  * neck, trapezius, shoulders, chest, abdomen and arms: a signed-distance
    model (smooth unions of ellipsoids and round cones) polygonised with
    surface nets, skinned to a 13-joint upper-body skeleton with weights from
    the primitives' blend.
  * materials: mask fabric and suit fabric with procedural web parameters
    (the web is evaluated per pixel in the shader from rest-pose positions),
    a procedurally drawn red spider emblem texture on the chest, white lenses.

Units: metres, +Y up, the character faces +Z (glTF conventions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import meshtools as mt
from .face_geometry import canonical_face, default_task_path
from .gltf import GltfBuilder

PROTOTYPE_VERSION = 3
H0 = np.array([0.0, 1.62, 0.0])  # head joint = canonical face origin, model space (m)

# name, parent, rest position (m). Rest rotations are identity.
JOINTS: list[tuple[str, str | None, tuple[float, float, float]]] = [
    ("root", None, (0.0, 0.95, -0.02)),
    ("spine", "root", (0.0, 1.10, -0.02)),
    ("chest", "spine", (0.0, 1.28, -0.025)),
    ("neck", "chest", (0.0, 1.45, -0.045)),
    ("head", "neck", tuple(H0)),
    ("shoulder.L", "chest", (0.035, 1.455, -0.04)),
    ("upper_arm.L", "shoulder.L", (0.19, 1.425, -0.035)),
    ("forearm.L", "upper_arm.L", (0.0, 0.0, 0.0)),  # filled in below (A-pose)
    ("hand.L", "forearm.L", (0.0, 0.0, 0.0)),
    ("shoulder.R", "chest", (-0.035, 1.455, -0.04)),
    ("upper_arm.R", "shoulder.R", (-0.19, 1.425, -0.035)),
    ("forearm.R", "upper_arm.R", (0.0, 0.0, 0.0)),
    ("hand.R", "forearm.R", (0.0, 0.0, 0.0)),
]
ARM_DOWN_DEG = 58.0  # A-pose: arms this far below horizontal
UPPER_ARM_LEN, FOREARM_LEN = 0.285, 0.255


def _joint_positions() -> dict[str, np.ndarray]:
    pos = {name: np.array(p, np.float64) for name, _, p in JOINTS}
    a = math.radians(ARM_DOWN_DEG)
    for side, sx in (("L", 1.0), ("R", -1.0)):
        d = np.array([sx * math.cos(a), -math.sin(a), 0.0])
        pos[f"forearm.{side}"] = pos[f"upper_arm.{side}"] + d * UPPER_ARM_LEN
        pos[f"hand.{side}"] = pos[f"forearm.{side}"] + d * FOREARM_LEN
    return pos


JOINT_POS = _joint_positions()
JOINT_NAMES = [j[0] for j in JOINTS]
JOINT_INDEX = {n: i for i, n in enumerate(JOINT_NAMES)}


def canon_to_model(p_cm: np.ndarray) -> np.ndarray:
    return H0 + np.asarray(p_cm, np.float64) / 100.0


# --------------------------------------------------------------------------- face mask
NOSE_IDS = (1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 131, 134, 168, 195, 197, 198, 209, 217, 218, 219, 220,
            236, 237, 238, 239, 241, 242, 248, 250, 274, 275, 278, 279, 294, 326, 327, 344, 360, 363, 393, 420,
            437, 438, 439, 440, 456, 457, 458, 459, 461, 462)
EYE_RING_R = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
EYE_RING_L = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
LIPS_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)


def _inside(poly: np.ndarray, pts: np.ndarray, grow: float) -> np.ndarray:
    c = poly.mean(axis=0)
    p = c + (poly - c) * grow
    return np.array([cv2.pointPolygonTest(p.astype(np.float32), (float(x), float(y)), False) >= 0
                     for x, y in pts])


def mask_face(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Canonical face (cm) -> fabric mask surface: keep the large forms, soften the features."""
    base = mt.taubin_smooth(verts, tris, iterations=60)
    detail = verts - base
    k = np.full(len(verts), 0.62)
    xy = verts[:, :2]
    k[list(NOSE_IDS)] = 0.6
    for ring in (EYE_RING_R, EYE_RING_L):
        k[_inside(verts[list(ring), :2], xy, 1.9)] = 0.25  # fabric bridges the eye socket
    k[_inside(verts[list(LIPS_OUTER), :2], xy, 1.15)] = 0.12  # no lips under the mask
    k = np.clip(mt.diffuse(k, tris, iterations=6, rate=0.5), 0.0, 1.0)
    out = base + detail * k[:, None]
    # Fabric is pulled taut across the eye sockets: lift them toward the brow / cheekbone plane.
    for ring in (EYE_RING_R, EYE_RING_L):
        sel = _inside(verts[list(ring), :2], xy, 1.6)
        out[sel, 2] += 0.35
    return mt.taubin_smooth(out, tris, iterations=4)


# --------------------------------------------------------------------------- head
HEAD_CENTER = np.array([0.0, 1.5, -2.0])  # cm, canonical; rays for the radial field start here
SKULL_CENTER = np.array([0.0, 2.8, -4.0])
SKULL_AXES = np.array([7.9, 10.8, 9.8])


def _ellipsoid_hit(origin: np.ndarray, dirs: np.ndarray, center: np.ndarray, axes: np.ndarray) -> np.ndarray:
    o = (origin - center) / axes
    d = dirs / axes
    a = (d * d).sum(1)
    b = 2 * (d * o).sum(1)
    c = (o * o).sum() - 1.0
    return (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / (2 * a)


def _skull_radius(dirs: np.ndarray) -> np.ndarray:
    r = _ellipsoid_hit(HEAD_CENTER, dirs, SKULL_CENTER, SKULL_AXES)
    # Slightly flatter temples/sides, and a narrower jaw region low on the head.
    p = HEAD_CENTER + dirs * r[:, None]
    low = np.clip((-p[:, 1] - 1.0) / 7.0, 0.0, 1.0)
    side = np.abs(dirs[:, 0])
    return r * (1.0 - 0.05 * side * (0.4 + 0.6 * low))


def _grid_distance(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Chamfer-ish distance (cells) from cells where mask is False, longitude wraps."""
    d = np.where(mask, 1e6, 0.0)
    for _ in range(iterations):
        nb = np.minimum.reduce([
            np.roll(d, 1, axis=1) + 1, np.roll(d, -1, axis=1) + 1,
            np.vstack([d[:1], d[:-1]]) + 1, np.vstack([d[1:], d[-1:]]) + 1,
        ])
        d = np.minimum(d, nb)
    return d


def build_head(face_mask: np.ndarray, face_tris: np.ndarray, n_lat: int = 120, n_lon: int = 160):
    """Radial-field head mesh in canonical cm: returns (vertices, triangles, face_weight per vertex)."""
    theta = np.linspace(0, math.pi, n_lat + 1)[1:-1]  # rings, poles added separately
    phi = np.linspace(0, 2 * math.pi, n_lon, endpoint=False)  # 0 = back of the head (seam hidden)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    dirs = np.stack([np.sin(th) * np.sin(ph + math.pi), np.cos(th), np.sin(th) * np.cos(ph + math.pi)], -1)
    flat = dirs.reshape(-1, 3)
    r_skull = _skull_radius(flat).reshape(th.shape)
    t, idx = mt.ray_triangles(np.repeat(HEAD_CENTER[None], len(flat), 0), flat, face_mask, face_tris)
    hit = (idx >= 0).reshape(th.shape)
    r_face = np.where(hit, t.reshape(th.shape), 0.0)

    inside_d = _grid_distance(hit, 40)  # distance into the face region
    outside_d = _grid_distance(~hit, 40)  # distance away from it
    face_fixed = inside_d >= 2
    skull_fixed = outside_d >= 16
    r = np.where(face_fixed, r_face, r_skull)
    free = ~(face_fixed | skull_fixed)
    # Harmonic (membrane) blend between face and skull: smooth, no ridge at the face edge.
    for _ in range(1500):
        avg = 0.25 * (np.roll(r, 1, 1) + np.roll(r, -1, 1) + np.vstack([r[:1], r[:-1]]) + np.vstack([r[1:], r[-1:]]))
        r = np.where(free, avg, r)
    # Light relaxation everywhere except the face interior, to soften the band boundary.
    relax = inside_d < 5
    for _ in range(30):
        avg = 0.25 * (np.roll(r, 1, 1) + np.roll(r, -1, 1) + np.vstack([r[:1], r[:-1]]) + np.vstack([r[1:], r[-1:]]))
        r = np.where(relax, 0.5 * r + 0.5 * avg, r)

    pts = HEAD_CENTER + dirs * r[..., None]
    top = HEAD_CENTER + np.array([0, 1, 0]) * r[0].mean()
    bottom = HEAD_CENTER + np.array([0, -1, 0]) * r[-1].mean()
    verts = np.vstack([top[None], pts.reshape(-1, 3), bottom[None]])
    face_w = np.concatenate([[0.0], np.clip(inside_d / 6.0, 0, 1).ravel(), [0.0]])

    def vid(i, j):
        return 1 + i * n_lon + (j % n_lon)

    tris = []
    for j in range(n_lon):
        tris.append((0, vid(0, j + 1), vid(0, j)))
    for i in range(n_lat - 2):
        for j in range(n_lon):
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
            tris.append((a, b, c))
            tris.append((a, c, d))
    last = len(verts) - 1
    for j in range(n_lon):
        tris.append((vid(n_lat - 2, j), vid(n_lat - 2, j + 1), last))
    tris = np.asarray(tris, np.int64)
    # Orient outward.
    fn = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    out = verts[tris].mean(1) - HEAD_CENTER
    if (fn * out).sum() < 0:
        tris = tris[:, ::-1].copy()
    return verts, tris, face_w


# --------------------------------------------------------------------------- lenses
# Left lens (subject's left = +X), canonical cm. Index 0 is the sharp outer tip; then the top edge
# inward, the rounded inner edge, and the bottom edge back out to the tip.
LENS_NEUTRAL = np.array([
    (6.50, 5.42),                                        # 0 outer tip (sharp, toward the temple)
    (5.45, 5.30), (4.10, 5.16), (2.80, 4.90),            # 1-3 top edge, slanting down toward the nose
    (1.70, 4.50), (1.08, 3.85), (0.92, 3.00),            # 4-6 inner-top corner, inner edge
    (1.14, 2.12), (1.84, 1.30), (3.02, 0.90),            # 7-9 inner-bottom, bottom edge
    (4.34, 1.10), (5.42, 2.02), (6.12, 3.55),            # 10-12 bottom edge sweeping up to the tip
])
TOP, INNER, BOTTOM = (1, 2, 3, 4), (5, 6, 7), (8, 9, 10, 11, 12)
LENS_BORDER_CM = 0.62
LENS_LIFT_CM = 0.16
EXPRESSIONS = ("blink", "squint", "wide", "angry", "sad", "happy")


def _lens_morph(name: str) -> np.ndarray:
    p = LENS_NEUTRAL.copy()
    x, y = p[:, 0], p[:, 1]
    if name == "blink":
        # Collapse onto a gentle arc through the lens: top and bottom edges meet (slit).
        mid = 2.95 + 0.30 * np.sin(np.clip((x - 0.9) / 5.6, 0, 1) * math.pi) + 0.25 * (x - 0.9) / 5.6
        side = np.where(np.isin(np.arange(len(p)), BOTTOM + (7,)), -1.0, 1.0)
        side[0] = 0.0
        p[:, 1] = mid + side * 0.10
        p[5:7, 0] += 0.45  # inner edge closes in
    elif name == "squint":
        idx = list(BOTTOM) + [7]
        p[idx, 1] += (3.10 - p[idx, 1]) * 0.55
        p[list(TOP), 1] -= 0.45
        p[0, 1] -= 0.25
    elif name == "wide":
        p[list(TOP), 1] += np.array([0.35, 0.55, 0.55, 0.45])
        p[list(BOTTOM), 1] -= np.array([0.45, 0.45, 0.40, 0.25, 0.10])
        p[list(INNER), 0] -= 0.18
        p[0, 1] += 0.25
    elif name == "angry":
        # Brow pushes the inner-top edge down: a hard slant toward the nose.
        p[list(TOP), 1] -= np.array([0.45, 0.95, 1.45, 1.75])
        p[5, 1] -= 1.30
        p[6, 1] -= 0.60  # keep the inner edge monotonic (no fold) under the lowered inner corner
        p[0, 1] -= 0.15
        p[list(BOTTOM), 1] += 0.18
    elif name == "sad":
        p[list(TOP), 1] += np.array([-0.30, 0.0, 0.30, 0.45])
        p[0, 1] -= 0.85
        p[0, 0] -= 0.25
        p[12, 1] -= 0.55
        p[11, 1] -= 0.30
    elif name == "happy":
        # Lower edge pushed up in the middle: smiling "^" eyes.
        p[list(BOTTOM), 1] += np.array([1.05, 1.55, 1.50, 1.00, 0.45])
        p[7, 1] += 0.55
        p[list(TOP), 1] += 0.15
    else:
        raise ValueError(name)
    return p


def _catmull_rom(points: np.ndarray, counts: list[int]) -> np.ndarray:
    """Open spline through points (first and last are corners), `counts` samples per segment."""
    p = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])
    out = []
    for i, n in enumerate(counts):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        for t in np.arange(n) / n:
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.asarray(out)


def outline_is_simple(poly: np.ndarray) -> bool:
    """True when the closed polyline does not cross itself."""
    n = len(poly)
    a, b = poly, np.roll(poly, -1, axis=0)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            p, r = a[i], b[i] - a[i]
            q, s = a[j], b[j] - a[j]
            den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-12:
                continue
            t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den
            u = ((q - p)[0] * r[1] - (q - p)[1] * r[0]) / den
            if 0 < t < 1 and 0 < u < 1:
                return False
    return True


def lens_outline(ctrl: np.ndarray, samples: int = 110) -> np.ndarray:
    closed = np.vstack([ctrl, ctrl[:1]])  # tip ... back to tip (the tip stays a corner)
    seg = np.linalg.norm(np.diff(_neutral_closed(), axis=0), axis=1)
    counts = np.maximum(2, np.round(seg / seg.sum() * samples)).astype(int).tolist()
    return _catmull_rom(closed, counts)


def _neutral_closed() -> np.ndarray:
    return np.vstack([LENS_NEUTRAL, LENS_NEUTRAL[:1]])


def _outward_offset(poly: np.ndarray, width: float, miter_limit: float = 1.7) -> np.ndarray:
    prev = np.roll(poly, 1, axis=0)
    nxt = np.roll(poly, -1, axis=0)
    t1 = poly - prev
    t2 = nxt - poly
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-9)
    t2 /= np.maximum(np.linalg.norm(t2, axis=1, keepdims=True), 1e-9)
    area = 0.5 * float(np.sum(poly[:, 0] * nxt[:, 1] - nxt[:, 0] * poly[:, 1]))
    sgn = 1.0 if area > 0 else -1.0  # counter-clockwise: outward normal is (dy, -dx)
    n1 = sgn * np.stack([t1[:, 1], -t1[:, 0]], 1)
    n2 = sgn * np.stack([t2[:, 1], -t2[:, 0]], 1)
    m = n1 + n2
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    cos_half = np.clip((m * n1).sum(1), 1.0 / miter_limit, 1.0)
    return poly + m * (width / cos_half)[:, None]


LENS_RINGS = (0.0, 0.18, 0.36, 0.52, 0.66, 0.78, 0.88, 0.95, 1.0, 1.35, 1.7, 2.0)  # 0 centre ... 1 white edge ... 2 outer red edge


def _lens_rings_2d(outline: np.ndarray) -> list[np.ndarray]:
    c = outline.mean(axis=0)
    border = _outward_offset(outline, LENS_BORDER_CM)
    rings = []
    for r in LENS_RINGS:
        if r <= 1.0:
            rings.append(c + (outline - c) * r)
        else:
            rings.append(outline + (border - outline) * (r - 1.0))
    return rings


def _project_front(xy: np.ndarray, verts: np.ndarray, tris: np.ndarray, normals: np.ndarray):
    o = np.c_[xy, np.full(len(xy), 40.0)]
    d = np.tile([0.0, 0.0, -1.0], (len(xy), 1))
    t, idx = mt.ray_triangles(o, d, verts, tris, both_sides=True)
    if (idx < 0).any():
        raise RuntimeError("lens point missed the head surface")
    p = o + d * t[:, None]
    tri = tris[idx]
    a, b, c = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    w1 = (d11 * d20 - d01 * d21) / den
    w2 = (d00 * d21 - d01 * d20) / den
    w0 = 1 - w1 - w2
    n = normals[tri[:, 0]] * w0[:, None] + normals[tri[:, 1]] * w1[:, None] + normals[tri[:, 2]] * w2[:, None]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return p + n * LENS_LIFT_CM, n


def build_lens(sign: float, head_v: np.ndarray, head_t: np.ndarray, head_n: np.ndarray):
    """One lens (sign=+1 left/+X, -1 right). Returns positions (cm), normals, uv, triangles, {expr: positions}."""
    cen = head_v[head_t].mean(1)
    lo = np.array([min(0.0, sign * 8.0) - 1.0, -1.5])
    hi = np.array([max(0.0, sign * 8.0) + 1.0, 8.0])
    sel = (cen[:, 2] > -5.0) & (cen[:, 0] > lo[0]) & (cen[:, 0] < hi[0]) & (cen[:, 1] > lo[1]) & (cen[:, 1] < hi[1])
    tris_front = head_t[sel]  # front of the head around this lens only (keeps ray casting cheap)

    def place(ctrl):
        outline = lens_outline(ctrl)
        rings = _lens_rings_2d(outline)
        xy = np.vstack(rings) * np.array([sign, 1.0])
        return _project_front(xy, head_v, tris_front, head_n), outline, rings

    (pos, nrm), outline, rings = place(LENS_NEUTRAL)
    n = len(outline)
    c = outline.mean(axis=0)
    half_h = max(1e-6, (outline[:, 1].max() - outline[:, 1].min()) / 2)
    uv = np.vstack([np.c_[np.full(n, r), (ring[:, 1] - c[1]) / half_h] for r, ring in zip(LENS_RINGS, rings)])
    tris = []
    for k in range(len(LENS_RINGS) - 1):
        for j in range(n):
            a, b = k * n + j, k * n + (j + 1) % n
            c2, d = (k + 1) * n + (j + 1) % n, (k + 1) * n + j
            tris += [(a, b, c2), (a, c2, d)]
    tris = np.asarray(tris, np.int64)
    # Ring 0 is the centre (all points coincide): degenerate triangles there are harmless.
    fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    good = np.linalg.norm(fn, axis=1) > 1e-12
    if (fn[good][:, 2]).sum() < 0:
        tris = tris[:, ::-1].copy()
    targets = {}
    for name in EXPRESSIONS:
        (p_m, _), _, _ = place(_lens_morph(name))
        targets[name] = p_m
    return pos, nrm, uv, tris, targets


# --------------------------------------------------------------------------- body (SDF)
def _ellipsoid(p, c, r):
    q = (p - c) / r
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / r, axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def _round_cone(p, a, b, r1, r2):
    ba = b - a
    l2 = float(ba @ ba)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = pa @ ba
    z = y - l2
    x2 = ((pa * l2 - y[:, None] * ba) ** 2).sum(1)
    y2 = y * y * l2
    z2 = z * z * l2
    k = math.copysign(1.0, rr) * rr * rr * x2
    d3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - r1
    d1 = np.sqrt(x2 + z2) * il2 - r2
    d2 = np.sqrt(x2 + y2) * il2 - r1
    return np.where(np.sign(z) * a2 * z2 > k, d1, np.where(np.sign(y) * a2 * y2 < k, d2, d3))


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def _t_along(p, a, b):
    ba = b - a
    return np.clip(((p - a) @ ba) / float(ba @ ba), 0.0, 1.0)


@dataclass
class Prim:
    name: str
    kind: str
    args: tuple
    group: str  # torso | arm.L | arm.R

    def dist(self, p):
        return _ellipsoid(p, *self.args) if self.kind == "ell" else _round_cone(p, *self.args)


def body_primitives() -> list[Prim]:
    J = JOINT_POS
    prims = [
        Prim("neck", "cone", (np.array([0, 1.405, -0.050]), np.array([0, 1.600, -0.022]), 0.058, 0.054), "torso"),
        Prim("chest", "ell", (np.array([0, 1.305, -0.028]), np.array([0.150, 0.150, 0.104])), "torso"),
        Prim("upper_chest", "ell", (np.array([0, 1.392, -0.035]), np.array([0.172, 0.075, 0.090])), "torso"),
        Prim("ribs", "ell", (np.array([0, 1.200, -0.025]), np.array([0.138, 0.140, 0.097])), "torso"),
        Prim("abdomen", "ell", (np.array([0, 1.060, -0.020]), np.array([0.128, 0.150, 0.092])), "torso"),
        Prim("hips", "ell", (np.array([0, 0.940, -0.020]), np.array([0.150, 0.090, 0.100])), "torso"),
    ]
    for side, sx in (("L", 1.0), ("R", -1.0)):
        sh = J[f"upper_arm.{side}"]
        prims += [
            Prim(f"trap.{side}", "cone", (np.array([0, 1.505, -0.058]), np.array([sx * 0.150, 1.448, -0.045]),
                                          0.046, 0.040), "torso"),
            Prim(f"pec.{side}", "ell", (np.array([sx * 0.074, 1.342, 0.046]), np.array([0.090, 0.058, 0.030])),
                 "torso"),
            Prim(f"lat.{side}", "ell", (np.array([sx * 0.095, 1.270, -0.045]), np.array([0.075, 0.120, 0.070])),
                 "torso"),
            Prim(f"deltoid.{side}", "ell", (sh + np.array([sx * 0.012, -0.012, 0.004]),
                                            np.array([0.060, 0.066, 0.060])), "torso"),
            Prim(f"upper_arm.{side}", "cone", (sh, J[f"forearm.{side}"], 0.050, 0.041), f"arm.{side}"),
            Prim(f"forearm.{side}", "cone", (J[f"forearm.{side}"], J[f"hand.{side}"], 0.042, 0.031), f"arm.{side}"),
        ]
    return prims


def body_sdf(p: np.ndarray) -> np.ndarray:
    prims = body_primitives()
    d = {pr.name: pr.dist(p) for pr in prims}
    torso = d["chest"]
    for name in ("upper_chest", "ribs", "abdomen", "hips", "pec.L", "pec.R", "lat.L", "lat.R"):
        torso = _smin(torso, d[name], 0.030)
    torso = _smin(torso, d["neck"], 0.035)
    for side in ("L", "R"):
        torso = _smin(torso, d[f"trap.{side}"], 0.040)
        torso = _smin(torso, d[f"deltoid.{side}"], 0.030)
    out = torso
    for side in ("L", "R"):
        arm = _smin(d[f"upper_arm.{side}"], d[f"forearm.{side}"], 0.012)
        out = _smin(out, arm, 0.016)
    return out


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def body_weights(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex (joints (N,4), weights (N,4)) from the primitives' soft-min blend."""
    prims = body_primitives()
    J = JOINT_POS
    D = np.stack([pr.dist(p) for pr in prims], 1)
    infl = np.exp(-(D - D.min(1, keepdims=True)) / 0.006)
    infl /= infl.sum(1, keepdims=True)
    W = np.zeros((len(p), len(JOINT_NAMES)))

    def add(name, w, joint):
        W[:, JOINT_INDEX[joint]] += w * infl[:, [pr.name for pr in prims].index(name)]

    y = p[:, 1]
    t_neck = _t_along(p, np.array([0, 1.405, -0.05]), np.array([0, 1.60, -0.022]))
    w_head = _smoothstep(0.55, 0.95, t_neck)
    w_chest = 1 - _smoothstep(0.05, 0.45, t_neck)
    add("neck", w_head, "head")
    add("neck", w_chest, "chest")
    add("neck", 1 - w_head - w_chest, "neck")
    for name in ("chest", "upper_chest", "pec.L", "pec.R", "lat.L", "lat.R"):
        add(name, 1.0, "chest")
    for name in ("ribs", "abdomen"):
        hi = _smoothstep(1.06, 1.24, y)
        add(name, hi, "chest")
        add(name, 1 - hi, "spine")
    add("hips", 1.0, "root")
    for side in ("L", "R"):
        t = _t_along(p, np.array([0, 1.505, -0.058]), J[f"upper_arm.{side}"])
        add(f"trap.{side}", 1 - 0.45 * t, "chest")
        add(f"trap.{side}", 0.45 * t, f"shoulder.{side}")
        add(f"deltoid.{side}", 0.45, f"shoulder.{side}")
        add(f"deltoid.{side}", 0.55, f"upper_arm.{side}")
        t = _t_along(p, J[f"upper_arm.{side}"], J[f"forearm.{side}"])
        top = 1 - _smoothstep(0.0, 0.18, t)
        elbow = _smoothstep(0.82, 1.0, t)
        add(f"upper_arm.{side}", 0.35 * top, f"shoulder.{side}")
        add(f"upper_arm.{side}", 1 - 0.35 * top - 0.5 * elbow, f"upper_arm.{side}")
        add(f"upper_arm.{side}", 0.5 * elbow, f"forearm.{side}")
        t = _t_along(p, J[f"forearm.{side}"], J[f"hand.{side}"])
        start = 1 - _smoothstep(0.0, 0.15, t)
        end = _smoothstep(0.88, 1.0, t)
        add(f"forearm.{side}", 0.5 * start, f"upper_arm.{side}")
        add(f"forearm.{side}", 1 - 0.5 * start - 0.6 * end, f"forearm.{side}")
        add(f"forearm.{side}", 0.6 * end, f"hand.{side}")
    order = np.argsort(-W, axis=1)[:, :4]
    w4 = np.take_along_axis(W, order, 1)
    w4 /= np.maximum(w4.sum(1, keepdims=True), 1e-9)
    return order.astype(np.uint16), w4.astype(np.float32)


def build_body(step: float = 0.0065):
    """Closed, manifold body mesh. Surface nets can pinch where two surfaces nearly touch (armpits);
    that depends on where the grid falls, so a shifted grid is tried until the mesh is closed."""
    for shift in (0.0, 0.33, 0.66, 0.17, 0.5, 0.83):
        lo = np.array([-0.58, 0.83, -0.18]) + shift * step
        hi = np.array([0.58, 1.66, 0.14]) + shift * step
        verts, tris = mt.surface_nets(body_sdf, lo, hi, step)
        if mt.is_closed(tris):
            break
    else:
        raise RuntimeError("body mesh is not closed for any grid offset")
    g = mt.sdf_gradient(body_sdf, verts, step * 0.25)
    normals = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    return verts, tris, normals


# --------------------------------------------------------------------------- emblem
EMBLEM_CENTER = np.array([0.0, 1.335])  # model x, y (m)
EMBLEM_SPAN = 0.36  # texture covers this many metres
EMBLEM_UNIT = 0.26  # one emblem unit in metres


def emblem_texture(size: int = 1024) -> np.ndarray:
    """RGBA texture: red spider (alpha = coverage) on transparent. Drawn procedurally."""
    ss = 4  # supersampling for clean edges
    s = size * ss
    canvas = np.zeros((s, s), np.uint8)

    def px(pt):
        u = 0.5 + pt[0] * EMBLEM_UNIT / EMBLEM_SPAN
        v = 0.5 + pt[1] * EMBLEM_UNIT / EMBLEM_SPAN
        return np.array([u * s, v * s])

    def ellipse(c, rx, ry, n=64):
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return np.stack([c[0] + rx * np.cos(t), c[1] + ry * np.sin(t)], 1)

    def fill(poly):
        cv2.fillPoly(canvas, [np.round(np.array([px(p) for p in poly])).astype(np.int32)], 255, cv2.LINE_8)

    # Body: small head, broad thorax, long abdomen tapering to a point (unit space, y down), with thin
    # gaps between the parts so the silhouette reads as a spider and not a blob.
    fill(ellipse((0.0, -0.335), 0.048, 0.044))
    fill(ellipse((0.0, -0.205), 0.080, 0.090))
    abd = ellipse((0.0, 0.040), 0.092, 0.165)
    low = abd[:, 1] > 0.06
    abd[low, 0] *= np.clip(1.0 - (abd[low, 1] - 0.06) / 0.17, 0.2, 1.0)
    fill(abd)
    legs = [  # right side (x > 0): root, knee, tip; mirrored for the left. Front legs reach up, back legs down.
        ((0.040, -0.255), (0.165, -0.445), (0.395, -0.520)),
        ((0.065, -0.215), (0.285, -0.325), (0.505, -0.265)),
        ((0.068, -0.170), (0.285, -0.110), (0.465, 0.055)),
        ((0.050, -0.130), (0.200, 0.045), (0.300, 0.330)),
    ]
    for leg in legs:
        for sx in (1.0, -1.0):
            pts = [np.array([sx * x, y]) for x, y in leg]
            widths = (0.030, 0.023, 0.004)
            for (a, wa), (b, wb) in zip(zip(pts[:-1], widths[:-1]), zip(pts[1:], widths[1:])):
                d = b - a
                nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
                fill([a + nrm * wa, b + nrm * wb, b - nrm * wb, a - nrm * wa])
            knee = pts[1]
            fill(ellipse(knee, widths[1], widths[1], 24))
    # Gaps between head / thorax / abdomen.
    for y in (-0.292, -0.112):
        cv2.line(canvas, tuple(np.round(px((-0.12, y))).astype(int)), tuple(np.round(px((0.12, y))).astype(int)), 0,
                 max(2, int(0.010 * EMBLEM_UNIT / EMBLEM_SPAN * s)))
    alpha = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 222, 18, 30
    rgba[..., 3] = alpha
    return rgba


def emblem_uv(p_model: np.ndarray, n_model: np.ndarray) -> np.ndarray:
    u = 0.5 + (p_model[:, 0] - EMBLEM_CENTER[0]) / EMBLEM_SPAN
    v = 0.5 - (p_model[:, 1] - EMBLEM_CENTER[1]) / EMBLEM_SPAN
    back = n_model[:, 2] < 0.05
    return np.where(back[:, None], -4.0, np.c_[u, v])


# --------------------------------------------------------------------------- materials
def _mat(name: str, color, extras: dict, texture: int | None = None) -> dict:
    pbr = {"baseColorFactor": [*color, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.8}
    if texture is not None:
        pbr["baseColorTexture"] = {"index": texture, "texCoord": 0}
    return {"name": name, "pbrMetallicRoughness": pbr, "extras": {"spiderverse": extras}}


SUIT_BLACK = (0.024, 0.024, 0.034)
MILES_RED = (0.86, 0.05, 0.10)


def build_prototype(path: str | Path, task_path: str | Path | None = None) -> Path:
    task_path = str(task_path or default_task_path())
    cv, _, ct = canonical_face(task_path)
    face = mask_face(cv, ct)
    head_cm, head_t, _ = build_head(face, ct)
    head_n = mt.vertex_normals(head_cm, head_t)

    b = GltfBuilder()
    b.json["asset"]["extras"] = {"spiderverse": {
        "prototype": True, "version": PROTOTYPE_VERSION,
        "note": "Procedural PROTOTYPE geometry (not a production character model).",
    }}
    emblem = emblem_texture()
    ok, png = cv2.imencode(".png", cv2.cvtColor(emblem, cv2.COLOR_RGBA2BGRA))
    tex = b.add_png_texture(png.tobytes(), "emblem")

    face_axis = canon_to_model(np.array([0.0, 3.7, 5.6])) - canon_to_model(HEAD_CENTER)
    m_mask = b.add_material(_mat("mask_fabric", SUIT_BLACK, {
        "shader": "toon", "role": "mask",
        "web": {"center": canon_to_model(HEAD_CENTER).tolist(), "axis": (face_axis / np.linalg.norm(face_axis)).tolist(),
                "ref": [0.0, 1.0, 0.0], "radials": 16, "ring_step": 0.255, "sag": 0.35, "width": 0.00105,
                "max_angle": 3.2},
    }))
    m_suit = b.add_material(_mat("suit_fabric", SUIT_BLACK, {
        "shader": "toon", "role": "suit", "decal": True,
        # Web radiating from the chest centre (under the emblem): straight strands up the neck and over the
        # shoulders, rings around them; red on the yoke, faint over the rest of the suit.
        "web": {"center": [0.0, 1.27, -0.03], "axis": [0.0, 1.0, 0.0], "ref": [0.0, 0.0, 1.0], "radials": 26,
                "ring_step": 0.105, "sag": 0.30, "width": 0.0013, "max_angle": 1.10, "faint": 0.35},
    }, texture=tex))
    m_lens = b.add_material(_mat("lens", (0.97, 0.97, 0.96), {"shader": "lens", "role": "lens",
                                                              "rim_color": list(MILES_RED)}))

    # Skeleton nodes.
    nodes = {}
    for name, parent, _ in JOINTS:
        pos = JOINT_POS[name]
        local = pos - (JOINT_POS[parent] if parent else 0.0)
        nodes[name] = b.add_node({"name": name, "translation": local.tolist()},
                                 parent=nodes[parent] if parent else None)
    from .face_geometry import CHIN, EYE_OUTER_L, EYE_OUTER_R, FOREHEAD, NOSE_TIP
    for label, lid in (("align_eye_outer_L", EYE_OUTER_L), ("align_eye_outer_R", EYE_OUTER_R),
                       ("align_chin", CHIN), ("align_nose_tip", NOSE_TIP), ("align_forehead", FOREHEAD)):
        b.add_node({"name": label, "translation": (cv[lid] / 100.0).tolist()}, parent=nodes["head"])
    inv_bind = np.stack([np.linalg.inv(_translation(JOINT_POS[n])) for n in JOINT_NAMES])
    skin = b.add_skin([nodes[n] for n in JOINT_NAMES], inv_bind, "upper_body", skeleton=nodes["root"])

    head_idx = JOINT_INDEX["head"]

    def rigid(n):
        j = np.zeros((n, 4), np.uint16)
        j[:, 0] = head_idx
        w = np.zeros((n, 4), np.float32)
        w[:, 0] = 1.0
        return j, w

    hp = canon_to_model(head_cm)
    j, w = rigid(len(hp))
    prim = b.primitive(hp, head_n, head_t, m_mask, uv0=np.zeros((len(hp), 2)), joints=j, weights=w)
    b.add_node({"name": "head_mask", "mesh": b.add_mesh("head_mask", [prim]), "skin": skin})

    for side, sign in (("L", 1.0), ("R", -1.0)):
        pos, nrm, uv, tris, targets = build_lens(sign, head_cm, head_t, head_n)
        pm = canon_to_model(pos)
        j, w = rigid(len(pm))
        names = [f"lens{e.capitalize()}_{side}" for e in EXPRESSIONS]
        deltas = [(targets[e] - pos) / 100.0 for e in EXPRESSIONS]
        prim = b.primitive(pm, nrm, tris, m_lens, uv0=uv, joints=j, weights=w, targets=deltas)
        b.add_node({"name": f"lens_{side}", "mesh": b.add_mesh(f"lens_{side}", [prim], names), "skin": skin})

    bv, bt, bn = build_body()
    bj, bw = body_weights(bv)
    prim = b.primitive(bv, bn, bt, m_suit, uv0=emblem_uv(bv, bn), joints=bj, weights=bw)
    b.add_node({"name": "body_suit", "mesh": b.add_mesh("body_suit", [prim]), "skin": skin})
    return b.write(path)


def _translation(t) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = t
    return m
