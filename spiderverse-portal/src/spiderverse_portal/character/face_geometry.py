"""MediaPipe face geometry: canonical face model and the virtual camera it is registered with.

The Face Landmarker returns a 4x4 *facial transformation matrix* that maps the
canonical face model (468 vertices, centimetres, X right, Y up, face looking
toward +Z) into a metric camera space (camera at the origin looking down -Z).
Projecting that space with MediaPipe's virtual perspective camera (vertical
FOV 63 degrees, principal point at the image centre) lands on the detected
landmarks. Measured on tests/data/astronaut.png: median error 5 px at 63
degrees versus 12 px at 60 and 10 px at 66.

So a 3D head built in canonical coordinates, transformed by that matrix and
rendered with the same camera, is registered with the face in the image,
independent of the real webcam's intrinsics.

The canonical mesh ships inside face_landmarker.task
(geometry_pipeline_metadata_landmarks.binarypb, a protobuf); it is parsed here
with a small wire-format reader so no protobuf schema is needed.
"""

from __future__ import annotations

import math
import struct
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

VERTICAL_FOV_DEG = 63.0
NEAR_CM = 2.0
FAR_CM = 2000.0

# Landmark ids used for alignment and lens placement (subject's left = +X in canonical space).
NOSE_TIP, NOSE_BRIDGE, FOREHEAD, CHIN = 4, 168, 10, 152
EYE_OUTER_R, EYE_INNER_R, EYE_OUTER_L, EYE_INNER_L = 33, 133, 263, 362
CHEEK_R, CHEEK_L = 234, 454
# Points that move little with expression and lie inside the face: forehead, brow ridge, nose bridge and
# tip, eye corners, cheekbones. Face-contour points are left out on purpose: they follow the visible
# silhouette, and on tests/data/astronaut.png they biased a 2D scale fit to 0.957 (inner points: 1.007).
RIGID_POINTS = (151, 9, 8, 168, 6, 197, 195, 5, 4, 33, 133, 263, 362, 116, 345, 123, 352, 46, 276, 70, 300)


def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def _fields(b: bytes):
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        num, wire = key >> 3, key & 7
        if wire == 0:
            v, i = _varint(b, i)
        elif wire == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        elif wire == 5:
            v, i = b[i:i + 4], i + 4
        elif wire == 1:
            v, i = b[i:i + 8], i + 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield num, wire, v


def parse_canonical_mesh(blob: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GeometryPipelineMetadata protobuf -> (vertices (468,3) cm, uvs (468,2), triangles (T,3))."""
    floats: list[float] = []
    indices: list[int] = []
    for num, _, v in _fields(blob):
        if num != 1:  # canonical_mesh
            continue
        for fnum, wire, fv in _fields(v):
            if fnum == 3:  # vertex_buffer (VERTEX_PT: x y z u v)
                if wire == 5:
                    floats.append(struct.unpack("<f", fv)[0])
                else:
                    floats.extend(np.frombuffer(fv, "<f4").tolist())
            elif fnum == 4:  # index_buffer
                if wire == 0:
                    indices.append(fv)
                else:
                    j = 0
                    while j < len(fv):
                        x, j = _varint(fv, j)
                        indices.append(x)
    vb = np.asarray(floats, np.float64).reshape(-1, 5)
    tris = np.asarray(indices, np.int64).reshape(-1, 3)
    return vb[:, :3], vb[:, 3:], tris


@lru_cache(maxsize=2)
def canonical_face(task_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with zipfile.ZipFile(task_path) as z:
        blob = z.read("geometry_pipeline_metadata_landmarks.binarypb")
    return parse_canonical_mesh(blob)


def focal_px(height: int | float, vfov_deg: float = VERTICAL_FOV_DEG) -> float:
    return (float(height) / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)


def project(points_cam: np.ndarray, size: tuple[int, int], vfov_deg: float = VERTICAL_FOV_DEG) -> np.ndarray:
    """Metric camera-space points (cm, camera looking down -Z) -> image pixels (x right, y down)."""
    w, h = size
    f = focal_px(h, vfov_deg)
    p = np.asarray(points_cam, np.float64)
    z = np.maximum(-p[..., 2], 1e-6)
    return np.stack([w / 2.0 + f * p[..., 0] / z, h / 2.0 - f * p[..., 1] / z], axis=-1)


def unproject(px: np.ndarray, depth_cm: np.ndarray | float, size: tuple[int, int],
              vfov_deg: float = VERTICAL_FOV_DEG) -> np.ndarray:
    """Image pixels at a given distance in front of the camera (positive cm) -> camera-space points."""
    w, h = size
    f = focal_px(h, vfov_deg)
    p = np.asarray(px, np.float64)
    d = np.asarray(depth_cm, np.float64)
    x = (p[..., 0] - w / 2.0) * d / f
    y = -(p[..., 1] - h / 2.0) * d / f
    return np.stack([x, y, -d * np.ones_like(x)], axis=-1)


def perspective(size: tuple[int, int], vfov_deg: float = VERTICAL_FOV_DEG, near: float = NEAR_CM,
                far: float = FAR_CM) -> np.ndarray:
    """OpenGL projection matrix (row-major) matching `project`."""
    w, h = size
    f = 1.0 / math.tan(math.radians(vfov_deg) / 2.0)
    aspect = w / float(h)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def similarity_2d(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Least-squares scale + translation (no rotation) mapping src points onto dst: dst ~ s * src + t."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    wts = np.ones(len(src)) if weights is None else np.asarray(weights, np.float64)
    wts = wts / max(wts.sum(), 1e-12)
    ms = (src * wts[:, None]).sum(0)
    md = (dst * wts[:, None]).sum(0)
    a, b = src - ms, dst - md
    var = float((wts[:, None] * a * a).sum())
    s = float((wts[:, None] * a * b).sum()) / max(var, 1e-12)
    return s, md - s * ms


def default_task_path() -> Path:
    from ..config import PROJECT_ROOT

    return PROJECT_ROOT / "models" / "face_landmarker.task"
