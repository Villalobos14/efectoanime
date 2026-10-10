"""Body anchor for costume graphics: where the chest is, which way the torso faces.

Everything the costume draws (emblem, web pattern) lives in this body frame,
so it moves, scales and rotates with the person instead of being pasted at a
fixed screen position. Pose shoulders are the primary source; when the
shoulders are not visible the frame is extrapolated from face geometry.
All parameters are One-Euro smoothed so the graphics stay steady.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..semantic.analyzer import L_SHOULDER, R_SHOULDER, PoseObservation
from ..tracking.face import FaceObservation
from ..tracking.smoothing import OneEuroFilter

FOREHEAD, CHIN, CHEEK_A, CHEEK_B = 10, 152, 234, 454


@dataclass
class BodyFrame:
    shoulder_mid: np.ndarray  # px
    u: np.ndarray  # unit vector along the shoulders, screen-left to screen-right
    v: np.ndarray  # unit vector down the torso
    scale: float  # shoulder width in px
    squash: float  # horizontal foreshortening from torso yaw (1 = facing the camera)
    source: str  # "pose" or "face"

    @property
    def chest(self) -> np.ndarray:
        return self.shoulder_mid + self.v * (0.42 * self.scale)

    def to_image(self, pts: np.ndarray, origin: np.ndarray | None = None, size: float | None = None) -> np.ndarray:
        """Map body-space points (x right, y down, in units of `size`) to image pixels."""
        origin = self.chest if origin is None else origin
        size = self.scale if size is None else size
        p = np.asarray(pts, np.float64)
        return origin + p[..., 0:1] * (self.squash * size) * self.u + p[..., 1:2] * size * self.v


def frame_from_pose(pose: PoseObservation) -> BodyFrame | None:
    if not (pose.visible(L_SHOULDER) and pose.visible(R_SHOULDER)):
        return None
    a, b = pose.landmarks[L_SHOULDER], pose.landmarks[R_SHOULDER]
    p1, p2 = (a, b) if a[0] <= b[0] else (b, a)
    width = float(np.linalg.norm(p2 - p1))
    if width < 10:
        return None
    u = (p2 - p1) / width
    v = np.array([-u[1], u[0]])
    squash = 1.0
    if pose.world is not None:
        d = pose.world[L_SHOULDER] - pose.world[R_SHOULDER]
        yaw = math.atan2(abs(d[2]), max(abs(d[0]), 1e-6))
        squash = float(np.clip(math.cos(yaw), 0.5, 1.0))
    # The 2D shoulder distance already includes foreshortening; recover the frontal width.
    return BodyFrame((p1 + p2) / 2.0, u, v, width / squash, squash, "pose")


def frame_from_face(face: FaceObservation) -> BodyFrame:
    lm = face.landmarks
    up = lm[FOREHEAD] - lm[CHIN]
    fh = float(np.linalg.norm(up))
    up = up / max(fh, 1e-6)
    v = -up
    u = np.array([v[1], -v[0]])  # perpendicular to "down", pointing screen-right
    if u[0] < 0:
        u = -u
    fw = float(np.linalg.norm(lm[CHEEK_B] - lm[CHEEK_A]))
    mid = lm[CHIN] + v * (0.75 * fh)
    return BodyFrame(mid, u, v, 2.4 * fw, 1.0, "face")


class BodyFrameEstimator:
    def __init__(self, min_cutoff: float = 1.2, beta: float = 4.0) -> None:
        self._filters = {k: OneEuroFilter(min_cutoff, beta) for k in ("mid", "angle", "scale", "squash")}
        self.last: BodyFrame | None = None
        self._last_t = -1.0

    def reset(self) -> None:
        for f in self._filters.values():
            f.reset()
        self.last = None

    def update(self, pose: PoseObservation | None, face: FaceObservation | None, t: float,
               frame_size: tuple[int, int]) -> BodyFrame | None:
        raw = frame_from_pose(pose) if pose is not None else None
        if raw is None and face is not None:
            raw = frame_from_face(face)
        if raw is None:
            # Hold the last frame briefly so the emblem does not blink during a detection gap.
            return self.last if self.last is not None and t - self._last_t < 0.4 else None
        diag = math.hypot(*frame_size)
        mid = self._filters["mid"](raw.shoulder_mid / diag, t) * diag
        angle = float(self._filters["angle"](np.array([math.atan2(raw.u[1], raw.u[0])]), t)[0])
        scale = float(self._filters["scale"](np.array([raw.scale / diag]), t)[0]) * diag
        squash = float(self._filters["squash"](np.array([raw.squash]), t)[0])
        u = np.array([math.cos(angle), math.sin(angle)])
        self.last = BodyFrame(mid, u, np.array([-u[1], u[0]]), scale, squash, raw.source)
        self._last_t = t
        return self.last
