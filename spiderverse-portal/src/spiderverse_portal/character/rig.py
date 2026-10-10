"""Tracking -> character rig: joint world matrices (model -> camera space, cm) and lens morph weights.

Head: MediaPipe's facial transformation matrix maps the canonical face into a
metric camera space; the asset's head joint is defined in canonical-face
coordinates, so the head joint's world matrix is that matrix (plus user
calibration: scale, offset, rotation). Translation and rotation are One-Euro
smoothed separately (depth is the noisiest axis and gets its own filter). An
optional 2D refinement fits a scale + shift between the projected rigid
canonical points and the detected landmarks, which removes the few pixels of
residual misregistration the rigid fit leaves.

Body: from MediaPipe Pose. Shoulders, elbows and wrists are 2D detections;
their depth comes from the head's metric depth plus the pose model's relative
world-space z. Single-camera 3D is an estimate: the torso is placed so its
shoulder joints project onto the detected shoulders, scaled to the measured
shoulder width (clamped), oriented by the shoulder line and the pose z
difference (torso yaw). Arm bones keep their length and rotate toward the
detected elbow / wrist. Low-visibility arms hang at rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..tracking.smoothing import OneEuroFilter
from .asset import CharacterAsset
from .expressions import ExpressionMapper
from .face_geometry import RIGID_POINTS, canonical_face, default_task_path, perspective, similarity_2d

L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24


# --------------------------------------------------------------------------- math helpers
def quat_from_matrix(r: np.ndarray) -> np.ndarray:
    """(w, x, y, z) from a rotation matrix."""
    m = r
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, np.float64)
    return q / np.linalg.norm(q)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, np.float64) / max(float(np.linalg.norm(q)), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking direction a to direction b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.9999:
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def euler_matrix(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """R = Rz(roll) Ry(yaw) Rx(pitch): the inverse of `euler_from_rotation`."""
    p, y, r = map(math.radians, (pitch_deg, yaw_deg, roll_deg))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def euler_from_rotation(r: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees for R = Rz(roll) Ry(yaw) Rx(pitch), as tracking.face._euler_from_matrix."""
    sy = math.hypot(r[0, 0], r[1, 0])
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def trs(t=(0, 0, 0), r: np.ndarray | None = None, s: float = 1.0) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = (np.eye(3) if r is None else r) * s
    m[:3, 3] = t
    return m


# --------------------------------------------------------------------------- camera
@dataclass
class VirtualCamera:
    """MediaPipe's face-geometry camera (vertical FOV 63 deg), plus a small 2D refinement in NDC."""

    size: tuple[int, int]
    vfov_deg: float = 63.0
    refine_scale: float = 1.0
    refine_shift: tuple[float, float] = (0.0, 0.0)  # NDC

    @property
    def focal(self) -> float:
        return (self.size[1] / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)

    def with_size(self, size: tuple[int, int]) -> "VirtualCamera":
        return VirtualCamera(size, self.vfov_deg, self.refine_scale, self.refine_shift)

    def projection(self) -> np.ndarray:
        a = np.eye(4)
        a[0, 0] = a[1, 1] = self.refine_scale
        a[0, 3], a[1, 3] = self.refine_shift
        return a @ perspective(self.size, self.vfov_deg)

    def project(self, p: np.ndarray) -> np.ndarray:
        w, h = self.size
        p = np.asarray(p, np.float64)
        z = np.maximum(-p[..., 2], 1e-6)
        nx = self.focal * p[..., 0] / z / (w / 2.0)
        ny = self.focal * p[..., 1] / z / (h / 2.0)
        nx = nx * self.refine_scale + self.refine_shift[0]
        ny = ny * self.refine_scale + self.refine_shift[1]
        return np.stack([(nx + 1) * w / 2.0, (1 - ny) * h / 2.0], -1)

    def unproject(self, px: np.ndarray, depth: np.ndarray | float) -> np.ndarray:
        w, h = self.size
        px = np.asarray(px, np.float64)
        nx = (px[..., 0] / (w / 2.0) - 1 - self.refine_shift[0]) / self.refine_scale
        ny = (1 - px[..., 1] / (h / 2.0) - self.refine_shift[1]) / self.refine_scale
        d = np.asarray(depth, np.float64) * np.ones_like(nx)
        return np.stack([nx * (w / 2.0) * d / self.focal, ny * (h / 2.0) * d / self.focal, -d], -1)


# --------------------------------------------------------------------------- rig
@dataclass
class RigSettings:
    head_scale: float = 1.0
    head_offset_cm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    head_rotation_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)  # pitch, yaw, roll
    auto_refine: bool = True
    body_scale: float = 1.0
    shoulder_depth_cm: float = 3.5
    head_min_cutoff: float = 1.6
    head_beta: float = 0.35
    depth_min_cutoff: float = 0.6
    rotation_beta: float = 4.0
    body_min_cutoff: float = 1.0
    body_beta: float = 0.3
    expression_gain: float = 1.0
    hold_s: float = 0.35  # keep the last pose this long when tracking drops
    fade_out_s: float = 0.30  # then fade the character out (the real head returns smoothly)
    fade_in_s: float = 0.12  # fade in when the face is found again


@dataclass
class RigPose:
    joint_world: np.ndarray
    morph_weights: dict[str, float]
    camera: VirtualCamera  # analysis-size camera with the 2D refinement
    head_matrix: np.ndarray  # canonical (cm) -> camera, calibrated and smoothed
    head_visible: bool
    body_source: str  # pose | face | none
    body_scale: float
    debug: dict = field(default_factory=dict)
    opacity: float = 1.0  # < 1 while fading in after a reacquisition or out after losing the face


class CharacterRig:
    def __init__(self, asset: CharacterAsset, settings: RigSettings | None = None, task_path=None) -> None:
        self.asset = asset
        self.s = settings or RigSettings()
        sk = asset.skeleton
        self.sk = sk
        self.rest = sk.rest_world
        self.rest_local = sk.rest_local()
        self.canon = canonical_face(str(task_path or default_task_path()))[0]
        self.rigid_ids = np.array(RIGID_POINTS)
        self.expr = ExpressionMapper(self.s.expression_gain)
        self.order = self._topological()
        self._filters_init()
        self.last: RigPose | None = None
        self._last_face_t = -1e9
        self._appear_t = -1e9
        self._last_pose_t = -1e9
        self._last_body: dict | None = None

    def _topological(self) -> list[int]:
        order, seen = [], set()

        def visit(j):
            if j in seen:
                return
            p = int(self.sk.parents[j])
            if p >= 0:
                visit(p)
            seen.add(j)
            order.append(j)

        for j in range(len(self.sk.names)):
            visit(j)
        return order

    def _filters_init(self) -> None:
        s = self.s
        self.f_xy = OneEuroFilter(s.head_min_cutoff, s.head_beta)
        self.f_z = OneEuroFilter(s.depth_min_cutoff, s.head_beta * 0.3)
        self.f_q = OneEuroFilter(s.head_min_cutoff, s.rotation_beta)
        self.f_refine = OneEuroFilter(0.8, 0.0)
        self.f_body = {}
        self._q_prev: np.ndarray | None = None

    def reset(self) -> None:
        self._filters_init()
        self.expr.reset()
        self.last = None
        self._last_body = None

    def _bf(self, key: str) -> OneEuroFilter:
        if key not in self.f_body:
            self.f_body[key] = OneEuroFilter(self.s.body_min_cutoff, self.s.body_beta)
        return self.f_body[key]

    # head ------------------------------------------------------------------------
    def _calibration(self) -> np.ndarray:
        s = self.s
        return trs(s.head_offset_cm, euler_matrix(*s.head_rotation_offset_deg), s.head_scale)

    def _smooth_head(self, m: np.ndarray, t: float) -> np.ndarray:
        scale = float(np.cbrt(abs(np.linalg.det(m[:3, :3]))))  # calibration scale (the tracker's matrix is rigid)
        u, _, vt = np.linalg.svd(m[:3, :3])
        r = u @ vt
        q = quat_from_matrix(r)
        if self._q_prev is not None and np.dot(q, self._q_prev) < 0:
            q = -q  # keep the quaternion on one hemisphere so the filter does not average q and -q
        qs = self.f_q(q, t)
        qs /= np.linalg.norm(qs)
        self._q_prev = qs
        xy = self.f_xy(m[:2, 3], t)
        z = self.f_z(m[2:3, 3], t)
        return trs(np.r_[xy, z], matrix_from_quat(qs), scale)

    def _refine(self, head: np.ndarray, landmarks: np.ndarray, cam: VirtualCamera, t: float) -> VirtualCamera:
        base = VirtualCamera(cam.size, cam.vfov_deg)
        pts = (head @ np.c_[self.canon[self.rigid_ids], np.ones(len(self.rigid_ids))].T).T[:, :3]
        proj = base.project(pts)
        w, h = cam.size
        to_ndc = lambda p: np.stack([p[:, 0] / (w / 2.0) - 1, 1 - p[:, 1] / (h / 2.0)], 1)  # noqa: E731
        s, tt = similarity_2d(to_ndc(proj), to_ndc(landmarks[self.rigid_ids]))
        v = self.f_refine(np.array([s, tt[0], tt[1]]), t)
        s = float(np.clip(v[0], 0.85, 1.15))
        shift = (float(np.clip(v[1], -0.12, 0.12)), float(np.clip(v[2], -0.12, 0.12)))
        return VirtualCamera(cam.size, cam.vfov_deg, s, shift)

    # body ------------------------------------------------------------------------
    def _rest_point(self, role: str) -> np.ndarray:
        j = self.sk.index(role)
        return self.rest[j][:3, 3] if j >= 0 else np.zeros(3)

    def _body_from_pose(self, pose, head_cam: np.ndarray, cam: VirtualCamera, t: float, vis_th: float = 0.5):
        if pose is None or self.sk.index("upper_arm.L") < 0 or self.sk.index("upper_arm.R") < 0:
            return None
        if pose.visibility[L_SHOULDER] < vis_th or pose.visibility[R_SHOULDER] < vis_th:
            return None
        z_head = -float(head_cam[2])
        z0 = z_head + self.s.shoulder_depth_cm
        wz = pose.world[:, 2] * 100.0 if pose.world is not None else np.zeros(len(pose.landmarks))
        mid_w = 0.5 * (wz[L_SHOULDER] + wz[R_SHOULDER])

        def point(idx, key):
            depth = z0 + (wz[idx] - mid_w)
            p = cam.unproject(pose.landmarks[idx], depth)
            return self._bf(key)(p, t)

        sl, sr = point(L_SHOULDER, "sl"), point(R_SHOULDER, "sr")
        rest_l, rest_r = self._rest_point("upper_arm.L"), self._rest_point("upper_arm.R")
        rest_w = float(np.linalg.norm(rest_l - rest_r)) * 100.0
        width = float(np.linalg.norm(sl - sr))
        scale = float(np.clip(width / max(rest_w, 1e-6), 0.88, 1.15)) * self.s.body_scale  # keep proportions
        scale = float(self._bf("scale")(np.array([scale]), t)[0])
        x = sl - sr
        x /= np.linalg.norm(x)
        up = np.array([0.0, 1.0, 0.0])
        if pose.visibility[L_HIP] > 0.6 and pose.visibility[R_HIP] > 0.6:
            hip = 0.5 * (cam.unproject(pose.landmarks[L_HIP], z0) + cam.unproject(pose.landmarks[R_HIP], z0))
            v = 0.5 * (sl + sr) - hip
            if np.linalg.norm(v) > 1e-3:
                up = 0.5 * up + 0.5 * v / np.linalg.norm(v)
        y = up - np.dot(up, x) * x
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        r_chest = np.stack([x, y, z], 1)
        mid_cam = 0.5 * (sl + sr)
        rest_mid = 0.5 * (rest_l + rest_r)
        g_chest = trs(mid_cam) @ trs(r=r_chest, s=100.0 * scale) @ trs(-rest_mid)
        arms = {}
        for side, (sh, el, wr) in (("L", (L_SHOULDER, L_ELBOW, L_WRIST)), ("R", (R_SHOULDER, R_ELBOW, R_WRIST))):
            arms[side] = self._arm(side, pose, el, wr, g_chest, r_chest, scale, z0, wz, mid_w, cam, t, vis_th)
        return {"g_chest": g_chest, "r_chest": r_chest, "scale": scale, "arms": arms,
                "shoulders_px": cam.project(np.stack([sl, sr]))}

    def _arm(self, side, pose, el, wr, g_chest, r_chest, scale, z0, wz, mid_w, cam, t, vis_th):
        j_up, j_fore = self.sk.index(f"upper_arm.{side}"), self.sk.index(f"forearm.{side}")
        if j_up < 0 or j_fore < 0:
            return None
        p_sh = self.rest[j_up][:3, 3]
        p_el = self.rest[j_fore][:3, 3]
        j_hand = self.sk.index(f"hand.{side}")
        p_wr = self.rest[j_hand][:3, 3] if j_hand >= 0 else p_el + (p_el - p_sh)
        sh_cam = (g_chest @ np.r_[p_sh, 1])[:3]
        d0 = r_chest @ ((p_el - p_sh) / np.linalg.norm(p_el - p_sh))
        sx = 1.0 if side == "L" else -1.0
        hang = r_chest @ np.array([sx * 0.22, -0.97, 0.06])
        if pose.visibility[el] >= vis_th:
            e = cam.unproject(pose.landmarks[el], z0 + (wz[el] - mid_w))
            d = e - sh_cam
            d = d / max(np.linalg.norm(d), 1e-6)
        else:
            d = hang
        d = self._bf(f"d_up_{side}")(d, t)
        d /= max(np.linalg.norm(d), 1e-6)
        r_up = rotation_between(d0, d) @ r_chest
        g_up = trs(sh_cam) @ trs(r=r_up, s=100.0 * scale) @ trs(-p_sh)
        el_cam = (g_up @ np.r_[p_el, 1])[:3]
        d1 = r_up @ ((p_wr - p_el) / np.linalg.norm(p_wr - p_el))
        if pose.visibility[wr] >= vis_th and pose.visibility[el] >= vis_th:
            w = cam.unproject(pose.landmarks[wr], z0 + (wz[wr] - mid_w))
            d2 = w - el_cam
            d2 = d2 / max(np.linalg.norm(d2), 1e-6)
        else:
            d2 = d1
        d2 = self._bf(f"d_fore_{side}")(d2, t)
        d2 /= max(np.linalg.norm(d2), 1e-6)
        r_fore = rotation_between(d1, d2) @ r_up
        g_fore = trs(el_cam) @ trs(r=r_fore, s=100.0 * scale) @ trs(-p_el)
        return {"g_up": g_up, "g_fore": g_fore}

    def _body_from_head(self, head_cam_m: np.ndarray) -> dict:
        """No pose: hang the rest-pose torso under the head, upright, following a little of the head yaw."""
        r_head = head_cam_m[:3, :3] / np.linalg.norm(head_cam_m[:3, 0])
        yaw, _, _ = euler_from_rotation(r_head)
        r = euler_matrix(0.0, 0.4 * yaw, 0.0)
        hj = self.sk.index("head")
        head_pos_cam = head_cam_m[:3, 3]
        head_rest = self.rest[hj][:3, 3]
        g = trs(head_pos_cam) @ trs(r=r, s=100.0) @ trs(-head_rest)
        return {"g_chest": g, "r_chest": r, "scale": 1.0, "arms": {}, "shoulders_px": None}

    # main --------------------------------------------------------------------------
    def update(self, face, pose, analysis_size: tuple[int, int], t: float) -> RigPose | None:
        cam = VirtualCamera(analysis_size)
        sk = self.sk
        hj = sk.index("head")
        matrix = getattr(face, "matrix", None) if face is not None else None
        if matrix is not None:
            if self.last is None:
                self._appear_t = t
                self._filters_init()  # a new appearance: do not glide in from where the face was lost
            self._last_face_t = t
            m = np.asarray(matrix, np.float64) @ self._calibration()
            head = self._smooth_head(m, t)
            if self.s.auto_refine:
                cam = self._refine(head, face.landmarks, cam, t)
            weights = self.expr.update(face.blendshapes, t)
            # Fade in from 1/3 on the first frame to full over fade_in_s.
            opacity = 1.0 if self.s.fade_in_s <= 0 else \
                float(min(1.0, 1.0 / 3.0 + (t - self._appear_t) / self.s.fade_in_s))
        elif self.last is not None and t - self._last_face_t < self.s.hold_s + self.s.fade_out_s:
            head = self.last.head_matrix
            cam = self.last.camera
            weights = self.last.morph_weights
            gone = t - self._last_face_t - self.s.hold_s
            opacity = 1.0 if gone <= 0 else float(np.clip(1.0 - gone / max(self.s.fade_out_s, 1e-6), 0.0, 1.0))
        else:
            self.last = None
            return None

        w_head = head @ np.linalg.inv(self.asset.head_from_canonical)  # head local -> camera
        body = self._body_from_pose(pose, w_head[:3, 3], cam, t)
        source = "pose"
        if body is not None:
            self._last_pose_t = t
            self._last_body = body
        elif self._last_body is not None and t - self._last_pose_t < self.s.hold_s:
            body, source = self._last_body, "pose (held)"
        else:
            body, source = self._body_from_head(w_head), "face"

        G = {}
        G[hj] = w_head @ np.linalg.inv(self.rest[hj])
        for role in ("chest", "spine", "root"):
            j = sk.index(role)
            if j >= 0:
                G[j] = body["g_chest"]
        for side in ("L", "R"):
            j = sk.index(f"shoulder.{side}")
            if j >= 0:
                G[j] = body["g_chest"]
            arm = body["arms"].get(side)
            ju, jf, jh = sk.index(f"upper_arm.{side}"), sk.index(f"forearm.{side}"), sk.index(f"hand.{side}")
            if arm is None:
                for j in (ju, jf, jh):
                    if j >= 0:
                        G[j] = body["g_chest"]
            else:
                if ju >= 0:
                    G[ju] = arm["g_up"]
                if jf >= 0:
                    G[jf] = arm["g_fore"]
                if jh >= 0:
                    G[jh] = arm["g_fore"]
        nj = sk.index("neck")
        if nj >= 0:
            G[nj] = self._neck(G.get(sk.index("chest"), body["g_chest"]), G[hj], nj)

        world = np.zeros((len(sk.names), 4, 4))
        for j in self.order:
            if j in G:
                world[j] = G[j] @ self.rest[j]
            else:  # joints the tracker does not drive (fingers, jaw, ...) follow their parent
                p = int(sk.parents[j])
                world[j] = world[p] @ self.rest_local[j] if p >= 0 else G[hj] @ self.rest[j]
        yaw, pitch, roll = euler_from_rotation(head[:3, :3])
        dbg = {"yaw": yaw, "pitch": pitch, "roll": roll, "depth_cm": -float(head[2, 3]),
               "refine": (cam.refine_scale, *cam.refine_shift), "shoulders_px": body.get("shoulders_px")}
        dbg["opacity"] = opacity
        self.last = RigPose(world, weights, cam, head, matrix is not None, source, float(body["scale"]), dbg, opacity)
        return self.last

    def _neck(self, g_chest: np.ndarray, g_head: np.ndarray, nj: int) -> np.ndarray:
        """Neck: anchored on the chest, rotation half-way between chest and head (less twist)."""
        def rot_scale(g):
            s = float(np.linalg.norm(g[:3, 0]))
            return g[:3, :3] / s, s

        rc, sc = rot_scale(g_chest)
        rh, sh = rot_scale(g_head)
        q = slerp(quat_from_matrix(rc), quat_from_matrix(rh), 0.5)
        r = matrix_from_quat(q)
        s = 0.5 * (sc + sh)
        p_rest = self.rest[nj][:3, 3]
        anchor = (g_chest @ np.r_[p_rest, 1])[:3]
        return trs(anchor) @ trs(r=r, s=s) @ trs(-p_rest)
