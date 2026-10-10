"""Character layer: tracking -> rig -> 3D render -> integration with the stylized frame.

Runs in the stylizer process on the same frame the face, pose and person mask
were computed from, so the character is registered to that frame exactly.

    begin()   rig update + GPU draw submission (the GPU renders while the CPU
              runs the comic pass)
    erase()   at render resolution: where the real person shows around the
              character (hair, ears, loose clothing), paint a learned clean
              background plate instead, so nothing real leaks past the
              character's silhouette. Real hands are never erased.
    composite() at camera resolution: premultiplied "over" of the character,
              with real hands cut out of it (hands are in front of the body
              when framing the portal).

The background plate is learned from pixels the person mask marks as clear
background; areas never seen (directly behind the head) are filled by
push-pull interpolation from their surroundings. That is an approximation: a
static camera and some head movement give the best plate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import PROJECT_ROOT, CharacterConfig
from ..tracking.hands import HAND_CONNECTIONS
from .rig import CharacterRig, RigPose, RigSettings, VirtualCamera

HEAD_HEIGHT_CM = 23.0
MASK_CUT = (1.425, 0.078, 0.012)  # miles_mask mode: keep the suit only around the neck above the collar (model m)


def push_pull(img: np.ndarray, weight: np.ndarray, levels: int = 7) -> np.ndarray:
    """Fill low-weight pixels from their weighted surroundings (image pyramid push-pull)."""
    w = np.ascontiguousarray(weight, np.float32)
    cw = img.astype(np.float32) * w[..., None]
    pyr = [(cw, w)]
    for _ in range(levels):
        cw, w = cv2.pyrDown(cw), cv2.pyrDown(w)
        pyr.append((cw, w))
        if min(w.shape) < 4:
            break
    cw, w = pyr[-1]
    col = cw / np.maximum(w, 1e-6)[..., None]
    for cw, w in reversed(pyr[:-1]):
        up = cv2.pyrUp(col, dstsize=(w.shape[1], w.shape[0]))
        a = np.clip(w * 4.0, 0.0, 1.0)[..., None]
        col = a * (cw / np.maximum(w, 1e-6)[..., None]) + (1.0 - a) * up
    return np.clip(col, 0, 255).astype(np.uint8)


class CleanPlate:
    """Background learned from frames where the person mask says "background".

    Seen pixels keep full detail; unseen ones (behind the head) come from a
    push-pull fill computed at quarter resolution, which is blurry by nature.
    """

    def __init__(self, rate: float = 0.15, refill_every: int = 4) -> None:
        self.rate = rate
        self.refill_every = refill_every
        self.plate: np.ndarray | None = None
        self.seen: np.ndarray | None = None
        self.filled: np.ndarray | None = None
        self._n = 0

    def reset(self) -> None:
        self.plate = self.seen = self.filled = None
        self._n = 0

    def update(self, img: np.ndarray, person: np.ndarray) -> None:
        h, w = img.shape[:2]
        bg = np.clip((0.2 - person) * 5.0, 0.0, 1.0).astype(np.float32)
        bg = cv2.erode(bg, np.ones((5, 5), np.uint8))  # stay clear of the silhouette edge
        if self.plate is None or self.plate.shape[:2] != (h, w):
            self.plate = img.copy()
            self.seen = bg.copy()
            self.filled = None
        else:
            a = bg * self.rate
            self.plate = cv2.blendLinear(img, self.plate, a, 1.0 - a)
            self.seen = np.maximum(self.seen * 0.9995, bg)
        self._n += 1
        if self.filled is None or self._n % self.refill_every == 0:
            sw, sh = max(8, w // 4), max(8, h // 4)
            small = cv2.resize(self.plate, (sw, sh), interpolation=cv2.INTER_AREA)
            seen_small = cv2.resize(self.seen, (sw, sh), interpolation=cv2.INTER_AREA)
            holes = cv2.resize(push_pull(small, seen_small, levels=5), (w, h), interpolation=cv2.INTER_LINEAR)
            s = np.ascontiguousarray(self.seen)
            self.filled = cv2.blendLinear(self.plate, holes, s, 1.0 - s)

    def fill(self) -> np.ndarray | None:
        return self.filled


def grow(mask_u8: np.ndarray, radius: float) -> np.ndarray:
    """Binary dilation by a disc of `radius` px via a distance transform (fast for large radii)."""
    if radius <= 0.5:
        return mask_u8
    d = cv2.distanceTransform(cv2.compare(mask_u8, 0, cv2.CMP_EQ), cv2.DIST_L2, 3)
    return cv2.compare(d, float(radius), cv2.CMP_LE)


def hand_mask(hands: list[np.ndarray], size: tuple[int, int], scale: float, grow: float = 1.0) -> np.ndarray | None:
    """Soft mask of the hands (from 21 landmarks each), uint8, at `size`; coordinates are multiplied by `scale`."""
    if not hands:
        return None
    w, h = size
    m = np.zeros((h, w), np.uint8)
    for lm in hands:
        p = np.asarray(lm, np.float64) * scale
        palm = float(np.linalg.norm(p[9] - p[0]))
        if palm < 2:
            continue
        thick = max(2, int(round(palm * 0.30 * grow)))
        pts = np.round(p).astype(np.int32)
        cv2.fillPoly(m, [pts[[0, 1, 5, 9, 13, 17]]], 255, cv2.LINE_AA)
        for a, b in HAND_CONNECTIONS:
            cv2.line(m, tuple(pts[a]), tuple(pts[b]), 255, thick, cv2.LINE_AA)
        for i in (4, 8, 12, 16, 20):
            cv2.circle(m, tuple(pts[i]), thick // 2 + 1, 255, -1, cv2.LINE_AA)
    return m if m.any() else None


@dataclass
class CharacterResult:
    image: np.ndarray
    rendered: bool
    timings: dict[str, float] = field(default_factory=dict)
    debug: dict = field(default_factory=dict)


class CharacterLayer:
    def __init__(self, cfg: CharacterConfig, project_root: Path = PROJECT_ROOT) -> None:
        self.cfg = cfg
        self.root = project_root
        self.asset = None
        self.renderer = None
        self.rig: CharacterRig | None = None
        self.plate = CleanPlate()
        self.error = ""
        self._pose: RigPose | None = None
        self._rgba: np.ndarray | None = None
        self._out_size: tuple[int, int] | None = None
        self._roi: tuple[int, int, int, int] | None = None
        self._hands: list[np.ndarray] = []
        self.timings: dict[str, float] = {}
        self.debug: dict = {}

    # setup -----------------------------------------------------------------------
    def asset_path(self) -> Path:
        p = Path(self.cfg.asset)
        return p if p.is_absolute() else self.root / p

    def ensure_loaded(self) -> None:
        if self.renderer is not None:
            return
        from .asset import load_character
        from .renderer import CharacterRenderer

        path = self.asset_path()
        if not path.exists() and path.name == "miles_prototype.glb":
            from .prototype import build_prototype

            build_prototype(path)
        t0 = time.perf_counter()
        self.asset = load_character(path)
        self.renderer = CharacterRenderer(self.asset, samples=max(1, self.cfg.msaa))
        self.rig = CharacterRig(self.asset, self._rig_settings())
        self._apply_style()
        self.timings["load_ms"] = (time.perf_counter() - t0) * 1000

    def _rig_settings(self) -> RigSettings:
        c = self.cfg
        return RigSettings(head_scale=c.head_scale, head_offset_cm=tuple(c.head_offset_cm),
                           head_rotation_offset_deg=tuple(c.head_rotation_offset_deg), auto_refine=c.auto_refine,
                           body_scale=c.body_scale, shoulder_depth_cm=c.shoulder_depth_cm,
                           head_min_cutoff=c.head_min_cutoff, head_beta=c.head_beta, rotation_beta=c.rotation_beta,
                           body_min_cutoff=c.body_min_cutoff, body_beta=c.body_beta,
                           expression_gain=c.expression_gain)

    def _apply_style(self) -> None:
        st = self.renderer.style
        st.halftone_strength = self.cfg.halftone_strength
        st.rim_strength = self.cfg.rim_strength
        st.key_dir = tuple(self.cfg.key_light)

    def update_config(self, cfg: CharacterConfig) -> None:
        reload = cfg.asset != self.cfg.asset or cfg.msaa != self.cfg.msaa
        self.cfg = cfg
        if reload and self.renderer is not None:
            self.renderer.release()
            self.renderer = None
            return
        if self.rig is not None:
            s = self._rig_settings()
            smoothing = (s.head_min_cutoff, s.head_beta, s.rotation_beta, s.body_min_cutoff, s.body_beta)
            old = self.rig.s
            self.rig.s = s
            self.rig.expr.gain = s.expression_gain
            if smoothing != (old.head_min_cutoff, old.head_beta, old.rotation_beta, old.body_min_cutoff,
                             old.body_beta):
                self.rig._filters_init()
        if self.renderer is not None:
            self._apply_style()

    def reset(self) -> None:
        if self.rig is not None:
            self.rig.reset()
        self.plate.reset()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.release()
            self.renderer = None

    # per frame -------------------------------------------------------------------
    def begin(self, sem, analysis_size: tuple[int, int], out_size: tuple[int, int], t: float,
              hands: list[np.ndarray] | None = None, mode: str | None = None) -> bool:
        """Rig + submit the GPU render. Returns False when there is nothing to draw (no face)."""
        mode = mode or self.cfg.mode
        self._rgba = None
        self._pose = None
        self._hands = hands or []
        self.timings = {}
        if self.renderer is None:
            try:
                self.ensure_loaded()
            except Exception as exc:  # noqa: BLE001 - reported in the HUD, the stylized frame still shows
                self.error = f"character load failed: {type(exc).__name__}: {exc}"[:200]
                return False
        t0 = time.perf_counter()
        pose = self.rig.update(sem.face if sem is not None else None, sem.pose if sem is not None else None,
                               analysis_size, t)
        t1 = time.perf_counter()
        self.timings["rig_ms"] = (t1 - t0) * 1000
        if pose is None:
            self.debug = {"state": "no face"}
            return False
        from .renderer import RenderFrame

        rs = self.cfg.render_scale
        rsize = (max(16, int(round(out_size[0] * rs))), max(16, int(round(out_size[1] * rs))))
        cam = pose.camera.with_size(rsize)
        head = pose.head_matrix
        depth = max(5.0, -float(head[2, 3]))
        head_px = cam.focal * HEAD_HEIGHT_CM / depth * cam.refine_scale
        rel = float(np.clip(head_px / (0.42 * rsize[1]), 0.55, 1.8))
        anchor = cam.project(head[:3, 3])
        roles = ("mask", "lens", "suit", "other")
        cut = MASK_CUT if mode == "miles_mask" else None
        axes = []
        debug3d = mode == "debug_3d"
        if debug3d:
            o = head[:3, 3]
            for k, color in ((0, (1.0, 0.2, 0.2)), (1, (0.2, 1.0, 0.2)), (2, (0.3, 0.5, 1.0))):
                axes.append((o, o + head[:3, k] * 10.0, color))
        roi = self.renderer.screen_roi(pose.joint_world, cam, rsize, roles if cut is None else ("mask", "lens"),
                                       margin_px=8.0 + self.cfg.outline_px * 3)
        if cut is not None and roi is not None:  # mask mode: the neck reaches a little below the head sphere
            roi = (roi[0], roi[1], roi[2], min(rsize[1], roi[3] + int(0.35 * (roi[3] - roi[1]))))
        frame = RenderFrame(rsize, cam.projection(), pose.joint_world, pose.morph_weights, roles,
                            outline_px=self.cfg.outline_px * rel * rsize[1] / 720.0,
                            halftone_anchor=(float(anchor[0]), float(anchor[1])),
                            halftone_cell=self.cfg.halftone_cell_px * rel * rsize[1] / 720.0,
                            cut=cut, wireframe=debug3d, axes=axes, roi=roi)
        self._roi = roi
        self.renderer.submit(frame)
        self.timings["submit_ms"] = (time.perf_counter() - t1) * 1000
        self._pose = pose
        self._out_size = out_size
        self._render_size = rsize
        self._mode = mode
        self.debug = self._debug_info(pose, analysis_size, out_size, sem, debug3d)
        return True

    def _read(self) -> np.ndarray | None:
        if self._pose is None:
            return None
        if self._rgba is None:
            t0 = time.perf_counter()
            rgba = self.renderer.read(self._roi) if self._roi is not None else \
                np.zeros((self._render_size[1], self._render_size[0], 4), np.uint8)
            if self._render_size != self._out_size:
                rgba = cv2.resize(rgba, self._out_size, interpolation=cv2.INTER_LINEAR)
            self._rgba = rgba
            self.timings["read_ms"] = (time.perf_counter() - t0) * 1000
        return self._rgba

    def erase(self, env: np.ndarray, person: np.ndarray | None, face) -> np.ndarray:
        """Replace real person pixels around the character with the background plate (render resolution)."""
        if person is None:
            return env
        rgba = self._read()  # waits for the GPU (timed separately as read_ms)
        t0 = time.perf_counter()
        self.plate.update(env, person)
        if rgba is None or not self.cfg.erase_real_head or self._mode == "debug_3d":
            self.timings["erase_ms"] = (time.perf_counter() - t0) * 1000
            return env
        h, w = env.shape[:2]
        mw, mh = w // 2, h // 2  # region maths at half resolution, the edge is feathered anyway
        alpha = cv2.resize(rgba[..., 3], (mw, mh), interpolation=cv2.INTER_AREA)
        person_s = cv2.resize(person, (mw, mh), interpolation=cv2.INTER_AREA)
        region = np.zeros((mh, mw), np.uint8)
        if face is not None:
            from ..costume.regions import head_mask
            from ..tracking.face import FaceObservation

            small_face = FaceObservation(face.landmarks * 0.5)
            band = head_mask(small_face, person_s, 1.15)
            region = grow((band > 0.15).astype(np.uint8) * 255, 0.025 * mh)
        if self._mode != "miles_mask":
            region = cv2.max(region, grow(cv2.compare(alpha, 8, cv2.CMP_GT), self.cfg.erase_margin * mh))
        # The person's soft edge belongs to the person: include it so no outline ghost remains.
        region = cv2.min(region, grow(cv2.compare(person_s, 0.08, cv2.CMP_GT), 2))
        hm = hand_mask(self._hands, (mw, mh), mw / float(self._out_size[0]), grow=1.5)
        if hm is not None:
            region = cv2.subtract(region, grow(hm, 2))
        region = cv2.resize(region, (w, h), interpolation=cv2.INTER_LINEAR)
        fill = self.plate.fill()
        if fill is not None and region.any():
            a = cv2.GaussianBlur(region, (0, 0), 1.5).astype(np.float32) * (1.0 / 255.0)
            if self._pose.opacity < 0.999:  # fading: the real head comes back as the character goes
                a *= self._pose.opacity
            env = cv2.blendLinear(fill, env, a, 1.0 - a)
        self.timings["erase_ms"] = (time.perf_counter() - t0) * 1000
        return env

    def composite(self, env: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        """Premultiplied character over `env` (camera resolution, BGR)."""
        rgba = self._read()
        if rgba is None:
            return env
        opacity *= self._pose.opacity
        t0 = time.perf_counter()
        alpha = rgba[..., 3]
        x, y, bw, bh = cv2.boundingRect(alpha)
        if bw == 0 or bh == 0:
            return env
        sl = (slice(y, y + bh), slice(x, x + bw))
        a = alpha[sl]
        col = cv2.cvtColor(rgba[sl], cv2.COLOR_RGBA2BGR)
        if self.cfg.hand_cutout and self._hands:
            hm = hand_mask(self._hands, env.shape[1::-1], 1.0)
            if hm is not None:
                keep = cv2.subtract(255, hm[sl])
                a = cv2.multiply(a, keep, scale=1.0 / 255.0)
                col = cv2.multiply(col, cv2.merge([keep, keep, keep]), scale=1.0 / 255.0)
        if opacity < 0.999:
            a = cv2.multiply(a, np.full_like(a, int(round(opacity * 255))), scale=1.0 / 255.0)
            col = cv2.multiply(col, np.full_like(col, int(round(opacity * 255))), scale=1.0 / 255.0)
        out = env.copy()
        inv = cv2.subtract(255, a)
        out[sl] = cv2.add(cv2.multiply(env[sl], cv2.merge([inv, inv, inv]), scale=1.0 / 255.0), col)
        self.timings["composite_ms"] = (time.perf_counter() - t0) * 1000
        return out

    def coverage(self) -> np.ndarray | None:
        rgba = self._read()
        return None if rgba is None else rgba[..., 3]

    # debug -----------------------------------------------------------------------
    def _debug_info(self, pose: RigPose, analysis_size, out_size, sem, full: bool) -> dict:
        s = out_size[0] / float(analysis_size[0])
        cam = pose.camera.with_size(out_size)
        d = {k: (tuple(float(x) for x in v) if isinstance(v, tuple) else v)
             for k, v in pose.debug.items() if k != "shoulders_px"}
        d.update(state="tracking", body=pose.body_source, body_scale=pose.body_scale,
                 expression={k: round(float(v), 2) for k, v in pose.morph_weights.items() if v > 0.05})
        if full:
            head = pose.head_matrix
            rig = self.rig
            pts = (head @ np.c_[rig.canon[rig.rigid_ids], np.ones(len(rig.rigid_ids))].T).T[:, :3]
            d["aligned_px"] = cam.project(pts).astype(np.float32)
            if sem is not None and sem.face is not None:
                d["landmarks_px"] = (sem.face.landmarks * s).astype(np.float32)
                d["rigid_px"] = (sem.face.landmarks[rig.rigid_ids] * s).astype(np.float32)
            if sem is not None and sem.pose is not None:
                d["pose_px"] = (sem.pose.landmarks * s).astype(np.float32)
                d["pose_vis"] = sem.pose.visibility.astype(np.float32)
            joints = pose.joint_world[:, :3, 3]
            d["joints_px"] = cam.project(joints).astype(np.float32)
            d["joint_parents"] = rig.sk.parents.astype(np.int32)
            o = head[:3, 3]
            d["axes_px"] = cam.project(np.stack([o, o + head[:3, 0] * 10, o + head[:3, 1] * 10,
                                                 o + head[:3, 2] * 10])).astype(np.float32)
        return d
