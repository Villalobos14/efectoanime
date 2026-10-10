"""Per-frame scene understanding for the costume layer: person mask, face, pose.

Runs inside the stylizer process on a worker thread. MediaPipe releases the
GIL while it computes, so this overlaps the Core ML inference on the main
thread instead of adding to it. Everything is computed on the same frame that
is being stylized, so masks and landmarks line up with the stylized pixels.

Measured on an M1 at 640x360: person segmentation 2.2 ms, face landmarks
~7 ms, pose (lite) ~10 ms. The multiclass segmenter (hair/face/clothes) was
measured at 128 ms per frame and is not used; head and torso are separated
geometrically from face landmarks instead (see costume/regions.py).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..tracking.face import FaceObservation, _euler_from_matrix

# MediaPipe pose landmark indices.
NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = 0, 11, 12, 13, 14, 15, 16, 23, 24


@dataclass
class PoseObservation:
    landmarks: np.ndarray  # (33, 2) pixels
    visibility: np.ndarray  # (33,)
    world: np.ndarray | None = None  # (33, 3) metres, hip-centred

    def visible(self, idx: int, threshold: float = 0.5) -> bool:
        return bool(self.visibility[idx] >= threshold)


@dataclass
class SemanticResult:
    person: np.ndarray | None = None  # float32 (H, W) in [0, 1], temporally smoothed
    face: FaceObservation | None = None
    pose: PoseObservation | None = None
    timings: dict[str, float] = field(default_factory=dict)
    error: str = ""


class SemanticAnalyzer:
    def __init__(self, models_dir: Path, aspect: float, mask_smoothing: float = 0.45) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._mp = mp
        seg_model = "selfie_segmenter_landscape.tflite" if aspect > 1.5 else "selfie_segmenter.tflite"
        self._segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(models_dir / seg_model)),
            running_mode=vision.RunningMode.VIDEO, output_category_mask=False, output_confidence_masks=True))
        self._face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models_dir / "face_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True,
            output_facial_transformation_matrixes=True))
        self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models_dir / "pose_landmarker_lite.task")),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))
        self.mask_smoothing = mask_smoothing
        self._prev_mask: np.ndarray | None = None
        self._last_pose: PoseObservation | None = None
        self._last_pose_ms = 0.0
        self._last_pose_ts = 0
        self._frames = 0
        self._last_ts = 0
        self._job: tuple[np.ndarray, int] | None = None
        self._result: SemanticResult | None = None
        self._cond = threading.Condition()
        self._stop = False
        self._pose_job = None
        self._pose_cond = threading.Condition()
        self._thread = threading.Thread(target=self._loop, name="semantic", daemon=True)
        self._thread.start()
        self._pose_thread = threading.Thread(target=self._pose_loop, name="semantic-pose", daemon=True)
        self._pose_thread.start()

    # threading -------------------------------------------------------------
    def submit(self, bgr: np.ndarray, timestamp_ms: int, full: bool = True) -> None:
        """Start analysing `bgr`. full=False runs only the person mask (no face/pose)."""
        with self._cond:
            self._job = (bgr, timestamp_ms, full)
            self._result = None
            self._cond.notify_all()

    def result(self, timeout: float = 0.5) -> SemanticResult | None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._result is None and not self._stop:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._result

    def analyze(self, bgr: np.ndarray, timestamp_ms: int, full: bool = True) -> SemanticResult:
        """Synchronous variant (tests, offline use): pose is computed on this same frame."""
        return self._run(bgr, timestamp_ms, full, with_pose=full)[0]

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._job is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                bgr, ts, full = self._job
                self._job = None
            image = None
            try:
                res, image = self._run(bgr, ts, full, with_pose=False)
            except Exception as exc:  # noqa: BLE001 - the costume layer degrades, frames keep flowing
                res = SemanticResult(error=f"{type(exc).__name__}: {exc}"[:200])
            with self._cond:
                self._result = res
                self._cond.notify_all()
            # Pose runs on its own thread at its own pace (newest frame only), off the frame's
            # critical path; the costume uses the latest pose, smoothed. Measured on an M1: running
            # it inline cost ~10 ms per frame and delayed the next frame's face pass.
            if full and image is not None:
                with self._pose_cond:
                    if self._pose_job is None:
                        self._pose_job = (image, ts, bgr.shape)
                        self._pose_cond.notify_all()

    # analysis ----------------------------------------------------------------
    def _run(self, bgr: np.ndarray, timestamp_ms: int, full: bool = True, with_pose: bool = True):
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        self._frames += 1
        h, w = bgr.shape[:2]
        timings: dict[str, float] = {}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        face = None
        t0 = time.perf_counter()
        if full:
            # Face first: it is the longest call, and starting it before the main thread enters the
            # Core ML predict (which holds the GIL) lets it run during the inference.
            fr = self._face.detect_for_video(image, ts)
            if fr.face_landmarks:
                lms = np.array([[p.x * w, p.y * h] for p in fr.face_landmarks[0]], np.float64)
                yaw = pitch = roll = 0.0
                matrix = None
                if fr.facial_transformation_matrixes:
                    matrix = np.asarray(fr.facial_transformation_matrixes[0], np.float64)
                    yaw, pitch, roll = _euler_from_matrix(matrix)
                shapes = {c.category_name: float(c.score) for c in fr.face_blendshapes[0]} \
                    if fr.face_blendshapes else {}
                face = FaceObservation(lms, shapes, yaw, pitch, roll, matrix)
            timings["face_ms"] = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        seg = self._segmenter.segment_for_video(image, ts)
        mask = seg.confidence_masks[0].numpy_view().squeeze().astype(np.float32)
        if self._prev_mask is not None and self._prev_mask.shape == mask.shape:
            mask = mask * (1.0 - self.mask_smoothing) + self._prev_mask * self.mask_smoothing
        self._prev_mask = mask
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        timings["segment_ms"] = (time.perf_counter() - t1) * 1000
        if not full:
            return SemanticResult(person=mask, timings=timings), image
        if with_pose:
            self._update_pose(image, ts, bgr.shape)
        if self._last_pose_ms:
            timings["pose_ms"] = self._last_pose_ms
        return SemanticResult(person=mask, face=face, pose=self._last_pose, timings=timings), image

    def _pose_loop(self) -> None:
        while True:
            with self._pose_cond:
                while self._pose_job is None and not self._stop:
                    self._pose_cond.wait()
                if self._stop:
                    return
                image, ts, shape = self._pose_job
            try:
                self._update_pose(image, ts, shape)
            except Exception:  # noqa: BLE001 - keep the last good pose
                pass
            with self._pose_cond:
                self._pose_job = None

    def _update_pose(self, image, ts: int, shape: tuple[int, ...]) -> None:
        h, w = shape[:2]
        t0 = time.perf_counter()
        ts = max(int(ts), self._last_pose_ts + 1)
        self._last_pose_ts = ts
        pr = self._pose.detect_for_video(image, ts)
        if pr.pose_landmarks:
            lm = pr.pose_landmarks[0]
            pts = np.array([[p.x * w, p.y * h] for p in lm], np.float64)
            vis = np.array([p.visibility for p in lm], np.float64)
            world = None
            if pr.pose_world_landmarks:
                world = np.array([[p.x, p.y, p.z] for p in pr.pose_world_landmarks[0]], np.float64)
            self._last_pose = PoseObservation(pts, vis, world)
        else:
            self._last_pose = None
        self._last_pose_ms = (time.perf_counter() - t0) * 1000

    def reset(self) -> None:
        self._prev_mask = None
        self._last_pose = None

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        with self._pose_cond:
            self._pose_cond.notify_all()
        self._thread.join(timeout=2.0)
        self._pose_thread.join(timeout=2.0)
        for task in (self._segmenter, self._face, self._pose):
            task.close()
