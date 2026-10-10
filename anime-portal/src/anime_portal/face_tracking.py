"""MediaPipe face tracking and the anchors used to place the anime face.

Alignment uses the two eyes, the nose tip, and the chin. Those points follow
the head. The hand quad only decides where that face layer is allowed to show.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

SUBJECT_RIGHT_EYE: tuple[int, ...] = (33, 133, 159, 145)
SUBJECT_LEFT_EYE: tuple[int, ...] = (263, 362, 386, 374)
NOSE_TIP = 1
CHIN = 152


@dataclass
class FaceDetection:
    landmarks_norm: np.ndarray


def select_alignment_points(landmarks_xy: np.ndarray, mirror: bool) -> np.ndarray | None:
    """Return image-left eye, image-right eye, nose, and chin.

    MediaPipe eye indices are anatomical. After a horizontal mirror, the
    subject's left eye (263) is on the left side of the preview. Without the
    mirror, the subject's right eye (33) is on the left side of the image.
    """
    points = np.asarray(landmarks_xy, dtype=np.float64)
    needed = max((*SUBJECT_RIGHT_EYE, *SUBJECT_LEFT_EYE, NOSE_TIP, CHIN))
    if points.shape[0] <= needed or points.shape[1] < 2:
        return None
    subject_right = np.mean(points[list(SUBJECT_RIGHT_EYE)], axis=0)
    subject_left = np.mean(points[list(SUBJECT_LEFT_EYE)], axis=0)
    if mirror:
        image_left, image_right = subject_left, subject_right
    else:
        image_left, image_right = subject_right, subject_left
    return np.stack(
        [image_left, image_right, points[NOSE_TIP, :2], points[CHIN, :2]],
        axis=0,
    )


class FaceTracker:
    def __init__(
        self,
        model_path: str,
        *,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        self._mp = mp
        self._lock = threading.Lock()
        self._latest: FaceDetection | None = None
        self._timestamp_ms = 0
        self._last_submitted_ms = 0
        self._busy = False
        self._busy_since = 0.0
        self._skipped = 0
        self._errors = 0
        self._last_error = ""
        self._inflight: dict[int, object] = {}
        options = FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            result_callback=self._on_result,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def skipped(self) -> int:
        with self._lock:
            return self._skipped

    @property
    def last_error(self) -> str:
        return self._last_error

    def submit(self, bgr_frame: np.ndarray, timestamp_ms: int) -> None:
        import cv2

        timestamp = int(timestamp_ms)
        if timestamp <= self._last_submitted_ms:
            timestamp = self._last_submitted_ms + 1
        now = time.perf_counter()
        with self._lock:
            if self._busy and (now - self._busy_since) < 0.75:
                self._skipped += 1
                return
            self._busy = True
            self._busy_since = now
        self._last_submitted_ms = timestamp
        try:
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            with self._lock:
                self._inflight[timestamp] = image
                if len(self._inflight) > 6:
                    for old in sorted(self._inflight)[:-6]:
                        self._inflight.pop(old, None)
            self._landmarker.detect_async(image, timestamp)
        except Exception as exc:
            with self._lock:
                self._busy = False
                self._inflight.pop(timestamp, None)
            self._errors += 1
            self._last_error = str(exc)

    def latest(self) -> tuple[FaceDetection | None, int]:
        with self._lock:
            return self._latest, self._timestamp_ms

    def close(self) -> None:
        landmarker = getattr(self, "_landmarker", None)
        if landmarker is not None:
            landmarker.close()
            self._landmarker = None
        with self._lock:
            self._inflight.clear()

    def _on_result(self, result, _output_image, timestamp_ms: int) -> None:
        detection = None
        faces = result.face_landmarks or []
        if faces:
            points = np.array([[lm.x, lm.y] for lm in faces[0]], dtype=np.float32)
            detection = FaceDetection(points)
        with self._lock:
            self._busy = False
            self._inflight.pop(int(timestamp_ms), None)
            if int(timestamp_ms) < self._timestamp_ms:
                return
            self._latest = detection
            self._timestamp_ms = int(timestamp_ms)
