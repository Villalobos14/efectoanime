"""MediaPipe hand tracking in live-stream mode.

The landmarker is created once. Each camera frame is submitted with a strictly
increasing timestamp and the newest landmarks are read back on the render thread.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class HandDetection:
    landmarks_norm: np.ndarray
    handedness: str
    score: float


@dataclass
class TrackedHand:
    landmarks_px: np.ndarray
    handedness: str
    score: float
    track_id: int
    center: np.ndarray


def normalized_to_pixels(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float64).copy()
    points[:, 0] *= float(width)
    points[:, 1] *= float(height)
    return points


class HandIdentityTracker:
    """Keep a stable id on each hand by matching wrist centers between frames."""

    def __init__(self, max_distance: float = 180.0, forget_frames: int = 12) -> None:
        self.max_distance = float(max_distance)
        self.forget_frames = int(forget_frames)
        self._tracks: dict[int, dict[str, object]] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def assign(self, hands: list[TrackedHand]) -> list[TrackedHand]:
        unused = set(self._tracks)
        for hand in hands:
            best_id: int | None = None
            best_distance = self.max_distance
            for track_id in list(unused):
                center = self._tracks[track_id]["center"]
                assert isinstance(center, np.ndarray)
                distance = float(np.linalg.norm(hand.center - center))
                if distance < best_distance:
                    best_distance = distance
                    best_id = track_id
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            else:
                unused.discard(best_id)
            self._tracks[best_id] = {"center": hand.center.copy(), "missed": 0}
            hand.track_id = best_id
        for track_id in unused:
            missed = int(self._tracks[track_id]["missed"]) + 1
            self._tracks[track_id]["missed"] = missed
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if int(track["missed"]) <= self.forget_frames
        }
        return hands


class HandTracker:
    def __init__(
        self,
        model_path: str,
        *,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        self._mp = mp
        self._lock = threading.Lock()
        self._latest: list[HandDetection] = []
        self._timestamp_ms = 0
        self._last_submitted_ms = 0
        self._busy = False
        self._busy_since = 0.0
        self._skipped = 0
        self._errors = 0
        self._last_error = ""
        self._inflight: dict[int, object] = {}
        options = HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self._on_result,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

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
            # A dropped live-stream frame does not call back. Only one frame is
            # in flight, and a watchdog clears a callback that never arrives.
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

    def latest(self) -> tuple[list[HandDetection], int]:
        with self._lock:
            return list(self._latest), self._timestamp_ms

    def close(self) -> None:
        landmarker = getattr(self, "_landmarker", None)
        if landmarker is not None:
            landmarker.close()
            self._landmarker = None
        with self._lock:
            self._inflight.clear()

    def _on_result(self, result, _output_image, timestamp_ms: int) -> None:
        detections: list[HandDetection] = []
        landmarks = result.hand_landmarks or []
        handedness = result.handedness or []
        for index, hand in enumerate(landmarks):
            points = np.array([[lm.x, lm.y] for lm in hand], dtype=np.float32)
            label = "Unknown"
            score = 0.0
            if index < len(handedness) and handedness[index]:
                category = handedness[index][0]
                label = category.category_name or category.display_name or "Unknown"
                score = float(category.score or 0.0)
            detections.append(HandDetection(points, label, score))
        with self._lock:
            self._busy = False
            self._inflight.pop(int(timestamp_ms), None)
            if int(timestamp_ms) < self._timestamp_ms:
                return
            self._latest = detections
            self._timestamp_ms = int(timestamp_ms)
