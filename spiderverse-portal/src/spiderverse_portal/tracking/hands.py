"""Two-hand tracking with MediaPipe Hand Landmarker and temporal identity.

MediaPipe returns detections in arbitrary order and its handedness label can
flip between frames. `HandAssociator` keeps a stable track id per physical
hand by matching each detection to a constant-velocity prediction of each
track (several keypoints, not only the wrist), with handedness as a weak cue.
This survives hand crossings, brief dropouts and label swaps.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 0, 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_TIP, RING_MCP, PINKY_MCP = 9, 12, 13, 17

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
_MATCH_POINTS = (WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, PINKY_MCP)


@dataclass
class HandDetection:
    landmarks: np.ndarray  # (21, 2) pixels in display space
    depth: np.ndarray  # (21,) MediaPipe relative z
    handedness: str
    score: float


@dataclass
class TrackedHand:
    track_id: int
    landmarks: np.ndarray
    depth: np.ndarray
    handedness: str
    score: float
    age_frames: int = 0

    @property
    def palm_length(self) -> float:
        return float(np.linalg.norm(self.landmarks[MIDDLE_MCP] - self.landmarks[WRIST]))

    @property
    def center(self) -> np.ndarray:
        return self.landmarks[[WRIST, INDEX_MCP, PINKY_MCP]].mean(axis=0)


@dataclass
class _Track:
    track_id: int
    points: np.ndarray
    velocity: np.ndarray
    handedness: str
    last_t: float
    age: int = 0
    hits: int = 1
    label_votes: dict[str, float] = field(default_factory=dict)


class HandAssociator:
    def __init__(self, max_distance: float = 0.25, forget_ms: float = 400.0) -> None:
        self.max_distance = max_distance  # fraction of frame diagonal
        self.forget_s = forget_ms / 1000.0
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def _cost(self, track: _Track, det: HandDetection, t: float, diag: float) -> float:
        dt = min(max(t - track.last_t, 0.0), 0.2)
        predicted = track.points + track.velocity * dt
        dist = float(np.linalg.norm(det.landmarks[list(_MATCH_POINTS)] - predicted, axis=1).mean()) / diag
        dominant = max(track.label_votes, key=track.label_votes.get) if track.label_votes else det.handedness
        if dominant != det.handedness:
            dist += 0.03  # weak cue; MediaPipe labels flip during crossings
        return dist

    def update(self, detections: list[HandDetection], t: float, frame_size: tuple[int, int]) -> list[TrackedHand]:
        diag = math.hypot(*frame_size)
        for tid in [tid for tid, tr in self._tracks.items() if t - tr.last_t > self.forget_s]:
            del self._tracks[tid]
        track_ids = list(self._tracks)
        best_assign: dict[int, int] = {}
        best_cost = math.inf
        # Brute force over assignments; at most 2 detections x 2 tracks.
        options = track_ids + [None] * len(detections)
        for perm in itertools.permutations(options, len(detections)):
            cost = 0.0
            assign: dict[int, int] = {}
            ok = True
            for di, tid in enumerate(perm):
                if tid is None:
                    cost += self.max_distance  # cost of starting a new track
                    continue
                c = self._cost(self._tracks[tid], detections[di], t, diag)
                if c > self.max_distance:
                    ok = False
                    break
                cost += c
                assign[di] = tid
            if ok and cost < best_cost:
                best_cost, best_assign = cost, assign

        out: list[TrackedHand] = []
        for di, det in enumerate(detections):
            pts = det.landmarks[list(_MATCH_POINTS)]
            tid = best_assign.get(di)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                track = _Track(tid, pts.copy(), np.zeros_like(pts), det.handedness, t)
                self._tracks[tid] = track
            else:
                track = self._tracks[tid]
                dt = t - track.last_t
                if dt > 1e-4:
                    vel = (pts - track.points) / dt
                    track.velocity = 0.6 * vel + 0.4 * track.velocity
                track.points = pts.copy()
                track.last_t = t
                track.age += 1
                track.hits += 1
            track.label_votes[det.handedness] = track.label_votes.get(det.handedness, 0.0) * 0.9 + det.score
            out.append(
                TrackedHand(tid, det.landmarks, det.depth, det.handedness, det.score, age_frames=track.age)
            )
        out.sort(key=lambda h: h.track_id)
        return out


class HandTracker:
    """MediaPipe Hand Landmarker in VIDEO mode (synchronous, releases the GIL)."""

    def __init__(self, model_path: str, max_hands: int = 2, min_detection: float = 0.5,
                 min_presence: float = 0.5, min_tracking: float = 0.5) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, timestamp_ms: int, out_size: tuple[int, int] | None = None) -> list[HandDetection]:
        """Landmarks in pixels of `out_size` (w, h), default the input size; MediaPipe output is normalised."""
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        w, h = out_size or (rgb.shape[1], rgb.shape[0])
        out: list[HandDetection] = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            arr = np.array([[lm.x * w, lm.y * h, lm.z] for lm in lms], dtype=np.float64)
            cat = handed[0] if handed else None
            out.append(
                HandDetection(
                    landmarks=arr[:, :2],
                    depth=arr[:, 2],
                    handedness=(cat.category_name if cat else "Unknown"),
                    score=float(cat.score if cat else 0.0),
                )
            )
        return out

    def close(self) -> None:
        self._landmarker.close()
