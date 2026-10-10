"""Face tracking with MediaPipe Face Landmarker.

The face tracker never drives the stylized image. It supplies:
  * feature regions (eyes, brows, mouth, face oval) so the comic renderer can
    keep print textures off the eyes and mouth and keep skin tones warm,
  * expression readouts (blink, jaw open, smile) and head rotation for the HUD,
  * an expression-fidelity check: the same landmarker runs on the stylized
    frame and its blendshapes are compared with the camera frame's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
EYE_A = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
EYE_B = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
BROW_A = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
BROW_B = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)

EXPRESSION_KEYS = ("eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "mouthSmileLeft", "mouthSmileRight")


@dataclass
class FaceObservation:
    landmarks: np.ndarray  # (478, 2) pixels in display space
    blendshapes: dict[str, float] = field(default_factory=dict)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    matrix: np.ndarray | None = None  # 4x4 facial transformation matrix: canonical face (cm) -> metric camera space

    def region(self, indices: tuple[int, ...]) -> np.ndarray:
        pts = self.landmarks[list(indices)].astype(np.float32)
        return cv2.convexHull(pts).reshape(-1, 2)

    def feature_regions(self) -> list[np.ndarray]:
        return [self.region(EYE_A), self.region(EYE_B), self.region(BROW_A), self.region(BROW_B), self.region(LIPS)]

    def oval(self) -> np.ndarray:
        return self.region(FACE_OVAL)

    @property
    def center(self) -> np.ndarray:
        return self.landmarks[list(FACE_OVAL)].mean(axis=0)

    def expression(self) -> dict[str, float]:
        b = self.blendshapes
        return {
            "blink": 0.5 * (b.get("eyeBlinkLeft", 0.0) + b.get("eyeBlinkRight", 0.0)),
            "jaw": b.get("jawOpen", 0.0),
            "smile": 0.5 * (b.get("mouthSmileLeft", 0.0) + b.get("mouthSmileRight", 0.0)),
        }


def _euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    r = m[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
        yaw = math.degrees(math.atan2(-r[2, 0], sy))
        roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-r[1, 2], r[1, 1]))
        yaw = math.degrees(math.atan2(-r[2, 0], sy))
        roll = 0.0
    return yaw, pitch, roll


class FaceTracker:
    def __init__(self, model_path: str, video_mode: bool = True) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._mp = mp
        self._video = video_mode
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, timestamp_ms: int | None = None,
               out_size: tuple[int, int] | None = None) -> FaceObservation | None:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            ts = max(int(timestamp_ms or 0), self._last_ts + 1)
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        w, h = out_size or (rgb.shape[1], rgb.shape[0])
        lms = np.array([[lm.x * w, lm.y * h] for lm in result.face_landmarks[0]], dtype=np.float64)
        shapes = {}
        if result.face_blendshapes:
            shapes = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        yaw = pitch = roll = 0.0
        matrix = None
        matrices = getattr(result, "facial_transformation_matrixes", None)
        if matrices:
            matrix = np.asarray(matrices[0], np.float64)
            yaw, pitch, roll = _euler_from_matrix(matrix)
        return FaceObservation(lms, shapes, yaw, pitch, roll, matrix)

    def close(self) -> None:
        self._landmarker.close()
