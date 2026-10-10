from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from spiderverse_portal.tracking.hands import TrackedHand

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"


def make_l_hand(wrist, index_dir, thumb_dir, palm: float = 80.0, track_id: int = 1, score: float = 0.95,
                pinch: bool = False) -> TrackedHand:
    """21 MediaPipe-style landmarks for a hand making an 'L' (index along index_dir, thumb along thumb_dir)."""
    w = np.asarray(wrist, np.float64)
    u = np.asarray(index_dir, np.float64)
    u /= np.linalg.norm(u)
    v = np.asarray(thumb_dir, np.float64)
    v /= np.linalg.norm(v)
    P = palm
    lm = np.zeros((21, 2))
    lm[0] = w
    lm[1] = w + v * 0.25 * P + u * 0.15 * P
    lm[2] = w + v * 0.50 * P + u * 0.25 * P
    lm[3] = lm[2] + v * 0.30 * P
    lm[4] = lm[2] + v * 0.60 * P
    lm[5] = w + u * 0.95 * P + v * 0.25 * P
    lm[6] = lm[5] + u * 0.35 * P
    lm[7] = lm[5] + u * 0.60 * P
    lm[8] = lm[5] + u * 0.80 * P
    for mcp, off in ((9, 0.0), (13, -0.2), (17, -0.4)):
        base = w + u * (1.0 if mcp == 9 else 0.95 if mcp == 13 else 0.85) * P + v * off * P
        lm[mcp] = base
        lm[mcp + 1] = base + u * 0.15 * P
        lm[mcp + 2] = base + u * 0.05 * P
        lm[mcp + 3] = base - u * 0.10 * P
    if pinch:
        lm[4] = lm[8].copy()
        lm[3] = lm[4] - v * 0.2 * P
    return TrackedHand(track_id=track_id, landmarks=lm, depth=np.zeros(21), handedness="Left", score=score)


def frame_pair(center=(320, 330), palm=80.0, angle_deg=0.0, spread=120.0):
    """Two hands framing a trapezoid around `center`, rotated by angle_deg."""
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.asarray(center, np.float64)
    up = rot @ np.array([0.0, -1.0])
    right = rot @ np.array([1.0, 0.0])
    left = make_l_hand(c - right * spread + rot @ np.array([0, 70.0]), up, right, palm, track_id=1)
    right_hand = make_l_hand(c + right * spread + rot @ np.array([0, 70.0]), up, -right, palm, track_id=2)
    return [left, right_hand]


@pytest.fixture
def astronaut_bgr():
    import cv2

    img = cv2.imread(str(DATA / "astronaut.png"))
    assert img is not None, "tests/data/astronaut.png missing"
    return cv2.resize(img, (640, 640))[80:560]


def models_available() -> bool:
    return (ROOT / "models" / "animegan2" / "animegan2_paprika.onnx").exists()


requires_models = pytest.mark.skipif(not models_available(), reason="run tools/setup_models.py first")
requires_camera = pytest.mark.skipif(os.environ.get("SPIDERVERSE_CAMERA_TESTS") != "1",
                                     reason="set SPIDERVERSE_CAMERA_TESTS=1 to use the physical webcam")
