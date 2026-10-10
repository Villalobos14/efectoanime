import time

import numpy as np
import pytest

from anime_portal.config import project_root


def _blank_image():
    import mediapipe as mp

    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def test_hand_landmarker_accepts_image_and_live_stream_modes():
    model = project_root() / "models" / "hand_landmarker.task"
    if not model.exists():
        pytest.skip("hand model is not downloaded")
    import mediapipe as mp
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    image = _blank_image()
    with HandLandmarker.create_from_options(
        HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model)))
    ) as landmarker:
        result = landmarker.detect(image)
        assert result.hand_landmarks is not None

    received: list[int] = []

    def callback(result, _output, timestamp_ms: int) -> None:
        received.append(timestamp_ms)
        assert result.hand_landmarks is not None

    live = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=callback,
        )
    )
    try:
        live.detect_async(image, 1000)
        deadline = time.time() + 5
        while not received and time.time() < deadline:
            time.sleep(0.05)
        assert received == [1000]
    finally:
        live.close()


def test_face_landmarker_accepts_image_mode():
    model = project_root() / "models" / "face_landmarker.task"
    if not model.exists():
        pytest.skip("face model is not downloaded")
    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

    image = _blank_image()
    with FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            num_faces=1,
            output_face_blendshapes=False,
        )
    ) as landmarker:
        result = landmarker.detect(image)
        assert result.face_landmarks is not None
