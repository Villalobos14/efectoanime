import numpy as np

from anime_portal.anime_renderer import (
    LEFT_EYE_PX,
    RIGHT_EYE_PX,
    compose_face_transform,
    write_placeholder_face,
)
from anime_portal.app import interpret_key
from anime_portal.calibration import Calibration
from anime_portal.config import FaceSettings, OcclusionSettings, PortalStyle, load_settings
from anime_portal.gesture_detection import SmoothConfig


def test_keyboard_commands():
    assert interpret_key(ord("q")) == "quit"
    assert interpret_key(ord("Q")) == "quit"
    assert interpret_key(ord("S")) == "save"
    assert interpret_key(ord("d")) == "debug"
    assert interpret_key(ord("s")) == "save"
    assert interpret_key(27) == "quit"
    assert interpret_key(-1) is None
    assert interpret_key(0) is None
    assert interpret_key(63234) == "left"
    assert interpret_key(ord("j")) == "left"
    assert interpret_key(ord("i")) == "up"
    assert interpret_key(ord("[")) == "rotate_ccw"
    assert interpret_key(ord("=")) == "scale_up"


def test_settings_load_the_project_defaults():
    settings = load_settings()
    assert settings.camera.width == 640
    assert settings.camera.height == 480
    assert settings.camera.mirror is True
    assert settings.hand_model.name == "hand_landmarker.task"
    assert settings.anchors.nose[0] == 0.5


def test_identity_alignment_and_calibration_offset():
    src = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 6.0], [5.0, 14.0]], dtype=np.float32)
    dst = src + np.array([50.0, 20.0], dtype=np.float32)
    transform = compose_face_transform(src, dst, Calibration(), pivot=dst[:2].mean(axis=0))
    assert transform is not None
    mapped = transform @ np.array([src[0, 0], src[0, 1], 1.0])
    assert np.allclose(mapped, dst[0], atol=0.05)
    shifted = compose_face_transform(src, dst, Calibration(offset_x=10, offset_y=-4), pivot=dst[:2].mean(axis=0))
    assert shifted is not None
    moved = shifted @ np.array([src[0, 0], src[0, 1], 1.0])
    assert np.allclose(moved, [dst[0, 0] + 10, dst[0, 1] - 4], atol=0.05)


def test_placeholder_eyes_are_opaque_and_the_corner_is_clear(tmp_path):
    path = tmp_path / "anime_face.png"
    write_placeholder_face(path)
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None
    assert image.shape[2] == 4
    assert image[LEFT_EYE_PX[1], LEFT_EYE_PX[0], 3] > 200
    assert image[RIGHT_EYE_PX[1], RIGHT_EYE_PX[0], 3] > 200
    assert image[0, 0, 3] == 0


def test_anime_is_clipped_to_the_quad_and_hands_stay_in_front(tmp_path):
    import cv2

    from anime_portal.anime_renderer import NOSE_PX, AnimeRenderer, placeholder_anchors

    path = tmp_path / "anime_face.png"
    write_placeholder_face(path)
    anchors = placeholder_anchors()
    renderer = AnimeRenderer(
        path,
        anchors,
        style=PortalStyle(feather_px=0, edge_width=1, grade_mix=0.0, grade_saturation=1.0, grade_value=1.0),
        occlusion=OcclusionSettings(thickness_scale=0.6, min_thickness=16, max_thickness=40, dilate_px=2),
        face_smoothing=SmoothConfig(min_cutoff=10, beta=1, d_cutoff=1, max_jump_px=500, jump_accept_frames=1),
        face_settings=FaceSettings(hold_sec=0.2, min_eye_px=5, max_eye_px=800),
    )
    shift = np.array([50.0, 20.0])
    face = renderer.src_anchors + shift
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (0, 180, 0)
    corners = np.array([[180.0, 200.0], [460.0, 200.0], [460.0, 470.0], [180.0, 470.0]])
    rendered = renderer.render(frame, corners, face, [], 1.0, Calibration(), 0.0)
    nose = (np.asarray(NOSE_PX, dtype=np.float64) + shift).astype(int)
    patch = rendered[nose[1] - 4 : nose[1] + 5, nose[0] - 4 : nose[0] + 5]
    assert patch[..., 0].max() > 100
    assert np.array_equal(rendered[5, 5], frame[5, 5])

    finger = np.zeros((21, 2), dtype=np.float64)
    finger[:] = (nose[0], nose[1])
    finger[0] = (nose[0], nose[1] + 80)
    finger[5] = (nose[0], nose[1] + 40)
    finger[9] = (nose[0] + 30, nose[1] + 70)
    covered = renderer.render(frame, corners, face, [finger], 1.0, Calibration(), 0.2)
    assert covered[nose[1], nose[0], 0] < 40
