import numpy as np

from anime_portal.face_tracking import (
    CHIN,
    NOSE_TIP,
    SUBJECT_LEFT_EYE,
    SUBJECT_RIGHT_EYE,
    select_alignment_points,
)


def _landmarks() -> np.ndarray:
    points = np.zeros((478, 2), dtype=np.float64)
    for index in SUBJECT_RIGHT_EYE:
        points[index] = (100.0, 80.0)
    for index in SUBJECT_LEFT_EYE:
        points[index] = (300.0, 82.0)
    points[NOSE_TIP] = (200.0, 140.0)
    points[CHIN] = (198.0, 220.0)
    return points


def test_mirrored_preview_maps_subject_left_eye_to_image_left():
    aligned = select_alignment_points(_landmarks(), mirror=True)
    assert aligned is not None
    assert np.allclose(aligned[0], [300.0, 82.0])
    assert np.allclose(aligned[1], [100.0, 80.0])
    assert aligned[2, 0] == 200.0
    assert aligned[3, 1] == 220.0


def test_unmirrored_preview_maps_subject_right_eye_to_image_left():
    aligned = select_alignment_points(_landmarks(), mirror=False)
    assert aligned is not None
    assert aligned[0, 0] == 100.0
    assert aligned[1, 0] == 300.0


def test_short_landmark_list_is_ignored():
    assert select_alignment_points(np.zeros((10, 2)), mirror=True) is None
