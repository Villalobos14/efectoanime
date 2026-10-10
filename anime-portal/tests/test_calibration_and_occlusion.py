import numpy as np

from anime_portal.calibration import Calibration, load_calibration
from anime_portal.config import OcclusionSettings
from anime_portal.hand_occlusion import HandOccluder
from anime_portal.hand_tracking import HandIdentityTracker, TrackedHand


def test_calibration_roundtrip_and_limits(tmp_path):
    calibration = Calibration(offset_x=4, offset_y=-2, scale=1.1, rotation_deg=3)
    path = tmp_path / "calibration.json"
    calibration.save(path)
    loaded = load_calibration(path, Calibration())
    assert loaded.offset_x == 4
    assert loaded.offset_y == -2
    assert loaded.scale == 1.1
    assert loaded.rotation_deg == 3
    loaded.scale_by(10)
    assert loaded.scale == 2.5
    loaded.scale_by(0.01)
    assert loaded.scale == 0.4
    loaded.reset_to(Calibration())
    assert loaded.offset_x == 0
    assert loaded.scale == 1


def test_missing_calibration_uses_the_fallback(tmp_path):
    fallback = Calibration(offset_x=7)
    loaded = load_calibration(tmp_path / "missing.json", fallback)
    assert loaded.offset_x == 7
    assert loaded is not fallback


def test_finger_mask_covers_the_finger_and_not_the_corner():
    landmarks = np.zeros((21, 2), dtype=np.float64)
    landmarks[:] = (20.0, 100.0)
    landmarks[0] = (20.0, 100.0)
    landmarks[5] = (50.0, 100.0)
    landmarks[6] = (90.0, 100.0)
    landmarks[7] = (120.0, 100.0)
    landmarks[8] = (160.0, 100.0)
    landmarks[9] = (40.0, 150.0)
    occluder = HandOccluder(OcclusionSettings(thickness_scale=0.5, min_thickness=10, dilate_px=3))
    mask = occluder.mask((200, 200), [landmarks])
    assert mask[100, 100] > 0.9
    assert mask[2, 2] == 0


def test_hand_ids_stay_stable_when_detection_order_flips():
    tracker = HandIdentityTracker(max_distance=40)
    first = TrackedHand(np.zeros((21, 2)), "Right", 0.9, 0, np.array([10.0, 10.0]))
    second = TrackedHand(np.zeros((21, 2)), "Left", 0.9, 0, np.array([180.0, 12.0]))
    tracker.assign([first, second])
    again_second = TrackedHand(np.zeros((21, 2)), "Left", 0.9, 0, np.array([182.0, 14.0]))
    again_first = TrackedHand(np.zeros((21, 2)), "Right", 0.9, 0, np.array([12.0, 11.0]))
    tracker.assign([again_second, again_first])
    assert again_first.track_id == first.track_id
    assert again_second.track_id == second.track_id
    assert first.track_id != second.track_id
