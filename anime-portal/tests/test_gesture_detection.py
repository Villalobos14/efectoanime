import numpy as np

from anime_portal.gesture_detection import GestureConfig, GestureHysteresis, PortalSession, evaluate_frame_gesture
from anime_portal.gesture_detection import SmoothConfig


def l_hand(wrist, index_dir, thumb_dir, scale=100.0) -> np.ndarray:
    wrist = np.asarray(wrist, dtype=np.float64)
    index_dir = np.asarray(index_dir, dtype=np.float64)
    thumb_dir = np.asarray(thumb_dir, dtype=np.float64)
    index_dir = index_dir / np.linalg.norm(index_dir)
    thumb_dir = thumb_dir / np.linalg.norm(thumb_dir)
    landmarks = np.repeat(wrist.reshape(1, 2), 21, axis=0)
    landmarks[2] = wrist + thumb_dir * scale * 0.25
    landmarks[3] = wrist + thumb_dir * scale * 0.55
    landmarks[4] = wrist + thumb_dir * scale
    landmarks[5] = wrist + index_dir * scale * 0.45
    landmarks[6] = wrist + index_dir * scale * 0.70
    landmarks[7] = wrist + index_dir * scale * 0.85
    landmarks[8] = wrist + index_dir * scale
    landmarks[9] = wrist + (index_dir + thumb_dir) * scale * 0.35
    return landmarks


def frame_hands() -> list[np.ndarray]:
    left = l_hand((160.0, 340.0), (0.0, -1.0), (1.0, 0.0))
    right = l_hand((480.0, 340.0), (0.0, -1.0), (-1.0, 0.0))
    return [left, right]


def test_two_l_hands_form_a_portal():
    estimate = evaluate_frame_gesture(frame_hands(), (480, 640), GestureConfig())
    assert estimate.valid, estimate.reason
    assert estimate.corners is not None
    assert estimate.corners.shape == (4, 2)
    assert estimate.score > 0.5


def test_one_hand_is_not_enough():
    estimate = evaluate_frame_gesture(frame_hands()[:1], (480, 640), GestureConfig())
    assert not estimate.valid
    assert estimate.reason == "need both hands"


def test_pinched_fingers_are_rejected():
    pinched = l_hand((200.0, 300.0), (0.0, -1.0), (0.08, -1.0))
    other = frame_hands()[1]
    estimate = evaluate_frame_gesture([pinched, other], (480, 640), GestureConfig())
    assert not estimate.valid


def test_hysteresis_requires_a_run_and_ignores_a_short_gap():
    machine = GestureHysteresis(activate_frames=3, deactivate_frames=4)
    assert not machine.update(True)
    assert not machine.update(True)
    assert machine.update(True)
    assert machine.update(False)
    assert machine.active
    assert machine.update(False)
    assert machine.update(False)
    assert not machine.update(False)


def test_portal_session_fades_state_without_dropping_on_one_lost_frame():
    session = PortalSession(
        GestureConfig(activate_frames=3, deactivate_frames=4),
        SmoothConfig(min_cutoff=8.0, beta=0.0, d_cutoff=1.0, max_jump_px=500, jump_accept_frames=2),
    )
    hands = frame_hands()
    state = None
    for index in range(3):
        state = session.update(hands, (480, 640), index * 0.03)
    assert state is not None
    assert state.active
    assert state.corners is not None
    held = session.update([], (480, 640), 0.2)
    assert held.active
    assert held.status == "holding"
    assert held.corners is not None
    for index in range(3):
        held = session.update([], (480, 640), 0.3 + index * 0.03)
    assert not held.active
