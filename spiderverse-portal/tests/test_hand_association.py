import numpy as np

from conftest import make_l_hand
from spiderverse_portal.tracking.hands import HandAssociator, HandDetection

FRAME = (640, 480)


def det(hand, label):
    return HandDetection(hand.landmarks.copy(), np.zeros(21), label, 0.9)


def test_ids_follow_physical_hands_through_a_crossing():
    assoc = HandAssociator(max_distance=0.25, forget_ms=400)
    rng = np.random.default_rng(1)
    ids_a = []
    for i in range(40):
        t = i / 30
        xa = 150 + i * 9  # A moves right
        xb = 500 - i * 9  # B moves left; they cross around i = 19
        a = make_l_hand((xa, 300), (0, -1), (1, 0))
        b = make_l_hand((xb, 330), (0, -1), (-1, 0))
        dets = [det(a, "Left"), det(b, "Right")]
        if rng.random() < 0.5:
            dets.reverse()  # MediaPipe order is arbitrary
        tracked = assoc.update(dets, t, FRAME)
        a_track = min(tracked, key=lambda h: np.linalg.norm(h.landmarks[0] - a.landmarks[0]))
        ids_a.append(a_track.track_id)
    assert len(set(ids_a)) == 1, f"hand A changed identity: {ids_a}"


def test_handedness_label_swap_does_not_swap_ids():
    assoc = HandAssociator()
    a = make_l_hand((200, 300), (0, -1), (1, 0))
    b = make_l_hand((440, 300), (0, -1), (-1, 0))
    first = assoc.update([det(a, "Left"), det(b, "Right")], 0.0, FRAME)
    ids = {round(h.landmarks[0][0]): h.track_id for h in first}
    swapped = assoc.update([det(a, "Right"), det(b, "Left")], 1 / 30, FRAME)
    assert {round(h.landmarks[0][0]): h.track_id for h in swapped} == ids


def test_track_survives_brief_disappearance_and_is_forgotten_later():
    assoc = HandAssociator(forget_ms=300)
    a = make_l_hand((200, 300), (0, -1), (1, 0))
    tid = assoc.update([det(a, "Left")], 0.0, FRAME)[0].track_id
    assert assoc.update([], 0.1, FRAME) == []
    assert assoc.update([det(a, "Left")], 0.2, FRAME)[0].track_id == tid
    assert assoc.update([det(a, "Left")], 1.0, FRAME)[0].track_id != tid


def test_far_detection_starts_new_track():
    assoc = HandAssociator(max_distance=0.1)
    a = make_l_hand((100, 300), (0, -1), (1, 0))
    b = make_l_hand((550, 300), (0, -1), (1, 0))
    t1 = assoc.update([det(a, "Left")], 0.0, FRAME)[0].track_id
    t2 = assoc.update([det(b, "Left")], 1 / 30, FRAME)[0].track_id
    assert t1 != t2
