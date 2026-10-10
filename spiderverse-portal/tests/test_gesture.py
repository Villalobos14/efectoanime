import numpy as np

from conftest import frame_pair, make_l_hand
from spiderverse_portal.config import PortalConfig, TrackingConfig
from spiderverse_portal.tracking.gesture import GestureState, PortalGestureRecognizer, l_pose_score

FRAME = (640, 480)
DT = 1 / 30


def recognizer(**portal):
    return PortalGestureRecognizer(PortalConfig(**portal), TrackingConfig())


def run(rec, hands_seq, t0=0.0):
    outs = []
    for i, hands in enumerate(hands_seq):
        outs.append(rec.update(hands, t0 + i * DT, FRAME))
    return outs


def test_l_pose_scores_high_for_l_and_low_for_pinch():
    good, _ = l_pose_score(make_l_hand((200, 400), (0, -1), (1, 0)))
    pinch, reasons = l_pose_score(make_l_hand((200, 400), (0, -1), (1, 0), pinch=True))
    assert good > 0.9
    assert pinch < 0.3 and reasons


def test_activation_requires_sustained_valid_frames():
    rec = recognizer(activate_ms=90)
    outs = run(rec, [frame_pair()] * 6)
    states = [o.state for o in outs]
    assert states[0] == GestureState.CANDIDATE
    assert GestureState.ACTIVE in states
    first_active = states.index(GestureState.ACTIVE)
    assert first_active * DT * 1000 >= 90 - 1e-6
    assert outs[-1].visible and outs[-1].quad is not None


def test_single_hand_never_activates():
    rec = recognizer()
    outs = run(rec, [[frame_pair()[0]]] * 10)
    assert all(o.state == GestureState.INACTIVE for o in outs)
    assert "need two hands" in outs[-1].reasons


def test_release_goes_through_lost_then_inactive():
    rec = recognizer(activate_ms=60, lost_grace_ms=200)
    run(rec, [frame_pair()] * 5)
    assert rec.state == GestureState.ACTIVE
    left, right = frame_pair()
    pinched = make_l_hand(left.landmarks[0], (0, -1), (1, 0), pinch=True, track_id=1)
    outs = run(rec, [[pinched, right]] * 12, t0=1.0)
    states = [o.state for o in outs]
    assert states[0] == GestureState.TEMPORARILY_LOST
    assert outs[0].visible, "the portal holds through a brief loss"
    assert states[-1] == GestureState.INACTIVE
    assert not outs[-1].visible


def test_brief_dropout_recovers_without_flicker():
    rec = recognizer(activate_ms=60, lost_grace_ms=250)
    run(rec, [frame_pair()] * 5)
    pair = frame_pair()
    outs = run(rec, [[pair[0]], [pair[0]], pair, pair], t0=1.0)
    assert [o.state for o in outs] == [GestureState.TEMPORARILY_LOST, GestureState.TEMPORARILY_LOST,
                                      GestureState.ACTIVE, GestureState.ACTIVE]
    assert all(o.visible for o in outs)


def test_small_portal_rejected():
    rec = recognizer()
    outs = run(rec, [frame_pair(palm=18, spread=25)] * 6)
    assert all(o.state in (GestureState.INACTIVE,) for o in outs)
    assert any("too small" in r for r in outs[-1].reasons)


def test_twisted_corners_rejected():
    """Each hand's two fingertips must be adjacent corners; diagonal pairs mean a twisted frame."""
    rec = recognizer()
    left, right = frame_pair()
    left.landmarks[8], left.landmarks[4] = (200, 200), (440, 380)  # left hand spans TL -> BR
    right.landmarks[8], right.landmarks[4] = (440, 200), (200, 380)  # right hand spans TR -> BL
    outs = run(rec, [[left, right]] * 5)
    assert all(o.state != GestureState.ACTIVE for o in outs)
    assert "twisted corners" in outs[-1].reasons


def test_implausible_corner_jump_is_rejected_for_one_frame():
    rec = recognizer(activate_ms=60)
    run(rec, [frame_pair()] * 5)
    jumped = frame_pair(center=(320 + 300, 330))
    out = rec.update(jumped, 5 * DT, FRAME)
    assert "corner jump" in out.reasons
    assert out.state == GestureState.TEMPORARILY_LOST
    out2 = rec.update(jumped, 6 * DT, FRAME)
    assert out2.state == GestureState.ACTIVE, "a consistent new position is accepted on the next frame"


def test_portal_follows_translation_rotation_and_scale():
    rec = recognizer(activate_ms=30)
    run(rec, [frame_pair()] * 4)
    for center, angle, spread in (((360, 300), 0, 120), ((320, 330), 20, 120), ((320, 330), 0, 160)):
        outs = run(rec, [frame_pair(center=center, angle_deg=angle, spread=spread)] * 30, t0=10.0 + angle + spread)
        quad = outs[-1].quad
        raw = outs[-1].raw_quad
        assert outs[-1].state == GestureState.ACTIVE
        assert np.abs(quad - raw).max() < 3.0, "smoothed corners converge to the hands"


def test_hysteresis_keeps_active_between_stay_and_enter(monkeypatch):
    rec = recognizer(enter_score=0.6, stay_score=0.4, activate_ms=30)
    pair = frame_pair()
    real = rec._evaluate
    scores = iter([0.9, 0.9, 0.9, 0.5, 0.5, 0.5, 0.3])

    def fake(hands, t, size, diag):
        out = list(real(hands, t, size, diag))
        out[0] = next(scores)
        return tuple(out)

    monkeypatch.setattr(rec, "_evaluate", fake)
    states = [rec.update(pair, i * DT, FRAME).state for i in range(7)]
    assert states[2] == GestureState.ACTIVE
    assert states[3:6] == [GestureState.ACTIVE] * 3, "0.5 is enough to stay but not to enter"
    assert states[6] == GestureState.TEMPORARILY_LOST


def test_candidate_below_enter_threshold_does_not_start():
    rec = recognizer(enter_score=0.99, stay_score=0.4)
    left, right = frame_pair()
    weak = make_l_hand(left.landmarks[0], (0.55, -1), (1, -0.35), track_id=1)  # narrow L, lower score
    outs = run(rec, [[weak, right]] * 5)
    assert all(o.state == GestureState.INACTIVE for o in outs)


def test_reset_clears_state():
    rec = recognizer(activate_ms=30)
    run(rec, [frame_pair()] * 4)
    rec.reset()
    assert rec.state == GestureState.INACTIVE and rec.last_quad() is None
