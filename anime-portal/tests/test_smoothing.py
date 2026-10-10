import math

import numpy as np

from anime_portal.smoothing import MultiPointSmoother, OneEuroFilter, SampleHold, TemporalFade


def test_constant_signal_stays_constant():
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    assert filt(5.0, 0.0) == 5.0
    assert filt(5.0, 0.1) == 5.0
    assert filt(5.0, 0.2) == 5.0


def test_step_reaches_target():
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.7, d_cutoff=1.0)
    timestamp = 0.0
    value = 0.0
    for _ in range(90):
        timestamp += 1.0 / 30.0
        value = filt(1.0, timestamp)
    assert value > 0.9


def test_high_frequency_noise_is_attenuated():
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    rng = np.random.default_rng(0)
    timestamp = 0.0
    raw = []
    smoothed = []
    for _ in range(300):
        timestamp += 1.0 / 30.0
        sample = math.sin(timestamp * 0.4) + float(rng.normal(0.0, 1.0))
        raw.append(sample)
        smoothed.append(filt(sample, timestamp))
    assert np.var(np.diff(smoothed)) < np.var(np.diff(raw)) * 0.5


def test_jump_is_rejected_until_it_persists():
    smoother = MultiPointSmoother(1, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, max_jump_px=10, jump_accept_frames=3)
    smoother.reset([[0.0, 0.0]], 0.0)
    first = smoother.smooth([[100.0, 0.0]], 0.03)
    second = smoother.smooth([[100.0, 0.0]], 0.06)
    third = smoother.smooth([[100.0, 0.0]], 0.09)
    assert first[0, 0] == 0.0
    assert second[0, 0] == 0.0
    assert third[0, 0] == 100.0


def test_small_motion_is_filtered_instead_of_snapped():
    smoother = MultiPointSmoother(1, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, max_jump_px=50, jump_accept_frames=3)
    smoother.reset([[0.0, 0.0]], 0.0)
    moved = smoother.smooth([[5.0, 0.0]], 1.0 / 30.0)
    assert 0.0 < moved[0, 0] < 5.0


def test_fade_moves_toward_the_requested_state():
    fade = TemporalFade(attack=40.0, release=40.0)
    opened = fade.update(True, 0.05)
    assert opened > 0.5
    closed = fade.update(False, 0.05)
    assert closed < opened


def test_sample_hold_bridges_a_short_gap():
    hold = SampleHold(0.3)
    first = hold.update(np.array([1.0, 2.0]), 0.0)
    assert first is not None
    assert hold.update(None, 0.2) is not None
    assert hold.update(None, 0.6) is None
