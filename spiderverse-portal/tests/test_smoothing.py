import numpy as np
import pytest

from spiderverse_portal.tracking.smoothing import ExponentialMovingAverage, KeyedOneEuro, OneEuroFilter


def test_one_euro_reduces_jitter_on_static_point():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=6.0)
    raw, out = [], []
    for i in range(200):
        x = np.array([0.5, 0.5]) + rng.normal(0, 0.004, 2)
        raw.append(x)
        out.append(f(x, i / 30.0))
    raw, out = np.array(raw[50:]), np.array(out[50:])
    assert out.std(axis=0).max() < raw.std(axis=0).max() / 3


def test_one_euro_follows_fast_motion_with_small_lag():
    f = OneEuroFilter(min_cutoff=1.4, beta=6.0)
    lags = []
    for i in range(60):
        x = np.array([i * 0.02, 0.0])  # 0.6 diagonals per second
        y = f(x, i / 30.0)
        lags.append(x[0] - y[0])
    assert lags[-1] < 0.03, "steady-state lag should stay within ~1.5 frames of motion"


def test_one_euro_slow_filter_lags_more_than_fast():
    slow, fast = OneEuroFilter(1.4, 0.0), OneEuroFilter(1.4, 6.0)
    for i in range(30):
        x = np.array([i * 0.02])
        a, b = slow(x, i / 30.0), fast(x, i / 30.0)
    assert (30 * 0.02 - a[0]) > (30 * 0.02 - b[0])


def test_low_confidence_smooths_more():
    hi, lo = OneEuroFilter(), OneEuroFilter()
    hi(np.zeros(1), 0.0)
    lo(np.zeros(1), 0.0)
    assert hi(np.ones(1), 1 / 30, confidence=1.0)[0] > lo(np.ones(1), 1 / 30, confidence=0.1)[0]


def test_ema():
    ema = ExponentialMovingAverage(0.5)
    assert ema(np.array([0.0]))[0] == 0.0
    assert ema(np.array([1.0]))[0] == 0.5
    with pytest.raises(ValueError):
        ExponentialMovingAverage(0.0)


def test_keyed_filters_are_independent_and_retained():
    k = KeyedOneEuro(1.4, 6.0, 1.0)
    k("a", np.zeros(2), 0.0)
    k("b", np.ones(2), 0.0)
    assert k("a", np.zeros(2), 0.03)[0] == 0.0
    k.retain({"a"})
    assert k("b", np.full(2, 5.0), 0.06)[0] == 5.0  # re-created: no history
