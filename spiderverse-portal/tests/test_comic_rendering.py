import numpy as np
import pytest

from spiderverse_portal.config import ComicConfig
from spiderverse_portal.rendering.colors import contrast_lut, hue_snap_lut, misregister, saturation_lut
from spiderverse_portal.rendering.comic import ComicRenderer, protect_mask
from spiderverse_portal.rendering.halftone import PrintTexture, dot_coverage, dot_distance_field
from spiderverse_portal.rendering.shading import band_curve
from spiderverse_portal.stylization.exposure import ExposureNormalizer
from spiderverse_portal.stylization.temporal import TemporalStabilizer


def test_luts_are_valid_tables():
    for lut in (contrast_lut(1.2), saturation_lut(1.3), hue_snap_lut(0.4)):
        assert lut.shape == (256,) and lut.dtype == np.uint8
    assert np.all(np.diff(contrast_lut(1.2).astype(int)) >= 0), "contrast curve is monotonic"
    assert hue_snap_lut(0.0)[:180].tolist() == list(range(180))


def test_band_curve_has_requested_levels():
    q = band_curve(4, 0.001)
    plateaus = np.unique(np.round(q, 3))
    assert len(plateaus) <= 4 + 8  # 4 levels plus a few anti-aliased transition samples
    assert np.all(np.diff(q) >= -1e-9)


def test_comic_output_shape_and_determinism(astronaut_bgr):
    r = ComicRenderer(ComicConfig())
    a = r.render(astronaut_bgr)
    b = r.render(astronaut_bgr)
    assert a.shape == astronaut_bgr.shape and a.dtype == np.uint8
    assert np.array_equal(a, b), "no random per-frame texture: identical input gives identical output"


def test_print_texture_is_screen_fixed():
    """Changing the scene in one area must not move the dots anywhere else (no shimmer)."""
    r = ComicRenderer(ComicConfig())
    base = np.full((240, 320, 3), 90, np.uint8)
    other = base.copy()
    other[:, :100] = 200
    a, b = r.render(base), r.render(other)
    assert np.array_equal(a[:, 180:], b[:, 180:])


def test_halftone_coverage_tracks_darkness():
    tex = PrintTexture(1.0, 6.0, 45.0, 0.0, 0.0)
    dark = tex.coverage(np.full((60, 60), 40, np.uint8), None)[0].mean()
    light = tex.coverage(np.full((60, 60), 220, np.uint8), None)[0].mean()
    assert dark > light + 0.1


def test_dot_coverage_matches_area():
    field = dot_distance_field(120, 120, 8.0, 0.0)
    cov = dot_coverage(field, np.full(field.shape, 0.3, np.float32), 8.0)
    assert cov.mean() == pytest.approx(0.3, abs=0.03)


def test_protect_mask_keeps_textures_off_features():
    tex = PrintTexture(1.0, 6.0, 45.0, 0.5, 0.5)
    tone = np.full((200, 200), 60, np.uint8)
    region = np.array([[60, 60], [140, 60], [140, 140], [60, 140]], np.float32)
    mask = protect_mask((200, 200), [region])
    tinted, neutral = tex.coverage(tone, 1.0 - mask)
    assert tinted[90:110, 90:110].mean() < 0.05 * max(tinted[:30, :30].mean(), 1e-6) + 1e-3


def test_misregistration_moves_colour_plates_only_slightly():
    img = np.zeros((20, 20, 3), np.uint8)
    img[10, 10] = (255, 255, 255)
    out = misregister(img, 1)
    assert out[11, 11, 2] == 255 and out[10, 9, 0] == 255 and out[10, 10, 1] == 255


def test_exposure_brightens_dark_frames_smoothly():
    exp = ExposureNormalizer(max_gain=4.0, smoothing=0.2)
    rng = np.random.default_rng(0)
    dark = rng.integers(5, 45, (120, 160, 3)).astype(np.uint8)
    first = exp.apply(dark)
    assert first.mean() > dark.mean() * 2
    bright = rng.integers(100, 250, (120, 160, 3)).astype(np.uint8)
    g_before = exp.gain
    exp.apply(bright)
    assert exp.gain < g_before and exp.gain > 1.0, "gain moves gradually, not in one jump"


def test_temporal_stabilizer_reduces_noise_but_keeps_motion():
    rng = np.random.default_rng(0)
    stab = TemporalStabilizer(input_denoise=0.6, output_blend=0.5, motion_threshold=10)
    base = np.full((60, 80, 3), 128, np.uint8)
    outs = []
    for _ in range(20):
        noisy = np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)
        inp = stab.stabilize_input(noisy)
        outs.append(stab.stabilize_output(inp).astype(float))
    assert np.std(np.array(outs[10:]), axis=0).mean() < 2.0
    moved = base.copy()
    moved[20:40, 20:40] = 250  # a real change (e.g. an eye closing)
    out = stab.stabilize_output(stab.stabilize_input(moved))
    assert out[30, 30].mean() > 240, "moving regions take the new frame immediately"


def test_stabilizer_keeps_float32_state_with_numpy_scalar_gain():
    """Regression: an np.float64 exposure gain must not promote the float32 history (NumPy 2 rules)."""
    stab = TemporalStabilizer()
    stab.noise_gain = np.float64(3.5)
    frame = np.full((48, 64, 3), 40, np.uint8)
    for i in range(4):
        out = stab.stabilize_output(stab.stabilize_input(frame + i))
    assert out.dtype == np.uint8
    assert stab._prev_in.dtype == np.float32 and stab._prev_out.dtype == np.float32


def test_exposure_reports_python_floats():
    exp = ExposureNormalizer()
    exp.apply(np.full((60, 80, 3), 30, np.uint8))
    assert type(exp.gain) is float and type(exp.gamma) is float
