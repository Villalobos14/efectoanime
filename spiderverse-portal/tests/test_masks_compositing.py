import numpy as np
import pytest

from spiderverse_portal.config import OcclusionConfig
from spiderverse_portal.geometry.masks import polygon_alpha, polygon_alpha_roi
from spiderverse_portal.rendering.compositor import PortalAnimator, PortalCompositor, blend_roi
from spiderverse_portal.rendering.occlusion import HandOccluder

FRAME = (640, 480)
RECT = np.array([[100, 120], [400, 120], [400, 360], [100, 360]], float)


def test_polygon_alpha_inside_outside_and_area():
    a = polygon_alpha(RECT, FRAME, 0.0)
    assert a.dtype == np.float32 and a.shape == (480, 640)
    assert a[240, 250] == 1.0
    assert a[50, 50] == 0.0 and a[400, 500] == 0.0
    perimeter = 2 * (300 + 240)
    assert abs(a.sum() - 300 * 240) < perimeter * 0.75  # sub-pixel edge accuracy


def test_feathering_produces_soft_edge():
    hard = polygon_alpha(RECT, FRAME, 0.0)
    soft = polygon_alpha(RECT, FRAME, 3.0)
    row = soft[240, 90:110]
    assert np.any((row > 0.05) & (row < 0.95))
    assert np.count_nonzero((hard[240, 90:110] > 0.05) & (hard[240, 90:110] < 0.95)) <= 2


def test_roi_is_clipped_to_frame():
    res = polygon_alpha_roi(np.array([[-50, -50], [100, -50], [100, 100], [-50, 100]], float), FRAME)
    alpha, (x0, y0, x1, y1) = res
    assert x0 == 0 and y0 == 0 and alpha.shape == (y1 - y0, x1 - x0)
    assert polygon_alpha_roi(np.array([[700, 500], [800, 500], [800, 600], [700, 600]], float), FRAME) is None


def test_composite_reveals_stylized_only_inside_portal():
    original = np.zeros((480, 640, 3), np.uint8)
    stylized = np.full((480, 640, 3), 255, np.uint8)
    comp = PortalCompositor(HandOccluder(OcclusionConfig(enabled=False)), 0.0, 0, 0)
    out = comp.compose(original, stylized, RECT, 1.0)
    assert (out[240, 250] == 255).all()
    assert (out[50, 50] == 0).all() and (out[400, 500] == 0).all()
    assert (original == 0).all(), "compose must not modify the camera frame"


def test_composite_preserves_pixel_alignment_under_rotation():
    """The portal shapes the mask only: every revealed pixel equals the stylized pixel at the same (x, y)."""
    yy, xx = np.mgrid[0:480, 0:640]
    stylized = np.dstack([(xx % 256), (yy % 256), ((xx + yy) % 256)]).astype(np.uint8)
    original = np.zeros_like(stylized)
    a = np.radians(28)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    quad = (RECT - RECT.mean(0)) @ rot.T + [320, 240]
    comp = PortalCompositor(HandOccluder(OcclusionConfig(enabled=False)), 0.0, 0, 0)
    out = comp.compose(original, stylized, quad, 1.0)
    inside = polygon_alpha(quad, FRAME, 0.0) >= 1.0
    assert inside.sum() > 10000
    assert np.array_equal(out[inside], stylized[inside])


def test_alpha_scale_blends_linearly():
    original = np.zeros((480, 640, 3), np.uint8)
    stylized = np.full((480, 640, 3), 200, np.uint8)
    comp = PortalCompositor(HandOccluder(OcclusionConfig(enabled=False)), 0.0, 0, 0)
    out = comp.compose(original, stylized, RECT, 0.5)
    assert abs(int(out[240, 250, 0]) - 100) <= 1


def test_blend_roi_in_place():
    dst = np.zeros((10, 10, 3), np.uint8)
    src = np.full((10, 10, 3), 100, np.uint8)
    blend_roi(dst, src, np.ones((4, 4), np.float32), (2, 2, 6, 6))
    assert (dst[2:6, 2:6] == 100).all() and (dst[0, 0] == 0).all()


def test_portal_animator_fades_with_time():
    anim = PortalAnimator(fade_in_ms=100, fade_out_ms=200)
    anim.update(True, RECT, now=0.0)
    assert anim.update(True, RECT, now=0.05) == pytest.approx(0.5)
    assert anim.update(True, RECT, now=0.2) == 1.0
    assert anim.update(False, None, now=0.3) == pytest.approx(0.5)
    assert anim.quad is not None, "fade-out keeps the last quad"
    assert anim.update(False, None, now=0.5) == 0.0
    assert anim.quad is None
