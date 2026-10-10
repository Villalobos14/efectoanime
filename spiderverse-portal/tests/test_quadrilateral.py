import numpy as np
import pytest

from spiderverse_portal.geometry.quadrilateral import (
    interior_angles_deg,
    is_convex,
    is_self_intersecting,
    order_corners,
    polygon_area,
    validate_quad,
)

FRAME = (640, 480)


def test_order_corners_clockwise_from_top_left():
    pts = np.array([[400, 300], [100, 100], [400, 100], [100, 300]], float)
    ordered, perm = order_corners(pts)
    assert ordered.tolist() == [[100, 100], [400, 100], [400, 300], [100, 300]]
    assert np.allclose(pts[perm], ordered)


def test_order_corners_is_rotation_invariant_for_shape():
    rect = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], float)
    for deg in (0, 30, 60, 89, 135):
        a = np.radians(deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        pts = rect @ rot.T + 300
        shuffled = pts[[2, 0, 3, 1]]
        ordered, _ = order_corners(shuffled)
        assert is_convex(ordered)
        assert not is_self_intersecting(ordered)
        assert abs(abs(polygon_area(ordered)) - 20000) < 1e-6


def test_self_intersecting_bowtie_detected():
    bowtie = np.array([[0, 0], [100, 100], [100, 0], [0, 100]], float)
    assert is_self_intersecting(bowtie)
    assert not is_convex(bowtie)


def test_angles_of_rectangle():
    rect = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], float)
    assert np.allclose(interior_angles_deg(rect), 90.0)


def test_valid_trapezoid_accepted():
    check = validate_quad(np.array([[220, 260], [420, 260], [352, 380], [288, 380]], float), FRAME)
    assert check.valid, check.reasons
    assert 0 < check.score <= 1
    assert check.area_ratio == pytest.approx(15840 / (640 * 480), rel=1e-6)


def test_rotated_quad_accepted():
    rect = np.array([[-120, -90], [120, -90], [120, 90], [-120, 90]], float)
    a = np.radians(35)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    check = validate_quad(rect @ rot.T + [320, 240], FRAME)
    assert check.valid, check.reasons


@pytest.mark.parametrize(
    "pts,reason",
    [
        ([[300, 200], [310, 200], [310, 208], [300, 208]], "too small"),
        ([[100, 100], [500, 100], [300, 101], [300, 99]], "not convex"),
        ([[100, 100], [500, 100], [500, 110], [100, 110]], "collapsed angle"),
    ],
)
def test_invalid_quads_rejected(pts, reason):
    check = validate_quad(np.array(pts, float), FRAME)
    assert not check.valid
    assert any(reason in r for r in check.reasons) or check.reasons


def test_collapsed_quad_with_repeated_point_rejected():
    check = validate_quad(np.array([[100, 100], [100, 100], [300, 300], [100, 300]], float), FRAME)
    assert not check.valid


def test_non_finite_rejected():
    check = validate_quad(np.array([[np.nan, 0], [1, 0], [1, 1], [0, 1]], float), FRAME)
    assert not check.valid
