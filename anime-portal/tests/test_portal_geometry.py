import numpy as np
import pytest

from anime_portal.portal_geometry import (
    canonical_order,
    interior_angles_deg,
    is_convex,
    is_self_intersecting,
    match_corners,
    polygon_area,
    validate_quad,
)

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
BOWTIE = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


def test_square_area_and_right_angles():
    assert polygon_area(SQUARE) == pytest.approx(1.0)
    assert np.all(np.isclose(interior_angles_deg(SQUARE), 90.0))
    assert is_convex(SQUARE)
    assert not is_self_intersecting(SQUARE)


def test_bowtie_is_rejected():
    assert is_self_intersecting(BOWTIE)
    assert not is_convex(BOWTIE)
    ok, reason = validate_quad(
        BOWTIE,
        image_area=4.0,
        min_area_ratio=0.01,
        max_area_ratio=0.9,
        min_corner_angle_deg=20,
        max_side_ratio=4,
    )
    assert not ok
    assert reason in {"self-intersection", "not convex"}


def test_point_inside_triangle_is_not_a_quad():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0], [1.0, 0.4]], dtype=np.float64)
    ordered = canonical_order(points)
    assert not is_convex(ordered)


def test_canonical_order_starts_at_the_top_left_of_a_square():
    ordered = canonical_order([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert ordered[0] == pytest.approx([0.0, 0.0])
    assert polygon_area(ordered) == pytest.approx(1.0)
    assert is_convex(ordered)


def test_corner_matching_keeps_identity_when_labels_are_rotated():
    previous = SQUARE * 10
    current = np.array([previous[1], previous[0], previous[3], previous[2]], dtype=np.float64)
    matched = match_corners(previous, current)
    assert matched == pytest.approx(previous)


def test_validate_accepts_a_reasonable_quad_and_rejects_a_sliver():
    ok, reason = validate_quad(
        SQUARE * 100,
        image_area=200 * 200,
        min_area_ratio=0.05,
        max_area_ratio=0.9,
        min_corner_angle_deg=20,
        max_side_ratio=3,
    )
    assert ok, reason
    sliver = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    ok, reason = validate_quad(
        sliver,
        image_area=200 * 200,
        min_area_ratio=0.001,
        max_area_ratio=0.9,
        min_corner_angle_deg=20,
        max_side_ratio=4,
    )
    assert not ok
    assert reason == "extreme aspect"
