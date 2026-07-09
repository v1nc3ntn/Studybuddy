"""Unit tests for the eye aspect ratio computation."""
import pytest

from scoring.eye_aspect_ratio import eye_aspect_ratio


def test_open_eye_has_high_ratio():
    """A tall eye shape should give a ratio around 0.3+."""
    points = [
        (0.0, 0.0),   # p1 left corner
        (0.3, 0.3),   # p2 top
        (0.7, 0.3),   # p3 top
        (1.0, 0.0),   # p4 right corner
        (0.7, -0.3),  # p5 bottom
        (0.3, -0.3),  # p6 bottom
    ]
    assert eye_aspect_ratio(points) > 0.25


def test_closed_eye_has_low_ratio():
    """A flat eye shape (lids together) should give a near-zero ratio."""
    points = [
        (0.0, 0.0),
        (0.3, 0.01),
        (0.7, 0.01),
        (1.0, 0.0),
        (0.7, -0.01),
        (0.3, -0.01),
    ]
    assert eye_aspect_ratio(points) < 0.10


def test_wrong_number_of_points_raises():
    """Passing the wrong number of points should raise ValueError."""
    with pytest.raises(ValueError):
        eye_aspect_ratio([(0.0, 0.0), (1.0, 1.0)])


def test_zero_width_returns_zero():
    """Degenerate eye with zero horizontal distance returns 0, not a crash."""
    points = [
        (0.5, 0.0),
        (0.5, 0.3),
        (0.5, 0.3),
        (0.5, 0.0),
        (0.5, -0.3),
        (0.5, -0.3),
    ]
    assert eye_aspect_ratio(points) == 0.0