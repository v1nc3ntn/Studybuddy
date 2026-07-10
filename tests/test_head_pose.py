"""Unit tests for head pose angle extraction."""
import math

import numpy as np

from scoring.head_pose import rotation_matrix_to_angles


def test_identity_matrix_is_zero_angles():
    """No rotation (identity matrix) should give ~0 for all angles."""
    identity = np.eye(3)
    pitch, yaw, roll = rotation_matrix_to_angles(identity)
    assert abs(pitch) < 1e-6
    assert abs(yaw) < 1e-6
    assert abs(roll) < 1e-6


def test_yaw_90_degrees():
    """A 90-degree rotation about the vertical (Y) axis -> yaw ~90."""
    angle = math.radians(90)
    ry = np.array([
        [math.cos(angle), 0, math.sin(angle)],
        [0, 1, 0],
        [-math.sin(angle), 0, math.cos(angle)],
    ])
    _, yaw, _ = rotation_matrix_to_angles(ry)
    assert abs(abs(yaw) - 90) < 1.0


def test_roll_45_degrees():
    """A 45-degree rotation about the Z axis -> roll ~45."""
    angle = math.radians(45)
    rz = np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1],
    ])
    _, _, roll = rotation_matrix_to_angles(rz)
    assert abs(roll - 45) < 1.0