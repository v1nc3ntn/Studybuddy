"""Pure rotation-matrix geometry for head pose.

No OpenCV dependency — only math and numpy — so this module is fully
unit-testable without a computer-vision stack installed.
"""
import math

import numpy as np


def rotation_matrix_to_angles(rmat: np.ndarray) -> tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to (pitch, yaw, roll) in degrees."""
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)