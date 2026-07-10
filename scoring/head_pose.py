"""Head pose estimation via solvePnP.

Estimates yaw/pitch/roll of the head from facial landmarks. The angle
extraction from a rotation matrix is a pure function (unit-tested);
the solvePnP call requires OpenCV.
"""
import math

import cv2
import numpy as np

# MediaPipe landmark indices for the six points solvePnP needs.
NOSE_TIP = 1
CHIN = 199
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

POSE_LANDMARKS = [
    NOSE_TIP,
    CHIN,
    LEFT_EYE_CORNER,
    RIGHT_EYE_CORNER,
    LEFT_MOUTH_CORNER,
    RIGHT_MOUTH_CORNER,
]

# Canonical 3D face model (arbitrary reference frame, millimetres).
# These are idealised positions of the six points on an average face.
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),         # Nose tip
        (0.0, -330.0, -65.0),    # Chin
        (-225.0, 170.0, -135.0), # Left eye corner
        (225.0, 170.0, -135.0),  # Right eye corner
        (-150.0, -150.0, -125.0),# Left mouth corner
        (150.0, -150.0, -125.0), # Right mouth corner
    ],
    dtype=np.float64,
)


def rotation_matrix_to_angles(rmat: np.ndarray) -> tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to (pitch, yaw, roll) in degrees.

    Pure function — no OpenCV. This is the unit-tested core.
    """
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


def estimate_head_pose(
    landmarks, image_width: int, image_height: int
) -> tuple[float, float, float] | None:
    """Estimate (pitch, yaw, roll) in degrees from face landmarks.

    Returns None if pose can't be solved. Requires OpenCV.
    """
    image_points = np.array(
        [
            (landmarks[i].x * image_width, landmarks[i].y * image_height)
            for i in POSE_LANDMARKS
        ],
        dtype=np.float64,
    )

    # Approximate camera intrinsics: focal length ~ image width,
    # principal point at image centre, no lens distortion.
    focal_length = float(image_width)
    center = (image_width / 2.0, image_height / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rotation_vector)
    return rotation_matrix_to_angles(rmat)