"""Eye Aspect Ratio (EAR) computation for blink/eye-closure detection.

Pure geometry — no OpenCV or MediaPipe dependencies — so this module
can be unit-tested with plain coordinate inputs.
"""
import math

# MediaPipe Face Mesh landmark indices for six points around each eye,
# ordered [outer_corner, top1, top2, inner_corner, bottom2, bottom1]
# following the standard EAR point layout (p1..p6).
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [263, 387, 385, 362, 380, 373]


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(points: list[tuple[float, float]]) -> float:
    """Compute EAR from six eye points ordered p1..p6.

    p1, p4 are the horizontal corners; p2, p3, p5, p6 are the lids.
    Returns a ratio: large when open (~0.3), small when closed (~0.1).
    """
    if len(points) != 6:
        raise ValueError(f"Expected 6 points, got {len(points)}")
    p1, p2, p3, p4, p5, p6 = points
    vertical = _distance(p2, p6) + _distance(p3, p5)
    horizontal = _distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / (2.0 * horizontal)


def extract_eye_points(
    landmarks, indices: list[int], image_width: int, image_height: int
) -> list[tuple[float, float]]:
    """Pull the six (x, y) pixel coords for one eye from a landmark list."""
    return [
        (landmarks[i].x * image_width, landmarks[i].y * image_height)
        for i in indices
    ]