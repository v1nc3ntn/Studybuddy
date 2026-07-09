"""MediaPipe Face Landmarker wrapper for StudyBuddy (Tasks API)."""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


class FaceMeshDetector:
    """Detects facial landmarks using MediaPipe's Face Landmarker task."""

    def __init__(
        self,
        model_path: str = "models/face_landmarker.task",
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def process(self, frame: np.ndarray, timestamp_ms: int):
        """Run detection on a BGR frame. Returns a FaceLandmarkerResult."""
        # Guarantee strictly increasing timestamps (VIDEO mode requires it)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def draw(self, frame: np.ndarray, result) -> np.ndarray:
        """Draw each landmark as a small green dot, in place."""
        if not result.face_landmarks:
            return frame
        h, w = frame.shape[:2]
        for face_landmarks in result.face_landmarks:
            for lm in face_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        return frame

    def close(self) -> None:
        """Release the landmarker."""
        self.landmarker.close()