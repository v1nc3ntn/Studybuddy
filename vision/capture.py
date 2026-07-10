"""Webcam capture and display for StudyBuddy."""
import time

import cv2

from vision.face_mesh import FaceMeshDetector
from scoring.eye_aspect_ratio import (
    eye_aspect_ratio,
    extract_eye_points,
    LEFT_EYE,
    RIGHT_EYE,
)
from scoring.head_pose import estimate_head_pose


class WebcamCapture:
    """Wraps an OpenCV webcam stream with a live FPS counter."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the webcam. Raises RuntimeError if it can't be accessed."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {self.camera_index}")

    def run(self) -> None:
        """Main loop: read frames, run detection, overlay readouts, quit on 'q'."""
        if self.cap is None:
            raise RuntimeError("Call open() before run()")

        detector = FaceMeshDetector()
        prev_time = time.time()
        start_time = time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("Failed to read frame; stopping.")
                break

            timestamp_ms = int((time.time() - start_time) * 1000)
            result = detector.process(frame, timestamp_ms)
            frame = detector.draw(frame, result)

            # Default readouts (shown when no face is present)
            ear_text = "EAR: --"
            pose_text = "Pose: --"

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                h, w = frame.shape[:2]

                # Eye aspect ratio
                left_pts = extract_eye_points(landmarks, LEFT_EYE, w, h)
                right_pts = extract_eye_points(landmarks, RIGHT_EYE, w, h)
                ear = (eye_aspect_ratio(left_pts) + eye_aspect_ratio(right_pts)) / 2.0
                state = "CLOSED" if ear < 0.20 else "open"
                ear_text = f"EAR: {ear:.3f} ({state})"

                # Head pose
                pose = estimate_head_pose(landmarks, w, h)
                if pose is not None:
                    pitch, yaw, roll = pose
                    looking = "away" if abs(yaw) > 20 else "forward"
                    pose_text = f"Yaw: {yaw:+.0f} ({looking})"

            # FPS
            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                ear_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                pose_text,
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 200, 0),
                2,
            )

            cv2.imshow("StudyBuddy - Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        detector.close()
        self.release()

    def release(self) -> None:
        """Release the webcam and close windows."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = WebcamCapture()
    cam.open()
    cam.run()