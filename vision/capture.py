"""Webcam capture and display for StudyBuddy."""
import time
import cv2


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
        """Main loop: read frames, overlay FPS, show window, quit on 'q'."""
        if self.cap is None:
            raise RuntimeError("Call open() before run()")

        prev_time = time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("Failed to read frame; stopping.")
                break

            # Compute FPS from time between frames
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

            cv2.imshow("StudyBuddy - Webcam", frame)

            # waitKey(1) waits 1ms for a keypress; & 0xFF for cross-platform safety
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

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