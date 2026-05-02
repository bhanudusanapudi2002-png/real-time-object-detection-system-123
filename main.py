"""
main.py
=======
Entry point for the Real-Time Object Detection application.

Usage
-----
# Webcam mode (default)
python main.py

# Image mode
python main.py --mode image --source path/to/image.jpg

# Custom model + confidence threshold
python main.py --model yolov8s --conf 0.50

# Full options
python main.py --help
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from detector import ObjectDetector

# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Define and return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="ObjectDetection",
        description="Real-time object detection with YOLOv8 + OpenCV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                  # webcam, yolov8n, conf=0.40
  python main.py --mode image --source dog.jpg    # detect in an image
  python main.py --model yolov8s --conf 0.55      # heavier model, higher threshold
  python main.py --camera 1                       # use second webcam (index 1)
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["webcam", "image"],
        default="webcam",
        help="Input mode: 'webcam' for live feed, 'image' for a single file. "
             "(default: webcam)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to the input image when --mode image is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        help=(
            "YOLO model variant to use (yolov8n/yolov8s/...) or a full path"
            " to a custom weights file (e.g. C:\\path\\to\\best.pt)."
            " (default: yolov8n)"
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Minimum confidence threshold (0-1) for displayed detections. "
             "(default: 0.40)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index. 0 = primary camera. (default: 0)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the annotated output (image or video) to disk.",
    )

    return parser


# ---------------------------------------------------------------------------
# FPS tracker
# ---------------------------------------------------------------------------

class FPSTracker:
    """Lightweight rolling-average FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._times: list[float] = []
        self._window = window

    def tick(self) -> float:
        """Call once per frame; returns current average FPS."""
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Overlay FPS in the top-left corner."""
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.65, 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    # Dark background
    cv2.rectangle(frame, (4, 4), (w + 14, h + 20), (30, 30, 30), cv2.FILLED)
    cv2.putText(frame, text, (10, h + 12), font, scale,
                (100, 255, 180), thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Webcam mode
# ---------------------------------------------------------------------------

def run_webcam(detector: ObjectDetector, camera_idx: int, save: bool) -> None:
    """
    Open the webcam and run continuous object detection.

    Controls
    --------
    Q  /  ESC  — quit
    S           — save current frame as 'snapshot.jpg'
    +  /  =     — increase confidence threshold by 5 %
    -           — decrease confidence threshold by 5 %
    """
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        sys.exit(
            f"[Error] Cannot open camera at index {camera_idx}. "
            "Check the --camera argument."
        )

    # Read one frame to get dimensions
    ok, frame = cap.read()
    if not ok:
        sys.exit("[Error] Failed to read from camera.")

    frame_h, frame_w = frame.shape[:2]
    print(f"[Webcam] Resolution: {frame_w}×{frame_h}  |  Press Q or ESC to quit.")

    # Optional video writer
    writer: cv2.VideoWriter | None = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output_webcam.mp4", fourcc, 25.0,
                                 (frame_w, frame_h))
        print("[Webcam] Recording to output_webcam.mp4")

    fps_tracker = FPSTracker()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Webcam] Frame grab failed — stopping.")
            break

        # --- Detection ---
        annotated = detector.detect(frame)

        # --- FPS overlay ---
        fps = fps_tracker.tick()
        draw_fps(annotated, fps)

        # --- Hot-key help overlay (bottom-left) ---
        _draw_help(annotated, detector.confidence_threshold)

        if writer:
            writer.write(annotated)

        cv2.imshow("Object Detection  |  Q / ESC = quit", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):               # Q or ESC
            break
        elif key == ord("s"):                    # S — snapshot
            cv2.imwrite("snapshot.jpg", annotated)
            print("[Webcam] Snapshot saved as snapshot.jpg")
        elif key in (ord("+"), ord("=")):        # Increase confidence
            detector.confidence_threshold = min(
                detector.confidence_threshold + 0.05, 0.95
            )
            print(f"[Webcam] Confidence → {detector.confidence_threshold:.0%}")
        elif key == ord("-"):                    # Decrease confidence
            detector.confidence_threshold = max(
                detector.confidence_threshold - 0.05, 0.05
            )
            print(f"[Webcam] Confidence → {detector.confidence_threshold:.0%}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[Webcam] Session ended.")


# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------

def run_image(detector: ObjectDetector, source: str, save: bool) -> None:
    """
    Run detection on a single image file and display the result.

    Controls
    --------
    Any key — close window
    S       — save annotated image
    """
    path = Path(source)
    if not path.is_file():
        sys.exit(f"[Error] Image not found: {source}")

    frame = cv2.imread(str(path))
    if frame is None:
        sys.exit(f"[Error] OpenCV could not read image: {source}")

    print(f"[Image] Processing: {path.name}  ({frame.shape[1]}×{frame.shape[0]})")

    start = time.perf_counter()
    annotated = detector.detect(frame)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"[Image] Inference time: {elapsed_ms:.1f} ms")

    if save:
        out_path = path.with_name(f"{path.stem}_detected{path.suffix}")
        cv2.imwrite(str(out_path), annotated)
        print(f"[Image] Saved annotated image → {out_path}")

    # Display
    # Resize for display if the image is very large (max 1200 px wide)
    display = _fit_to_screen(annotated, max_width=1200, max_height=800)
    cv2.imshow(f"Detection — {path.name}  |  any key to close", display)
    key = cv2.waitKey(0) & 0xFF

    if key == ord("s") and not save:
        out_path = path.with_name(f"{path.stem}_detected{path.suffix}")
        cv2.imwrite(str(out_path), annotated)
        print(f"[Image] Saved annotated image → {out_path}")

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _fit_to_screen(
    frame: np.ndarray, max_width: int = 1200, max_height: int = 800
) -> np.ndarray:
    """Downscale *frame* so it fits within max_width × max_height (keeps AR)."""
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def _draw_help(frame: np.ndarray, conf: float) -> None:
    """Print key-bindings in the bottom-left corner of *frame* (in-place)."""
    lines = [
        f"Conf: {conf:.0%}  |  +/- adjust",
        "S = snapshot  |  Q/ESC = quit",
    ]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    y0 = frame.shape[0] - 8
    for line in reversed(lines):
        (w, h), _ = cv2.getTextSize(line, font, scale, thick)
        cv2.rectangle(frame,
                      (4, y0 - h - 4), (w + 10, y0 + 2),
                      (30, 30, 30), cv2.FILLED)
        cv2.putText(frame, line, (8, y0 - 2),
                    font, scale, (210, 210, 210), thick, cv2.LINE_AA)
        y0 -= h + 10


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Validate image mode arguments
    if args.mode == "image" and args.source is None:
        parser.error("--source is required when --mode image is used.")

    # Validate confidence range
    if not (0.01 <= args.conf <= 0.99):
        parser.error("--conf must be between 0.01 and 0.99.")

    print("=" * 56)
    print("  Real-Time Object Detection  |  YOLOv8 + OpenCV")
    print("=" * 56)
    print(f"  Mode   : {args.mode}")
    print(f"  Model  : {args.model}")
    print(f"  Conf   : {args.conf:.0%}")
    if args.mode == "webcam":
        print(f"  Camera : index {args.camera}")
    else:
        print(f"  Source : {args.source}")
    print("=" * 56)

    # Initialise detector (downloads weights on first run — ~6 MB for yolov8n)
    detector = ObjectDetector(
        model_name=args.model,
        confidence_threshold=args.conf,
    )

    if args.mode == "webcam":
        run_webcam(detector, camera_idx=args.camera, save=args.save)
    else:
        run_image(detector, source=args.source, save=args.save)


if __name__ == "__main__":
    main()