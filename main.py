"""
main.py
=======
CLI entry point for the Real-Time Object Detection application.

Usage
-----
# Webcam mode (default)
python main.py

# Image mode
python main.py --mode image --source path/to/image.jpg

# Custom model + confidence threshold
python main.py --model yolov8s --conf 0.50 --iou 0.45

# Save output + use second camera
python main.py --camera 1 --save

# Full help
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
        prog="VisionAI",
        description="Real-time object detection with YOLOv8 + OpenCV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # webcam, yolov8n, conf=0.40
  python main.py --mode image --source dog.jpg      # detect in a single image
  python main.py --model yolov8s --conf 0.55        # heavier model
  python main.py --camera 1 --save                  # second webcam, record video
  python main.py --mode image --source img.jpg --iou 0.40 --save
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
            "YOLO model variant (yolov8n / yolov8s / yolov8m / yolov8l / yolov8x) "
            "or a full path to custom weights (e.g. runs/train/best.pt). "
            "(default: yolov8n)"
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Minimum confidence threshold 0-1. (default: 0.40)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS 0-1. Lower = fewer overlapping boxes. (default: 0.45)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (0 = primary camera). (default: 0)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output to disk (snapshot.jpg / output_webcam.mp4).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested webcam frame width in pixels. (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested webcam frame height in pixels. (default: 720)",
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
        """Call once per frame; returns the rolling-average FPS."""
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# On-frame HUD overlays
# ---------------------------------------------------------------------------

def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Overlay FPS counter in the top-left corner."""
    text  = f"FPS: {fps:.1f}"
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.6
    thick = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    # Dark pill background
    cv2.rectangle(frame, (6, 6), (w + 18, h + 20), (15, 15, 20), cv2.FILLED)
    cv2.rectangle(frame, (6, 6), (w + 18, h + 20), (60, 60, 80), 1)
    cv2.putText(frame, text, (12, h + 14), font, scale, (100, 255, 180), thick, cv2.LINE_AA)


def _draw_help(frame: np.ndarray, conf: float) -> None:
    """Print key-bindings overlay in the bottom-left corner of *frame*."""
    lines = [
        f"Conf: {conf:.0%}  |  [+/-] adjust",
        "[S] snapshot   [R] record   [Q/ESC] quit",
    ]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.44
    thick = 1
    y0 = frame.shape[0] - 10
    for line in reversed(lines):
        (w, h), _ = cv2.getTextSize(line, font, scale, thick)
        cv2.rectangle(frame, (4, y0 - h - 5), (w + 12, y0 + 3), (15, 15, 20), cv2.FILLED)
        cv2.rectangle(frame, (4, y0 - h - 5), (w + 12, y0 + 3), (60, 60, 80), 1)
        cv2.putText(frame, line, (8, y0 - 2),
                    font, scale, (200, 200, 215), thick, cv2.LINE_AA)
        y0 -= h + 12


def _draw_banner(frame: np.ndarray, text: str, duration_frames: int,
                 frame_num: int) -> None:
    """Flash a centred text banner for *duration_frames* frames."""
    if frame_num >= duration_frames:
        return
    alpha = min(1.0, (duration_frames - frame_num) / 20)
    overlay = frame.copy()
    h_fr, w_fr = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.8
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = (w_fr - tw) // 2
    ty = h_fr // 2
    cv2.rectangle(overlay, (tx - 14, ty - th - 10), (tx + tw + 14, ty + 10), (15, 15, 20), cv2.FILLED)
    cv2.putText(overlay, text, (tx, ty), font, scale, (100, 255, 180), thick, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ---------------------------------------------------------------------------
# Webcam mode
# ---------------------------------------------------------------------------

def run_webcam(
    detector:   ObjectDetector,
    camera_idx: int,
    save:       bool,
    width:      int = 1280,
    height:     int = 720,
) -> None:
    """
    Open the webcam and run continuous object detection.

    Hot-keys
    --------
    Q / ESC  — quit
    S        — save current frame as snapshot_<timestamp>.jpg
    R        — toggle video recording on/off
    +  / =   — raise confidence threshold by 5 %
    -        — lower confidence threshold by 5 %
    """
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        sys.exit(
            f"[Error] Cannot open camera at index {camera_idx}. "
            "Try a different --camera value."
        )

    # Request resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ok, frame = cap.read()
    if not ok:
        sys.exit("[Error] Failed to read from camera.")

    frame_h, frame_w = frame.shape[:2]
    print(f"[Webcam] Resolution : {frame_w}x{frame_h}")
    print(f"[Webcam] Hot-keys   : Q/ESC=quit  S=snapshot  R=record  +/-=confidence")

    # Optional video writer
    writer: cv2.VideoWriter | None = None
    recording = False

    if save:
        writer, recording = _start_recording(frame_w, frame_h)

    fps_tracker  = FPSTracker()
    banner_text  = ""
    banner_timer = 0
    frame_num    = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Webcam] Frame grab failed — stopping.")
            break

        annotated = detector.detect(frame)

        fps = fps_tracker.tick()
        draw_fps(annotated, fps)
        _draw_help(annotated, detector.confidence_threshold)

        # Recording indicator
        if recording:
            cv2.circle(annotated, (annotated.shape[1] - 24, 24), 9, (0, 0, 220), cv2.FILLED)

        if writer and recording:
            writer.write(annotated)

        # Flash banner
        if banner_text:
            _draw_banner(annotated, banner_text, 60, frame_num)
            frame_num += 1
            if frame_num >= 60:
                banner_text = ""
                frame_num   = 0

        cv2.imshow("VisionAI  |  Object Detection  |  Q/ESC = quit", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):                   # Q or ESC — quit
            break

        elif key == ord("s"):                        # S — snapshot
            ts   = time.strftime("%Y%m%d_%H%M%S")
            name = f"snapshot_{ts}.jpg"
            cv2.imwrite(name, annotated)
            print(f"[Webcam] Snapshot saved: {name}")
            banner_text = f"Snapshot saved: {name}"
            frame_num   = 0

        elif key == ord("r"):                        # R — toggle recording
            if not recording:
                writer, recording = _start_recording(frame_w, frame_h)
                banner_text = "Recording started"
            else:
                recording = False
                if writer:
                    writer.release()
                    writer = None
                banner_text = "Recording stopped"
            frame_num = 0

        elif key in (ord("+"), ord("=")):            # + — raise conf
            detector.confidence_threshold = min(detector.confidence_threshold + 0.05, 0.95)
            banner_text = f"Confidence: {detector.confidence_threshold:.0%}"
            frame_num   = 0
            print(f"[Webcam] Confidence -> {detector.confidence_threshold:.0%}")

        elif key == ord("-"):                        # - — lower conf
            detector.confidence_threshold = max(detector.confidence_threshold - 0.05, 0.05)
            banner_text = f"Confidence: {detector.confidence_threshold:.0%}"
            frame_num   = 0
            print(f"[Webcam] Confidence -> {detector.confidence_threshold:.0%}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[Webcam] Session ended.")


def _start_recording(
    w: int, h: int
) -> tuple[cv2.VideoWriter, bool]:
    """Create a VideoWriter and return (writer, True)."""
    ts     = time.strftime("%Y%m%d_%H%M%S")
    name   = f"output_{ts}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(name, fourcc, 25.0, (w, h))
    print(f"[Webcam] Recording -> {name}")
    return writer, True


# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------

def run_image(detector: ObjectDetector, source: str, save: bool) -> None:
    """
    Run detection on a single image and display the annotated result.

    Hot-keys
    --------
    S       — save annotated image
    Any key — close window
    """
    path = Path(source)
    if not path.is_file():
        sys.exit(f"[Error] Image not found: {source}")

    frame = cv2.imread(str(path))
    if frame is None:
        sys.exit(f"[Error] OpenCV could not read: {source}")

    print(f"[Image] Input  : {path.name}  ({frame.shape[1]}x{frame.shape[0]})")

    t0         = time.perf_counter()
    annotated  = detector.detect(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    raw_dets = detector.get_raw_detections(frame)
    print(f"[Image] Inference : {elapsed_ms:.1f} ms")
    print(f"[Image] Detections: {len(raw_dets)}")
    for d in raw_dets:
        b = d["bbox"]
        print(f"         {d['label']:<20s} {d['confidence']:.0%}  "
              f"  box=({b['x1']},{b['y1']},{b['x2']},{b['y2']})")

    if save:
        out_path = path.with_name(f"{path.stem}_detected{path.suffix}")
        cv2.imwrite(str(out_path), annotated)
        print(f"[Image] Saved -> {out_path}")

    display = _fit_to_screen(annotated, max_width=1280, max_height=800)
    window  = f"VisionAI — {path.name}  |  S=save  any key=close"
    cv2.imshow(window, display)
    key = cv2.waitKey(0) & 0xFF

    if key == ord("s") and not save:
        out_path = path.with_name(f"{path.stem}_detected{path.suffix}")
        cv2.imwrite(str(out_path), annotated)
        print(f"[Image] Saved -> {out_path}")

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _fit_to_screen(
    frame: np.ndarray, max_width: int = 1280, max_height: int = 800
) -> np.ndarray:
    """Downscale *frame* to fit within max_width x max_height, preserving aspect ratio."""
    h, w  = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.mode == "image" and args.source is None:
        parser.error("--source is required when --mode image is used.")

    if not (0.01 <= args.conf <= 0.99):
        parser.error("--conf must be between 0.01 and 0.99.")

    if not (0.01 <= args.iou <= 0.99):
        parser.error("--iou must be between 0.01 and 0.99.")

    # Print startup banner
    sep = "=" * 58
    print(sep)
    print("  VisionAI  |  Real-Time Object Detection  |  YOLOv8")
    print(sep)
    print(f"  Mode   : {args.mode}")
    print(f"  Model  : {args.model}")
    print(f"  Conf   : {args.conf:.0%}")
    print(f"  IoU    : {args.iou:.0%}")
    if args.mode == "webcam":
        print(f"  Camera : index {args.camera}  ({args.width}x{args.height})")
    else:
        print(f"  Source : {args.source}")
    print(sep)

    detector = ObjectDetector(
        model_name=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
    )

    if args.mode == "webcam":
        run_webcam(
            detector,
            camera_idx=args.camera,
            save=args.save,
            width=args.width,
            height=args.height,
        )
    else:
        run_image(detector, source=args.source, save=args.save)


if __name__ == "__main__":
    main()