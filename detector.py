"""
detector.py
===========
Core object-detection logic.

Responsibilities
----------------
* Load a YOLO model (any variant: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
* Run inference on a single frame / image
* Draw bounding boxes, labels and confidence scores on the frame
* Expose a clean, reusable interface that main.py can call

Design decisions
----------------
- Detection is encapsulated in a class so model loading happens only once.
- Drawing helpers are kept as private methods to keep the public API minimal.
- Confidence threshold and model name are constructor parameters so they can
  be changed without touching any other file.
"""

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Colour palette — one distinct BGR colour per COCO class (80 classes)
# ---------------------------------------------------------------------------
def _build_colour_palette(num_classes: int = 80) -> list[tuple[int, int, int]]:
    """
    Generate visually distinct BGR colours using the HSV colour wheel.
    Spreading hues evenly ensures neighbouring class IDs look different.
    """
    colours: list[tuple[int, int, int]] = []
    for i in range(num_classes):
        hue = int(180 * i / num_classes)          # OpenCV hue: 0-179
        hsv_pixel = np.uint8([[[hue, 220, 255]]])  # High saturation & value
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        colours.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colours


COLOUR_PALETTE = _build_colour_palette()


# ---------------------------------------------------------------------------
# ObjectDetector
# ---------------------------------------------------------------------------
class ObjectDetector:
    """
    Wraps a YOLO model and provides a single `detect()` method.

    Parameters
    ----------
    model_name : str
        Ultralytics model identifier, e.g. 'yolov8n', 'yolov8s', 'yolov8m'.
        The '.pt' suffix is appended automatically if missing.
    confidence_threshold : float
        Minimum confidence score (0-1) for a detection to be shown.
    device : str | None
        Inference device: 'cpu', 'cuda', 'mps', or None (auto-select).
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        confidence_threshold: float = 0.15,
        device: str | None = None,
    ) -> None:
        # Normalise model name
        if not model_name.endswith(".pt"):
            model_name = f"{model_name}.pt"

        print(f"[Detector] Loading model: {model_name}")
        self.model = YOLO(model_name)

        self.confidence_threshold = confidence_threshold
        self.device = device  # None → Ultralytics picks the best device

        # Class names come from the model itself (80 COCO classes by default)
        self.class_names: list[str] = self.model.names  # type: ignore[assignment]
        print(f"[Detector] Ready — {len(self.class_names)} classes, "
              f"confidence ≥ {confidence_threshold:.0%}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run object detection on *frame* and return an annotated copy.

        Parameters
        ----------
        frame : np.ndarray
            BGR image as returned by cv2.VideoCapture.read() or cv2.imread().

        Returns
        -------
        np.ndarray
            A new BGR image with bounding boxes, labels and scores drawn on it.
        """
        annotated = frame.copy()

        # Run inference (verbose=False suppresses per-frame console output)
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        # results is a list with one element per image; we only pass one frame
        detections = results[0]

        for box in detections.boxes:
            self._draw_detection(annotated, box)

        # Overlay detection count
        count = len(detections.boxes)
        self._draw_counter(annotated, count)

        return annotated

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_detection(self, frame: np.ndarray, box) -> None:
        """Draw a single bounding box + label on *frame* (in-place)."""

        # --- Bounding box coordinates (pixel integers) ---
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # --- Class info ---
        class_id   = int(box.cls[0])
        confidence  = float(box.conf[0])
        label       = self.class_names.get(class_id, f"class_{class_id}")
        colour      = COLOUR_PALETTE[class_id % len(COLOUR_PALETTE)]

        # --- Draw rectangle ---
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

        # --- Build label string ---
        text = f"{label}  {confidence:.0%}"

        # --- Label background (filled rectangle) ---
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thick = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, font_scale, font_thick
        )
        label_x1 = x1
        label_y1 = max(y1 - text_h - baseline - 6, 0)
        label_x2 = x1 + text_w + 6
        label_y2 = y1

        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2),
                      colour, cv2.FILLED)

        # --- Contrasting text colour (white or black) ---
        brightness = 0.299 * colour[2] + 0.587 * colour[1] + 0.114 * colour[0]
        text_colour = (0, 0, 0) if brightness > 128 else (255, 255, 255)

        cv2.putText(
            frame, text,
            (label_x1 + 3, label_y2 - baseline - 2),
            font, font_scale, text_colour, font_thick,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_counter(frame: np.ndarray, count: int) -> None:
        """Show total object count in the top-right corner."""
        text = f"Objects: {count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.65, 2
        (w, h), _ = cv2.getTextSize(text, font, scale, thick)
        x = frame.shape[1] - w - 12
        y = h + 12
        # Dark background pill
        cv2.rectangle(frame, (x - 6, 4), (frame.shape[1] - 4, h + 20),
                      (30, 30, 30), cv2.FILLED)
        cv2.putText(frame, text, (x, y), font, scale,
                    (200, 230, 255), thick, cv2.LINE_AA)