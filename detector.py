"""
detector.py
===========
Core object-detection logic for the Real-Time Object Detection system.

Responsibilities
----------------
* Load a YOLO model (any variant: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
* Run inference on a single frame / image
* Draw modern bounding boxes, labels, confidence bars, and corner-tick markers
* Expose a clean, reusable interface used by main.py and server.py

Design decisions
----------------
- Detection is encapsulated in a class so model loading happens only once.
- Drawing helpers are kept as private methods to keep the public API minimal.
- Confidence threshold and model name are constructor parameters.
- get_raw_detections() returns structured dicts without drawing anything,
  useful for the API layer to return JSON to the frontend.
"""

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Colour palette — distinct BGR colour per class (supports up to 256 classes)
# ---------------------------------------------------------------------------

def _build_colour_palette(num_classes: int = 256) -> list[tuple[int, int, int]]:
    """
    Generate visually distinct BGR colours using the HSV colour wheel.
    Spreading hues evenly ensures neighbouring class IDs look different.
    """
    colours: list[tuple[int, int, int]] = []
    for i in range(num_classes):
        hue = int(180 * i / num_classes)           # OpenCV hue: 0-179
        sat = 200 + (i % 2) * 55                   # alternate 200 / 255
        val = 220 + (i % 3) * 12                   # vary brightness slightly
        sat = min(sat, 255)
        val = min(val, 255)
        hsv_pixel = np.uint8([[[hue, sat, val]]])
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        colours.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colours


COLOUR_PALETTE = _build_colour_palette()


# ---------------------------------------------------------------------------
# ObjectDetector
# ---------------------------------------------------------------------------

class ObjectDetector:
    """
    Wraps a YOLO model and provides detect() and get_raw_detections() methods.

    Parameters
    ----------
    model_name : str
        Ultralytics model identifier, e.g. 'yolov8n', 'yolov8s', 'yolov8m'.
        The '.pt' suffix is appended automatically if missing.
    confidence_threshold : float
        Minimum confidence score (0-1) for a detection to be shown.
    iou_threshold : float
        IoU threshold for Non-Maximum Suppression. Lower = fewer overlapping boxes.
    device : str | None
        Inference device: 'cpu', 'cuda', 'mps', or None (auto-select).
    """

    def __init__(
        self,
        model_name:           str   = "yolov8n",
        confidence_threshold: float = 0.40,
        iou_threshold:        float = 0.45,
        device:               str | None = None,
    ) -> None:
        # Normalise model name
        if not model_name.endswith(".pt"):
            model_name = f"{model_name}.pt"

        print(f"[Detector] Loading model : {model_name}")
        self.model = YOLO(model_name)

        self.confidence_threshold = confidence_threshold
        self.iou_threshold        = iou_threshold
        self.device               = device   # None -> Ultralytics auto-picks

        # Class names dict {id: name} — 80 COCO classes by default
        self.class_names: dict[int, str] = self.model.names  # type: ignore[assignment]
        print(
            f"[Detector] Ready — {len(self.class_names)} classes  |  "
            f"conf >= {confidence_threshold:.0%}  |  iou <= {iou_threshold:.0%}"
        )

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
            A new BGR image with bounding boxes, labels, and scores drawn on it.
        """
        annotated = frame.copy()
        results   = self._run_inference(frame)
        detections = results[0]

        for box in detections.boxes:
            self._draw_detection(annotated, box)

        count = len(detections.boxes)
        self._draw_counter(annotated, count)

        return annotated

    def get_raw_detections(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference and return structured detection dicts (no drawing).

        Returns
        -------
        list[dict]  Each dict has keys: label, confidence, bbox (x1,y1,x2,y2).
        """
        results    = self._run_inference(frame)
        detections = results[0]
        out: list[dict] = []
        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            out.append({
                "label":      self.class_names.get(cls_id, f"class_{cls_id}"),
                "confidence": round(float(box.conf[0]), 4),
                "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_inference(self, frame: np.ndarray):
        """Run YOLO inference; returns raw results list."""
        return self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

    def _draw_detection(self, frame: np.ndarray, box) -> None:
        """Draw a modern bounding box + label pill on *frame* (in-place)."""
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id   = int(box.cls[0])
        confidence = float(box.conf[0])
        label      = self.class_names.get(class_id, f"class_{class_id}")
        colour     = COLOUR_PALETTE[class_id % len(COLOUR_PALETTE)]

        # ---- Semi-transparent box fill ----
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, cv2.FILLED)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        # ---- Bounding box outline ----
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # ---- Corner tick markers (premium look) ----
        tick = max(10, int(min(x2 - x1, y2 - y1) * 0.15))
        t    = 3
        for (cx, cy, sx, sy) in [
            (x1, y1,  1,  1),
            (x2, y1, -1,  1),
            (x1, y2,  1, -1),
            (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + sx * tick, cy), colour, t)
            cv2.line(frame, (cx, cy), (cx, cy + sy * tick), colour, t)

        # ---- Label pill ----
        text       = f"{label}  {confidence:.0%}"
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        font_thick = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thick)

        pad_x, pad_y = 8, 5
        pill_x1 = x1
        pill_y1 = max(y1 - text_h - baseline - pad_y * 2, 0)
        pill_x2 = x1 + text_w + pad_x * 2
        pill_y2 = y1

        # Rounded rectangle background
        _filled_rounded_rect(frame, pill_x1, pill_y1, pill_x2, pill_y2, colour, radius=5)

        # Auto-contrast text (white on dark, black on bright)
        b, g, r   = colour
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        text_colour = (0, 0, 0) if brightness > 140 else (255, 255, 255)

        cv2.putText(
            frame, text,
            (pill_x1 + pad_x, pill_y2 - baseline - 2),
            font, font_scale, text_colour, font_thick, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_counter(frame: np.ndarray, count: int) -> None:
        """Show total object count pill in the top-right corner."""
        text  = f"Objects: {count}"
        font  = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.6
        thick = 1
        (w, h), _ = cv2.getTextSize(text, font, scale, thick)
        margin = 10
        x = frame.shape[1] - w - margin * 2
        y = margin
        _filled_rounded_rect(frame, x, y, x + w + margin * 2, y + h + margin,
                             (20, 20, 30), radius=7)
        cv2.putText(frame, text, (x + margin, y + h + 2),
                    font, scale, (190, 220, 255), thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Drawing utility
# ---------------------------------------------------------------------------

def _filled_rounded_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    colour: tuple[int, int, int],
    radius: int = 8,
) -> None:
    """Draw a solid filled rounded rectangle on *img* in-place."""
    # Clamp coords to image bounds
    H, W = img.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, W - 1), min(y2, H - 1)
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r < 1:
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, cv2.FILLED)
        return

    # Fill body
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), colour, cv2.FILLED)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), colour, cv2.FILLED)

    # Four corner circles
    for (cx, cy) in [(x1 + r, y1 + r), (x2 - r, y1 + r),
                     (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(img, (cx, cy), r, colour, cv2.FILLED)