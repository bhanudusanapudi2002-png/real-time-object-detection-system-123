"""
server.py
=========
FastAPI web server for the Real-Time Object Detection application.

Endpoints
---------
GET  /              -> Serves frontend/index.html
GET  /health        -> JSON health/status check
POST /detect/image  -> Upload image, returns annotated image + detection JSON
GET  /detect/webcam -> SSE stream of annotated webcam frames
GET  /classes       -> List all detectable class names

Run
---
uvicorn server:app --reload --port 8000
"""

import base64
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from detector import ObjectDetector


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once on startup; clean up on shutdown."""
    global detector
    detector = ObjectDetector(
        model_name="yolov8n",
        confidence_threshold=0.40,
        iou_threshold=0.45,
    )
    yield
    # Nothing to release for YOLO in-memory model


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VisionAI Object Detection API",
    description="YOLOv8-powered real-time object detection — image upload + webcam stream.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files at /static (avoids shadowing API routes)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Placeholder — populated by lifespan
detector: ObjectDetector = None  # type: ignore[assignment]

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_UPLOAD_MB  = 10


# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class Detection(BaseModel):
    label:      str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox:       BBox

class DetectionResponse(BaseModel):
    annotated_image: str          # base64-encoded JPEG
    detections:      list[Detection]
    count:           int
    inference_ms:    float
    image_width:     int
    image_height:    int
    model:           str = "yolov8n"

class HealthResponse(BaseModel):
    status:           str
    model:            str
    num_classes:      int
    conf_threshold:   float
    iou_threshold:    float
    version:          str = "2.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def frame_to_base64(frame: np.ndarray, quality: int = 88) -> str:
    """Encode a BGR numpy frame as a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def decode_upload(raw: bytes) -> np.ndarray:
    """Decode raw image bytes to a BGR numpy array."""
    arr   = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image data.")
    return frame


def build_detections(raw: list[dict]) -> list[Detection]:
    """Convert raw detection dicts to Pydantic Detection objects."""
    return [
        Detection(
            label      = d["label"],
            confidence = d["confidence"],
            bbox       = BBox(**d["bbox"]),
        )
        for d in raw
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root_index():
    """Serve the main frontend HTML."""
    return FileResponse("frontend/index.html")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Return model status and configuration."""
    return HealthResponse(
        status         = "ok",
        model          = "yolov8n",
        num_classes    = len(detector.class_names),
        conf_threshold = detector.confidence_threshold,
        iou_threshold  = detector.iou_threshold,
    )


@app.get("/classes", tags=["System"])
def list_classes():
    """Return all detectable class names with their IDs."""
    return JSONResponse({
        "count":   len(detector.class_names),
        "classes": [
            {"id": k, "name": v}
            for k, v in sorted(detector.class_names.items())
        ],
    })


@app.post("/detect/image", response_model=DetectionResponse, tags=["Detection"])
async def detect_image(
    image: UploadFile = File(..., description="Image to run detection on"),
):
    """
    Upload a JPEG / PNG / WEBP / BMP image.
    Returns an annotated image (base64) plus per-object detection data.
    """
    # Validate content type
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{image.content_type}'. "
                   f"Allowed: {sorted(ALLOWED_TYPES)}",
        )

    # Read + size guard
    raw = await image.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB.",
        )

    frame      = decode_upload(raw)
    h, w       = frame.shape[:2]

    t0         = time.perf_counter()
    annotated  = detector.detect(frame)
    raw_dets   = detector.get_raw_detections(frame)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return DetectionResponse(
        annotated_image = frame_to_base64(annotated),
        detections      = build_detections(raw_dets),
        count           = len(raw_dets),
        inference_ms    = elapsed_ms,
        image_width     = w,
        image_height    = h,
    )


@app.get("/detect/webcam", tags=["Detection"])
def webcam_stream(
    camera: int   = Query(0,    ge=0, description="Webcam device index"),
    conf:   float = Query(0.40, ge=0.05, le=0.95, description="Confidence threshold override"),
):
    """
    Server-Sent Events stream of annotated webcam frames (base64 JPEG).

    Each event payload is a raw base64 string.
    Errors are prefixed with 'ERROR:'.
    """
    def generate():
        original_conf = detector.confidence_threshold
        detector.confidence_threshold = conf

        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            yield f"data: ERROR:Cannot open camera at index {camera}\n\n"
            detector.confidence_threshold = original_conf
            return

        # Optimise: request 30 fps at 1280x720
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated = detector.detect(frame)
                b64       = frame_to_base64(annotated, quality=80)
                yield f"data: {b64}\n\n"
                time.sleep(0.033)   # ~30 fps cap
        finally:
            cap.release()
            detector.confidence_threshold = original_conf

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )