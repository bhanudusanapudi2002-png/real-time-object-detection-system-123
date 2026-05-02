import base64
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from detector import ObjectDetector

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Object Detection API",
    description="YOLOv8-powered object detection — image upload + webcam stream",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files from the `frontend/` folder so the UI and API
# are available from the same origin (useful for local testing).
# Serve frontend files under /static to avoid shadowing API routes.
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def root_index():
    # Serve the main index.html from the frontend directory
    return FileResponse("frontend/index.html")

# Load model once at startup
detector = ObjectDetector(model_name="yolov8n", confidence_threshold=0.40)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


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
    confidence: float
    bbox:       BBox

class DetectionResponse(BaseModel):
    annotated_image: str
    detections:      list[Detection]
    count:           int
    inference_ms:    float
    image_width:     int
    image_height:    int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def frame_to_base64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode("utf-8")


def extract_detections(frame: np.ndarray) -> list[Detection]:
    results = detector.model.predict(
        frame, conf=detector.confidence_threshold, verbose=False
    )
    out = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        out.append(Detection(
            label      = detector.class_names.get(cls_id, f"class_{cls_id}"),
            confidence = round(float(box.conf[0]), 3),
            bbox       = BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        ))
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "model":   "yolov8n",
        "classes": len(detector.class_names),
    }


@app.post("/detect/image", response_model=DetectionResponse)
async def detect_image(image: UploadFile = File(...)):
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type '{image.content_type}'. Allowed: {ALLOWED_TYPES}",
        )

    raw   = await image.read()
    arr   = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    h, w = frame.shape[:2]

    t0         = time.perf_counter()
    annotated  = detector.detect(frame)
    detections = extract_detections(frame)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return DetectionResponse(
        annotated_image = frame_to_base64(annotated),
        detections      = detections,
        count           = len(detections),
        inference_ms    = elapsed_ms,
        image_width     = w,
        image_height    = h,
    )


@app.get("/detect/webcam")
def webcam_stream():
    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            yield "data: ERROR:Cannot open webcam\n\n"
            return
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated = detector.detect(frame)
                b64       = frame_to_base64(annotated)
                yield f"data: {b64}\n\n"
                time.sleep(0.04)
        finally:
            cap.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )