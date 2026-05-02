# ObjectDetection

YOLOv8-based object detection project focused on earphones/earbuds (demo + training utilities).

Contents
- `detector.py` — `ObjectDetector` class: model load, inference, and drawing helpers.
- `main.py` — CLI for webcam/image demo and quick local runs.
- `server.py` — FastAPI `app` exposing image upload and webcam stream endpoints (serve with `uvicorn`).
- `yolov8n.pt` — example model weights (replace with your own if desired).
- `training/` — training and data tooling (`train.py`, `evaluate.py`, `auto_label.py`, `fix_labels.py`, `scraper.py`).
- `data/labeled/` — YOLO-format labeled data and `dataset.yaml`.
- `frontend/index.html` — small web UI for uploads / webcam viewing.

Requirements
- Python 3.8+ (3.10+ recommended)
- See `requirements.txt` for exact dependency pins. Key packages: `ultralytics`, `torch`, `opencv-python`, `fastapi`, `uvicorn`.

Quick install (Windows PowerShell)
```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Quickstart — Run detection locally
- Webcam demo:
```powershell
python main.py --mode webcam
```
- Run on a single image:
```powershell
python main.py --mode image --source path\to\image.jpg --model yolov8n.pt --conf 0.4
```

Quickstart — Run API + frontend
`server.py` defines the FastAPI `app` but does not start an ASGI server by itself. Start it with `uvicorn`:
```powershell
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```
- Frontend: http://127.0.0.1:8000/ or http://127.0.0.1:8000/static/index.html
- Useful endpoints:
	- `GET /health` — health check
	- `POST /detect/image` — multipart image upload → returns annotated image (base64) + detections
	- `GET /detect/webcam` — SSE/streaming annotated frames

Training (overview)
1. Prepare YOLO-format data under `data/labeled/`:
	 - `images/train`, `images/val` — image files
	 - `labels/train`, `labels/val` — `.txt` files, YOLO format: `<class> <cx> <cy> <w> <h>` (normalized)
	 - `dataset.yaml` — paths + `nc` + `names` (see `data/labeled/dataset.yaml`).
2. Train (example):
```powershell
python training/train.py --data data/labeled/dataset.yaml --model yolov8n --epochs 50 --batch 16 --device 0
```
Outputs: `runs/train/<name>/weights/best.pt`

Evaluate a model:
```powershell
python training/evaluate.py --model runs/train/<name>/weights/best.pt --data data/labeled/dataset.yaml --imgsz 640
```

Optional data collection & auto-labeling
```powershell
python training/scraper.py --classes earphones earbuds --per-class 200 --out data/raw
python training/auto_label.py --raw-dir data/raw --out-dir data/labeled --conf 0.35
```
`auto_label.py` can optionally use CLIP (`transformers` + `torch`) for better filtering; without it the script will still produce YOLO `.txt` labels but with simpler heuristics.

Data layout & label format
- `data/labeled/`:
	- `images/train`, `images/val`
	- `labels/train`, `labels/val` (.txt, YOLO format)
	- `dataset.yaml` (relative paths + `nc` + `names`)
- YOLO label: `<class_id> <cx> <cy> <w> <h>` (normalized 0..1)

Important notes
- `data/labeled/dataset.yaml` currently lists `nc: 83` (80 COCO classes + 3 custom classes appended). Some helper scripts (e.g., `training/fix_labels.py`) remap class ids (80,81,82 → 0,1,2). Only run `fix_labels.py` if your labels were created with a different indexing scheme — otherwise keep class ids consistent with `dataset.yaml`.
- `server.py` must be run with an ASGI server (`uvicorn`) — calling `python server.py` alone will not start the app unless you add a `uvicorn.run(...)` block.
- For GPU training and CLIP, install a CUDA-compatible `torch` build matching your GPU/driver.

Examples and outputs
- CLI demo: annotated video window or saved images
- Training: `runs/train/<name>/weights/best.pt`
- Server detection response: JSON detections + base64 annotated image

Helpful files
- `detector.py` — inference helpers
- `main.py` — demo runner
- `server.py` — FastAPI app (start with `uvicorn server:app`)
- `training/train.py` — training wrapper

Next steps / recommendations
- Add a `LICENSE` file (e.g., MIT) and a `CONTRIBUTING.md` if accepting external contributions.
- Document tested Python and CUDA versions and provide a `requirements-dev.txt` for optional tooling (CLIP/transformers).

---
If you'd like, I can (pick one):
- add a minimal `uvicorn.run(...)` wrapper to `server.py` so `python server.py` starts the app, OR
- add a `CONTRIBUTING.md` and an MIT `LICENSE` file. Reply with which and I'll implement it.