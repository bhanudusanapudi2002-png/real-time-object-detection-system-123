"""
auto_label.py
=============
Automatically generate YOLO-format bounding-box labels for scraped images.

Strategy
--------
1. Run a pretrained YOLOv8 model on every scraped image.
2. For images where YOLO already knows the object (e.g. "cell phone"),
   use those detections directly (class remapping applied).
3. For truly new classes (earphones, buds, etc.) YOLO has no label.
   We use a CLIP zero-shot image classifier to score whether the image
   actually contains the target object — if confident enough, we assign
   a whole-image bounding box (common practice when scraping web images
   that are centred on the subject).
4. Output is written in standard YOLO label format:
      <class_id> <cx> <cy> <w> <h>   (all values 0-1, relative)

Usage
-----
python training/auto_label.py \
    --raw-dir  data/raw \
    --out-dir  data/labeled \
    --classes  "earphones" "earbuds" "tws buds" \
    --conf     0.35
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# COCO → custom class bridge
# Map COCO class names that are close enough to your custom classes.
# Edit this dict to add more fuzzy mappings for your use case.
# ---------------------------------------------------------------------------
COCO_ALIAS: dict[str, str] = {
    # coco_name : your_custom_class_name
    "cell phone"  : "phone",
    "laptop"      : "laptop",
    "keyboard"    : "keyboard",
    "mouse"       : "mouse",
    "remote"      : "remote",
    "tv"          : "tv_monitor",
    "book"        : "book",
    "bottle"      : "bottle",
    "cup"         : "cup",
    # Add more as needed, e.g.: "headphones": "earphones"
}


# ---------------------------------------------------------------------------
# Optional CLIP for zero-shot scoring of new classes
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


class CLIPScorer:
    """
    Uses OpenAI CLIP to score how likely an image contains a given text label.
    Falls back gracefully when transformers / torch are not installed.
    """

    def __init__(self) -> None:
        if not HAS_CLIP:
            print("[AutoLabel] CLIP not available — will use whole-image box for "
                  "new classes without confidence filtering.\n"
                  "  To enable CLIP:  pip install transformers torch")
            self.model = None
            return
        print("[AutoLabel] Loading CLIP (ViT-B/32) …")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model      = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        print("[AutoLabel] CLIP ready.")

    def score(self, image_path: Path, label: str) -> float:
        """Return probability (0-1) that the image contains *label*."""
        if self.model is None:
            return 1.0   # no filter — accept everything

        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        prompts = [f"a photo of {label}", f"a photo of something else"]
        inputs  = self.processor(text=prompts, images=image, return_tensors="pt",
                                 padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits_per_image
            probs  = logits.softmax(dim=1)[0]
        return float(probs[0])   # probability for "a photo of <label>"


# ---------------------------------------------------------------------------
# YOLO label writer
# ---------------------------------------------------------------------------

def write_yolo_label(
    label_path: Path,
    detections: list[tuple[int, float, float, float, float]],
) -> None:
    """
    Write YOLO-format label file.

    Each detection is (class_id, cx, cy, w, h) all normalised 0-1.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
             for cls, cx, cy, w, h in detections]
    label_path.write_text("\n".join(lines))


def whole_image_box(class_id: int) -> tuple[int, float, float, float, float]:
    """Return a bounding box covering ~85% of the image (typical centred subject)."""
    return (class_id, 0.5, 0.5, 0.85, 0.85)


# ---------------------------------------------------------------------------
# AutoLabeler
# ---------------------------------------------------------------------------

class AutoLabeler:
    """
    Orchestrates auto-labeling for a mix of COCO and new custom classes.

    Parameters
    ----------
    raw_dir    : root dir containing one sub-folder per class (from scraper.py)
    out_dir    : destination — images copied to out_dir/images/train|val,
                 labels to out_dir/labels/train|val
    classes    : list of NEW custom class names (earphones, buds, etc.)
    conf       : minimum YOLO confidence to keep a detection
    val_split  : fraction of images reserved for validation (default 0.15)
    """

    def __init__(
        self,
        raw_dir    : Path,
        out_dir    : Path,
        classes    : list[str],
        conf       : float = 0.35,
        val_split  : float = 0.15,
        yolo_model : str   = "yolov8n.pt",
    ) -> None:
        self.raw_dir   = Path(raw_dir)
        self.out_dir   = Path(out_dir)
        self.conf      = conf
        self.val_split = val_split

        # Build combined class list:
        # COCO 80 classes come first (indices 0-79), custom classes appended
        self.coco_classes  = self._load_coco_names()           # list[str]
        self.custom_classes = [c.lower() for c in classes]
        self.all_classes    = self.coco_classes + self.custom_classes

        print(f"[AutoLabel] {len(self.coco_classes)} COCO classes  +  "
              f"{len(self.custom_classes)} custom classes  "
              f"= {len(self.all_classes)} total")

        self.yolo  = YOLO(yolo_model)
        self.clip  = CLIPScorer()

        # Build reverse lookup: coco_name → combined_class_id
        self._coco_id = {name: idx for idx, name in enumerate(self.coco_classes)}

        # Build custom class id lookup
        self._custom_id = {
            name: len(self.coco_classes) + i
            for i, name in enumerate(self.custom_classes)
        }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Process all scraped images and write labeled dataset."""
        stats = {"labeled": 0, "skipped": 0}

        for class_dir in sorted(self.raw_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name.lower()
            images     = sorted(class_dir.glob("*.jpg")) + \
                         sorted(class_dir.glob("*.png")) + \
                         sorted(class_dir.glob("*.jpeg"))

            if not images:
                print(f"[AutoLabel] No images in {class_dir} — skipping.")
                continue

            print(f"\n[AutoLabel] Processing class '{class_name}'  "
                  f"({len(images)} images)")

            # Decide labeling strategy
            is_coco_class    = class_name in self._coco_id
            is_custom_class  = class_name in self._custom_id
            is_aliased       = class_name in COCO_ALIAS

            for idx, img_path in enumerate(images):
                split     = "val" if idx < len(images) * self.val_split else "train"
                dest_img  = self.out_dir / "images" / split / img_path.name
                dest_lbl  = (self.out_dir / "labels" / split /
                             img_path.with_suffix(".txt").name)

                detections: list[tuple[int, float, float, float, float]] = []

                if is_coco_class or is_aliased:
                    # YOLO already knows this object → run inference
                    detections = self._label_with_yolo(img_path, class_name)

                elif is_custom_class:
                    # New class → CLIP score then whole-image box
                    detections = self._label_new_class(img_path, class_name)

                if not detections:
                    stats["skipped"] += 1
                    continue

                # Copy image + write label
                dest_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest_img)
                write_yolo_label(dest_lbl, detections)
                stats["labeled"] += 1

                if (idx + 1) % 20 == 0:
                    print(f"  {idx+1}/{len(images)} processed …")

        # Write dataset YAML
        yaml_path = self._write_dataset_yaml()

        print(f"\n[AutoLabel] ✓  Labeled: {stats['labeled']}  |  "
              f"Skipped: {stats['skipped']}")
        print(f"[AutoLabel] Dataset YAML → {yaml_path}")
        print(f"[AutoLabel] Next step: python training/train.py "
              f"--data {yaml_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _label_with_yolo(
        self, img_path: Path, target_class: str
    ) -> list[tuple[int, float, float, float, float]]:
        """Run YOLOv8 and keep boxes matching *target_class* (or its alias)."""
        # Resolve target to combined class id
        resolved = COCO_ALIAS.get(target_class, target_class)
        class_id = self._coco_id.get(resolved, self._custom_id.get(resolved))
        if class_id is None:
            return []

        img     = cv2.imread(str(img_path))
        if img is None:
            return []
        h, w    = img.shape[:2]
        results = self.yolo.predict(img, conf=self.conf, verbose=False)
        boxes   = results[0].boxes

        detections = []
        for box in boxes:
            coco_id = int(box.cls[0])
            coco_name = self.coco_classes[coco_id] if coco_id < len(self.coco_classes) else ""
            if coco_name != resolved and COCO_ALIAS.get(coco_name) != target_class:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            detections.append((class_id, cx, cy, bw, bh))

        return detections

    def _label_new_class(
        self, img_path: Path, class_name: str
    ) -> list[tuple[int, float, float, float, float]]:
        """Use CLIP to verify image content, then assign a whole-image box."""
        class_id = self._custom_id.get(class_name)
        if class_id is None:
            return []

        score = self.clip.score(img_path, class_name)
        if score < 0.40:   # CLIP not confident this image shows the class
            return []

        return [whole_image_box(class_id)]

    def _write_dataset_yaml(self) -> Path:
        """Write dataset.yaml for Ultralytics training."""
        yaml_path = self.out_dir / "dataset.yaml"
        lines = [
            f"path: {self.out_dir.resolve()}",
            f"train: images/train",
            f"val:   images/val",
            f"",
            f"nc: {len(self.all_classes)}",
            f"names: {json.dumps(self.all_classes)}",
        ]
        yaml_path.write_text("\n".join(lines))
        return yaml_path

    @staticmethod
    def _load_coco_names() -> list[str]:
        """Return the 80 standard COCO class names in order."""
        return [
            "person","bicycle","car","motorcycle","airplane","bus","train",
            "truck","boat","traffic light","fire hydrant","stop sign",
            "parking meter","bench","bird","cat","dog","horse","sheep","cow",
            "elephant","bear","zebra","giraffe","backpack","umbrella","handbag",
            "tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
            "baseball bat","baseball glove","skateboard","surfboard",
            "tennis racket","bottle","wine glass","cup","fork","knife","spoon",
            "bowl","banana","apple","sandwich","orange","broccoli","carrot",
            "hot dog","pizza","donut","cake","chair","couch","potted plant",
            "bed","dining table","toilet","tv","laptop","mouse","remote",
            "keyboard","cell phone","microwave","oven","toaster","sink",
            "refrigerator","book","clock","vase","scissors","teddy bear",
            "hair drier","toothbrush",
        ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Auto-label scraped images using YOLOv8 + CLIP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training/auto_label.py \\
      --raw-dir data/raw \\
      --out-dir data/labeled \\
      --classes "earphones" "earbuds" "tws buds"

  python training/auto_label.py \\
      --raw-dir data/raw --out-dir data/labeled \\
      --classes "smartwatch" "fitness band" \\
      --conf 0.40 --model yolov8s.pt
        """,
    )
    p.add_argument("--raw-dir",  default="data/raw",     dest="raw_dir")
    p.add_argument("--out-dir",  default="data/labeled", dest="out_dir")
    p.add_argument("--classes",  nargs="+", required=True)
    p.add_argument("--conf",     type=float, default=0.35)
    p.add_argument("--model",    default="yolov8n.pt")
    p.add_argument("--val-split",type=float, default=0.15, dest="val_split")
    args = p.parse_args()

    labeler = AutoLabeler(
        raw_dir     = Path(args.raw_dir),
        out_dir     = Path(args.out_dir),
        classes     = args.classes,
        conf        = args.conf,
        val_split   = args.val_split,
        yolo_model  = args.model,
    )
    labeler.run()


if __name__ == "__main__":
    main()