"""
evaluate.py
===========
Evaluate a trained model on the validation split and print a per-class report.

Usage
-----
python training/evaluate.py --model runs/train/custom_yolo/weights/best.pt \\
                             --data  data/labeled/dataset.yaml
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a YOLO model on val set.")
    p.add_argument("--model", required=True, help="Path to best.pt")
    p.add_argument("--data",  required=True, help="Path to dataset.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf",  type=float, default=0.25)
    p.add_argument("--iou",   type=float, default=0.50)
    args = p.parse_args()

    model = YOLO(args.model)
    metrics = model.val(
        data   = args.data,
        imgsz  = args.imgsz,
        conf   = args.conf,
        iou    = args.iou,
        verbose= True,
    )

    print("\n[Evaluate] Summary")
    print(f"  mAP@0.5      : {metrics.box.map50:.3f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.3f}")
    print(f"  Precision    : {metrics.box.mp:.3f}")
    print(f"  Recall       : {metrics.box.mr:.3f}")


if __name__ == "__main__":
    main()