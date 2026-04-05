"""
Fine-tune YOLOv8 on Cholec80 surgical instrument dataset.

Usage:
    python src/train.py \\
        --data data/yolo_format/dataset.yaml \\
        --model yolov8m.pt \\
        --epochs 50 \\
        --imgsz 640 \\
        --output runs/train
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                        help="YOLOv8 model variant or checkpoint path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output", type=str, default="runs/train")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        project=args.output,
        name="surgical_tools",
        patience=15,
        save=True,
        val=True,
        plots=True,
        device=args.device or None,
        augment=True,
        mosaic=1.0,
        degrees=10.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    print(f"\nTraining complete.")
    print(f"Best weights: {args.output}/surgical_tools/weights/best.pt")
    return results


if __name__ == "__main__":
    main()
