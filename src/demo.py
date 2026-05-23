"""Open-data demo for the surgical / endoscopic instrument tracker.

Fully end-to-end: downloads Kvasir-Instrument (open, ~170 MB), converts
bounding-box annotations to YOLO format, fine-tunes yolov8n for a few
epochs, runs inference on validation frames, and saves a detection grid
to figures/kvasir_detection.png.

Kvasir-Instrument is a GI-endoscopy dataset (single `instrument` class)
published by Simula Research Lab — open, no registration required.
The pipeline architecture (YOLO + ByteTrack) is the same as the Cholec80
production path; only the data source changes.

Usage:
    python src/demo.py --epochs 15 --imgsz 640
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO

KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-instrument.zip"
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "kvasir-instrument"
WORK_DIR = REPO / "data" / "kvasir_yolo"
FIG_DIR = REPO / "figures"


def download_and_extract() -> Path:
    """Download Kvasir-Instrument and extract images + bounding-box metadata."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR.parent / "kvasir-instrument.zip"
    if not (DATA_DIR / "bboxes.json").exists():
        if not zip_path.exists():
            print(f"downloading {KVASIR_URL} ...")
            subprocess.run(["curl", "-sL", "-o", str(zip_path), KVASIR_URL],
                            check=True)
        print("extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR.parent)
    # The archive expands to ./kvasir-instrument/ with images.tar.gz inside.
    images_tar = DATA_DIR / "images.tar.gz"
    if images_tar.exists() and not (DATA_DIR / "images").exists():
        subprocess.run(["tar", "-xzf", str(images_tar), "-C", str(DATA_DIR)],
                        check=True)
    return DATA_DIR


def convert_to_yolo(src_dir: Path, work_dir: Path) -> Path:
    """Convert Kvasir's bboxes.json layout to YOLO label files + split dirs."""
    bboxes = json.loads((src_dir / "bboxes.json").read_text())
    train_ids = (src_dir / "train.txt").read_text().strip().split("\n")
    test_ids = (src_dir / "test.txt").read_text().strip().split("\n")

    for split, ids in [("train", train_ids), ("val", test_ids)]:
        img_dir = work_dir / "images" / split
        lbl_dir = work_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_id in ids:
            src = src_dir / "images" / f"{img_id}.jpg"
            if not src.exists():
                continue
            dst = img_dir / f"{img_id}.jpg"
            if not dst.exists():
                shutil.copy(src, dst)
            meta = bboxes.get(img_id)
            if not meta:
                continue
            W, H = meta["width"], meta["height"]
            lines = []
            for b in meta["bbox"]:
                xc = (b["xmin"] + b["xmax"]) / 2 / W
                yc = (b["ymin"] + b["ymax"]) / 2 / H
                w = (b["xmax"] - b["xmin"]) / W
                h = (b["ymax"] - b["ymin"]) / H
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            (lbl_dir / f"{img_id}.txt").write_text("\n".join(lines) + "\n")

    yaml_path = work_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {work_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: instrument\n"
    )
    return yaml_path


def render(model: YOLO, src_dir: Path, sample_ids: list[str], out_path: Path) -> None:
    """Render a 1xN grid of validation frames with YOLO predictions overlaid."""
    BG = "#fdf9f0"
    ROSE = "#a04f6a"
    TEXT = "#1e1a1c"
    MUTED = "#6b5a62"

    fig, axes = plt.subplots(1, len(sample_ids), figsize=(11, 3.2))
    if len(sample_ids) == 1:
        axes = [axes]

    for ax, img_id in zip(axes, sample_ids):
        img_path = src_dir / "images" / f"{img_id}.jpg"
        ax.imshow(Image.open(img_path).convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        res = model.predict(source=str(img_path), conf=0.25, verbose=False)[0]
        for box, conf in zip(res.boxes.xyxy.cpu().numpy(),
                              res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.0, edgecolor=ROSE, facecolor="none",
            ))
            ax.text(x1 + 4, y1 - 8, f"instrument {conf:.2f}",
                     color="white", fontsize=9, family="DejaVu Sans Mono",
                     bbox=dict(boxstyle="round,pad=0.2",
                                facecolor=ROSE, edgecolor="none"))
        ax.set_title(f"frame · {img_id[-8:]}", color=MUTED,
                     fontsize=9, pad=4, family="DejaVu Sans")

    fig.suptitle(
        "YOLOv8n · Kvasir-Instrument · fine-tuned for endoscopic instrument detection",
        color=TEXT, fontsize=12, fontweight="bold", y=1.02,
        family="DejaVu Sans",
    )
    fig.set_facecolor(BG)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu",
                    help="cpu | 0 | mps | auto")
    args = ap.parse_args()

    src_dir = download_and_extract()
    yaml_path = convert_to_yolo(src_dir, WORK_DIR)

    print(f"\nfine-tuning yolov8n for {args.epochs} epochs ...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device,
        project=str(WORK_DIR), name="run",
        verbose=False, save=True, workers=2, patience=10,
    )
    best = WORK_DIR / "run" / "weights" / "best.pt"
    if not best.exists():
        best = WORK_DIR / "run" / "weights" / "last.pt"

    bboxes = json.loads((src_dir / "bboxes.json").read_text())
    test_ids = (src_dir / "test.txt").read_text().strip().split("\n")
    samples = [i for i in test_ids if i in bboxes][:4]

    render(YOLO(str(best)), src_dir, samples,
           FIG_DIR / "kvasir_detection.png")


if __name__ == "__main__":
    main()
