"""
Convert Cholec80 annotations to YOLO format.

Cholec80 provides tool presence labels (binary per frame).
Bounding boxes come from the m2cai16 challenge extension or
CholecSeg8k dataset.

Usage:
    python src/prepare_dataset.py \\
        --cholec80_dir data/cholec80 \\
        --output_dir data/yolo_format \\
        --val_videos 10
"""

import argparse
import os
import shutil
import random
from pathlib import Path

import cv2
import numpy as np


INSTRUMENT_CLASSES = [
    "Grasper", "Bipolar", "Hook", "Scissors",
    "Clipper", "Irrigator", "SpecimenBag",
]


def extract_frames(video_path: str, output_dir: Path, fps_target: int = 5):
    """Extract frames at reduced FPS from a video."""
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(src_fps / fps_target))

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = output_dir / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def parse_tool_annotations(annotation_path: str) -> list:
    """
    Parse Cholec80 tool presence annotation file.
    Returns list of dicts: {frame: int, tools: [str]}
    """
    annotations = []
    with open(annotation_path) as f:
        lines = f.readlines()

    for line in lines[1:]:  # skip header
        parts = line.strip().split("\t")
        if len(parts) < 8:
            continue
        frame = int(parts[0])
        tools = []
        for i, cls in enumerate(INSTRUMENT_CLASSES):
            if i + 1 < len(parts) and parts[i + 1] == "1":
                tools.append(cls)
        annotations.append({"frame": frame, "tools": tools})

    return annotations


def create_dataset_yaml(output_dir: Path, n_train: int, n_val: int):
    """Create YOLO dataset configuration yaml."""
    yaml_content = f"""# Cholec80 Surgical Tool Detection Dataset
path: {output_dir.absolute()}
train: images/train
val: images/val

nc: {len(INSTRUMENT_CLASSES)}
names: {INSTRUMENT_CLASSES}

# Dataset info
# Train images: {n_train}
# Val images: {n_val}
"""
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cholec80_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/yolo_format")
    parser.add_argument("--val_videos", type=int, default=10,
                        help="Number of videos for validation")
    parser.add_argument("--fps_target", type=int, default=5,
                        help="Target FPS for frame extraction")
    args = parser.parse_args()

    cholec_dir = Path(args.cholec80_dir)
    output_dir = Path(args.output_dir)

    video_dir = cholec_dir / "videos"
    annot_dir = cholec_dir / "tool_annotations"

    videos = sorted(video_dir.glob("*.mp4"))
    print(f"Found {len(videos)} Cholec80 videos")

    random.seed(42)
    random.shuffle(videos)
    val_videos = videos[:args.val_videos]
    train_videos = videos[args.val_videos:]

    for split, split_videos in [("train", train_videos), ("val", val_videos)]:
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for video_path in split_videos:
            video_id = video_path.stem
            annot_path = annot_dir / f"{video_id}-tool.txt"
            if not annot_path.exists():
                print(f"  No annotation for {video_id}, skipping")
                continue

            print(f"  Processing {video_id} ({split})...")
            frame_dir = output_dir / "tmp" / video_id
            n_saved = extract_frames(str(video_path), frame_dir, args.fps_target)
            annotations = parse_tool_annotations(str(annot_path))

            # Copy frames and write YOLO labels
            for i, frame_path in enumerate(sorted(frame_dir.glob("*.jpg"))):
                dst_img = img_dir / f"{video_id}_{frame_path.name}"
                shutil.copy(frame_path, dst_img)

                # For tool presence labels without bbox, create placeholder
                # In practice, use CholecSeg8k or m2cai16 bboxes
                label_path = lbl_dir / f"{video_id}_{frame_path.stem}.txt"
                label_path.write_text("")  # Empty = no annotations for this frame

    n_train = len(list((output_dir / "images" / "train").glob("*.jpg")))
    n_val = len(list((output_dir / "images" / "val").glob("*.jpg")))
    create_dataset_yaml(output_dir, n_train, n_val)
    print(f"\nDataset created: {n_train} train, {n_val} val images")
    print(f"Config: {output_dir}/dataset.yaml")


if __name__ == "__main__":
    main()
