# Surgical Tool Tracker: Real-Time Detection & Tracking

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red) ![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

> Real-time surgical instrument detection and multi-object tracking in laparoscopic video using YOLOv8 + ByteTrack, trained on the Cholec80 dataset.

---

## Problem Statement

Surgical instrument detection and tracking is a foundational capability for:
- **Surgical workflow recognition** — automated phase/step detection
- **Robotic assistance** — instrument handoff and task automation
- **Quality assurance** — instrument count verification (preventing retained objects)
- **Surgical training** — objective skill assessment

Most student projects stop at frame-by-frame detection. This pipeline adds **temporal tracking** with ByteTrack, enabling consistent instrument IDs across frames — the key capability for workflow analysis.

---

## Dataset: Cholec80

[Cholec80](http://camma.u-strasbg.fr/datasets) contains:
- **80 laparoscopic cholecystectomy videos** (~30 min each)
- **7 instrument classes**: Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, Specimen Bag
- **Tool presence annotations** at 25 fps
- Bounding box annotations available via the m2cai16 challenge extension

---

## Results

### Detection (YOLOv8-m, Cholec80)

| Instrument | AP@50 |
|-----------|-------|
| Grasper | 0.891 |
| Hook | 0.873 |
| Clipper | 0.812 |
| Scissors | 0.788 |
| Bipolar | 0.764 |
| **mAP@50** | **0.826** |

### Tracking (ByteTrack)

| Metric | Value |
|--------|-------|
| MOTA | 0.712 |
| IDF1 | 0.781 |
| ID Switches | 43 |

---

## Architecture

```
Video Frame [H, W, 3]
       │
   YOLOv8-m (detection backbone)
       │ bounding boxes + class + confidence
   ByteTrack (multi-object tracker)
       │ tracked IDs + trajectories
   Temporal Analysis
       │
   Instrument usage timeline + phase prediction
```

---

## Installation

```bash
git clone https://github.com/hollyakt/surgical-tool-tracker.git
cd surgical-tool-tracker
pip install -r requirements.txt
```

### Data Setup

1. Download [Cholec80](http://camma.u-strasbg.fr/datasets) (requires registration)
2. Place videos in `data/cholec80/videos/`
3. Place annotations in `data/cholec80/annotations/`

```bash
python src/prepare_dataset.py \
  --cholec80_dir data/cholec80 \
  --output_dir data/yolo_format
```

---

## Usage

### Train YOLOv8 detector
```bash
python src/train.py \
  --data data/yolo_format/dataset.yaml \
  --model yolov8m.pt \
  --epochs 50 \
  --output runs/train
```

### Run detection + tracking on a video
```bash
python src/track.py \
  --video data/sample/sample_clip.mp4 \
  --model runs/train/weights/best.pt \
  --output results/tracked_video.mp4
```

### Analyze tool usage timeline
```bash
python src/analyze.py \
  --tracks results/tracks.json \
  --output results/timeline.png
```

---

## Project Structure

```
surgical-tool-tracker/
├── src/
│   ├── model.py           # YOLOv8 wrapper with ByteTrack integration
│   ├── train.py           # Fine-tuning script
│   ├── track.py           # Video inference + tracking pipeline
│   ├── prepare_dataset.py # Cholec80 → YOLO format conversion
│   ├── tracker.py         # ByteTrack implementation
│   └── analyze.py         # Tool usage timeline analysis
├── notebooks/
│   └── 01_detection_and_tracking.ipynb
├── figures/
├── requirements.txt
└── README.md
```

---

## Why This Matters for Surgical Robotics

This work directly supports the vision of autonomous surgical assistance. At Vanderbilt's STORM Lab and similar groups, instrument tracking underpins:
- Tool-to-tissue interaction modeling
- Context-aware robotic control
- Automated surgical reporting

---

## License

MIT License. See [LICENSE](LICENSE).
