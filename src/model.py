"""
YOLOv8 surgical instrument detector with ByteTrack integration.

Wraps Ultralytics YOLOv8 for surgical tool detection on Cholec80.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

# Cholec80 instrument classes
INSTRUMENT_CLASSES = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag",
]
NUM_CLASSES = len(INSTRUMENT_CLASSES)

# Color map for visualization (BGR for OpenCV)
CLASS_COLORS = {
    "Grasper":     (0, 255, 0),
    "Bipolar":     (255, 0, 0),
    "Hook":        (0, 165, 255),
    "Scissors":    (0, 0, 255),
    "Clipper":     (255, 0, 255),
    "Irrigator":   (255, 255, 0),
    "SpecimenBag": (128, 0, 128),
}


class SurgicalDetector:
    """
    YOLOv8-based surgical instrument detector.

    Args:
        model_path:  Path to YOLOv8 checkpoint (.pt) or 'yolov8m.pt' for pretrained.
        conf:        Detection confidence threshold.
        iou:         NMS IoU threshold.
        device:      'cuda', 'cpu', or 'mps'.
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image array [H, W, 3].

        Returns:
            List of detections: {'class': str, 'bbox': [x1,y1,x2,y2], 'conf': float}
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            cls_name = INSTRUMENT_CLASSES[cls_id] if cls_id < NUM_CLASSES else f"class_{cls_id}"
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class": cls_name,
                "class_id": cls_id,
                "bbox": [x1, y1, x2, y2],
                "conf": float(box.conf.item()),
            })
        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """Run detection on a batch of frames."""
        return [self.detect(f) for f in frames]

    def get_present_instruments(self, frame: np.ndarray) -> List[str]:
        """Return list of instrument class names present in frame."""
        dets = self.detect(frame)
        return list({d["class"] for d in dets})


def build_detector(checkpoint: Optional[str] = None, **kwargs) -> SurgicalDetector:
    """Convenience function to build detector from checkpoint or pretrained weights."""
    model_path = checkpoint or "yolov8m.pt"
    return SurgicalDetector(model_path=model_path, **kwargs)
