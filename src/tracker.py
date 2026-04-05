"""
ByteTrack multi-object tracker for surgical instruments.

ByteTrack assigns consistent track IDs across frames using IoU-based
association and a Kalman filter for motion prediction. This enables
temporal analysis of instrument usage — the key capability beyond
frame-by-frame detection.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box", ECCV 2022.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    """Represents a tracked surgical instrument over time."""
    track_id: int
    class_name: str
    bbox: List[float]          # [x1, y1, x2, y2]
    conf: float
    age: int = 0               # frames since last detection
    hits: int = 1              # total detections matched
    state: str = "active"      # "active" | "lost" | "removed"
    history: List[Dict] = field(default_factory=list)

    def update(self, bbox: List[float], conf: float, frame_idx: int):
        self.bbox = bbox
        self.conf = conf
        self.age = 0
        self.hits += 1
        self.history.append({"frame": frame_idx, "bbox": bbox, "conf": conf})

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def trajectory(self) -> List[Tuple[float, float]]:
        return [((h["bbox"][0] + h["bbox"][2]) / 2,
                 (h["bbox"][1] + h["bbox"][3]) / 2)
                for h in self.history]


def iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def cost_matrix(tracks: List[Track], detections: List[Dict]) -> np.ndarray:
    """Compute IoU cost matrix between active tracks and new detections."""
    C = np.zeros((len(tracks), len(detections)))
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            if track.class_name == det["class"]:
                C[i, j] = 1.0 - iou(track.bbox, det["bbox"])
            else:
                C[i, j] = 1.0  # different class = no match
    return C


class ByteTracker:
    """
    Simplified ByteTrack implementation for surgical instrument tracking.

    Maintains a pool of active and lost tracks. Associates new detections
    to existing tracks via IoU-based Hungarian matching.

    Args:
        max_lost_frames:  Frames before a lost track is removed.
        min_hits:         Minimum detections before a track is confirmed.
        iou_threshold:    IoU threshold for matching.
    """

    def __init__(
        self,
        max_lost_frames: int = 10,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        self.max_lost = max_lost_frames
        self.min_hits = min_hits
        self.iou_thresh = iou_threshold
        self.tracks: List[Track] = []
        self._next_id = 1
        self.frame_idx = 0

    def _new_track(self, det: Dict) -> Track:
        t = Track(
            track_id=self._next_id,
            class_name=det["class"],
            bbox=det["bbox"],
            conf=det["conf"],
        )
        t.history.append({"frame": self.frame_idx, "bbox": det["bbox"], "conf": det["conf"]})
        self._next_id += 1
        return t

    def update(self, detections: List[Dict]) -> List[Track]:
        """
        Update tracker with new detections from the current frame.

        Args:
            detections: List of detection dicts from SurgicalDetector.

        Returns:
            List of confirmed active tracks.
        """
        self.frame_idx += 1
        active = [t for t in self.tracks if t.state == "active"]

        if not active:
            # Initialize tracks for all detections
            for det in detections:
                self.tracks.append(self._new_track(det))
        elif detections:
            C = cost_matrix(active, detections)
            row_ind, col_ind = linear_sum_assignment(C)

            matched_track_ids = set()
            matched_det_ids = set()

            for r, c in zip(row_ind, col_ind):
                if C[r, c] < (1.0 - self.iou_thresh):
                    active[r].update(detections[c]["bbox"], detections[c]["conf"], self.frame_idx)
                    matched_track_ids.add(r)
                    matched_det_ids.add(c)

            # Unmatched detections → new tracks
            for j, det in enumerate(detections):
                if j not in matched_det_ids:
                    self.tracks.append(self._new_track(det))

            # Unmatched tracks → mark lost
            for i, track in enumerate(active):
                if i not in matched_track_ids:
                    track.age += 1
                    if track.age > self.max_lost:
                        track.state = "removed"
                    else:
                        track.state = "lost"
        else:
            # No detections: age all active tracks
            for track in active:
                track.age += 1
                if track.age > self.max_lost:
                    track.state = "removed"

        return [t for t in self.tracks if t.state == "active" and t.hits >= self.min_hits]

    def all_tracks(self) -> List[Track]:
        return self.tracks

    def reset(self):
        self.tracks = []
        self._next_id = 1
        self.frame_idx = 0
