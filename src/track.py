"""
Run YOLOv8 detection + ByteTrack on a surgical video.

Usage:
    python src/track.py \\
        --video data/sample/sample_clip.mp4 \\
        --model runs/train/weights/best.pt \\
        --output results/tracked_video.mp4 \\
        --save_tracks results/tracks.json
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from model import SurgicalDetector, CLASS_COLORS, INSTRUMENT_CLASSES
from tracker import ByteTracker


def draw_tracks(frame: np.ndarray, tracks, show_trajectory: bool = True) -> np.ndarray:
    """Overlay bounding boxes, labels, and trajectories on frame."""
    annotated = frame.copy()

    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        color = CLASS_COLORS.get(track.class_name, (200, 200, 200))
        label = f"#{track.track_id} {track.class_name} {track.conf:.2f}"

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Trajectory
        if show_trajectory and len(track.history) > 1:
            traj = track.trajectory()
            for i in range(1, len(traj)):
                p1 = (int(traj[i-1][0]), int(traj[i-1][1]))
                p2 = (int(traj[i][0]), int(traj[i][1]))
                alpha = i / len(traj)
                tcolor = tuple(int(c * alpha) for c in color)
                cv2.line(annotated, p1, p2, tcolor, 1)

    return annotated


def process_video(
    video_path: str,
    detector: SurgicalDetector,
    tracker: ByteTracker,
    output_path: str = None,
    save_tracks: str = None,
    show_trajectory: bool = True,
    max_frames: int = None,
) -> dict:
    """
    Process a video file with detection and tracking.

    Returns:
        Summary dict with per-frame tracks and usage statistics.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    frame_records = []
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break

        detections = detector.detect(frame)
        active_tracks = tracker.update(detections)

        frame_record = {
            "frame": frame_idx,
            "time_sec": frame_idx / fps,
            "detections": len(detections),
            "active_tracks": [
                {"id": t.track_id, "class": t.class_name, "bbox": t.bbox, "conf": t.conf}
                for t in active_tracks
            ],
        }
        frame_records.append(frame_record)

        if writer:
            annotated = draw_tracks(frame, active_tracks, show_trajectory)
            fps_text = f"{frame_idx / (time.time() - t0 + 1e-8):.1f} FPS"
            cv2.putText(annotated, fps_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            writer.write(annotated)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total} frames")

    cap.release()
    if writer:
        writer.release()

    # Usage statistics
    instrument_frames = {cls: 0 for cls in INSTRUMENT_CLASSES}
    for record in frame_records:
        for t in record["active_tracks"]:
            if t["class"] in instrument_frames:
                instrument_frames[t["class"]] += 1

    total_frames = len(frame_records)
    usage_pct = {k: 100 * v / total_frames for k, v in instrument_frames.items() if v > 0}

    summary = {
        "video": video_path,
        "total_frames": total_frames,
        "duration_sec": total_frames / fps,
        "total_tracks": tracker._next_id - 1,
        "instrument_usage_pct": usage_pct,
        "frame_records": frame_records,
    }

    if save_tracks:
        Path(save_tracks).parent.mkdir(parents=True, exist_ok=True)
        with open(save_tracks, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Tracks saved to {save_tracks}")

    if output_path:
        print(f"Video saved to {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8m.pt")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save_tracks", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    detector = SurgicalDetector(args.model, conf=args.conf, iou=args.iou)
    tracker = ByteTracker()

    summary = process_video(
        args.video, detector, tracker,
        output_path=args.output,
        save_tracks=args.save_tracks,
        max_frames=args.max_frames,
    )

    print(f"\nProcessed {summary['total_frames']} frames ({summary['duration_sec']:.1f}s)")
    print(f"Total unique instrument tracks: {summary['total_tracks']}")
    print("\nInstrument usage:")
    for inst, pct in sorted(summary["instrument_usage_pct"].items(), key=lambda x: -x[1]):
        print(f"  {inst:<15} {pct:.1f}%")


if __name__ == "__main__":
    main()
