"""
Analyze surgical tool usage timeline from tracking results.

Produces:
- Tool presence timeline (Gantt-style chart)
- Usage statistics per instrument
- Phase prediction based on tool usage patterns

Usage:
    python src/analyze.py \\
        --tracks results/tracks.json \\
        --output results/timeline.png
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


# Simplified phase detection heuristics based on Cholec80 phase annotations
# Phase ← dominant instrument in time window
PHASE_RULES = {
    "Preparation":        ["Grasper"],
    "ClippingCutting":    ["Clipper", "Scissors"],
    "Dissection":         ["Hook", "Bipolar"],
    "SpecimenRetrieval":  ["SpecimenBag"],
    "WashingSuction":     ["Irrigator"],
}

INSTRUMENT_COLORS = {
    "Grasper":     "#4CAF50",
    "Bipolar":     "#2196F3",
    "Hook":        "#FF9800",
    "Scissors":    "#F44336",
    "Clipper":     "#9C27B0",
    "Irrigator":   "#00BCD4",
    "SpecimenBag": "#795548",
}


def load_tracks(tracks_path: str) -> dict:
    with open(tracks_path) as f:
        return json.load(f)


def build_presence_matrix(tracks_data: dict) -> tuple:
    """Build a binary matrix [frames × instruments] of tool presence."""
    frame_records = tracks_data["frame_records"]
    instruments = list(INSTRUMENT_COLORS.keys())
    inst_idx = {name: i for i, name in enumerate(instruments)}

    n_frames = len(frame_records)
    matrix = np.zeros((n_frames, len(instruments)), dtype=int)

    for rec in frame_records:
        f = rec["frame"]
        for track in rec["active_tracks"]:
            if track["class"] in inst_idx:
                matrix[f, inst_idx[track["class"]]] = 1

    return matrix, instruments


def predict_phases(matrix: np.ndarray, instruments: list,
                   window: int = 50) -> list:
    """Sliding window phase prediction based on dominant instrument."""
    phases = []
    for i in range(len(matrix)):
        start = max(0, i - window // 2)
        end = min(len(matrix), i + window // 2)
        window_sum = matrix[start:end].sum(axis=0)

        phase = "Other"
        for phase_name, phase_insts in PHASE_RULES.items():
            if all(instruments.index(inst) < len(window_sum) and
                   window_sum[instruments.index(inst)] > 0
                   for inst in phase_insts if inst in instruments):
                dominant_idx = window_sum.argmax()
                if instruments[dominant_idx] in phase_insts:
                    phase = phase_name
                    break
        phases.append(phase)
    return phases


def plot_timeline(tracks_data: dict, output_path: str):
    """Create a Gantt-style tool presence timeline."""
    matrix, instruments = build_presence_matrix(tracks_data)
    fps = tracks_data["total_frames"] / max(tracks_data["duration_sec"], 1)
    times = np.arange(len(matrix)) / fps  # seconds

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={"height_ratios": [3, 1]})

    # Tool presence timeline
    ax = axes[0]
    for i, (inst, color) in enumerate(INSTRUMENT_COLORS.items()):
        if i >= matrix.shape[1]:
            continue
        presence = matrix[:, i].astype(float)
        presence[presence == 0] = np.nan
        ax.fill_between(times, i - 0.4, i + 0.4 * presence,
                         where=~np.isnan(presence),
                         color=color, alpha=0.7, step="mid")

    ax.set_yticks(range(len(INSTRUMENT_COLORS)))
    ax.set_yticklabels(list(INSTRUMENT_COLORS.keys()))
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Surgical Instrument Presence Timeline", fontweight="bold", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, times[-1] if len(times) > 0 else 1)

    # Usage percentage bar
    ax2 = axes[1]
    usage = matrix.mean(axis=0) * 100
    colors = list(INSTRUMENT_COLORS.values())
    bars = ax2.bar(list(INSTRUMENT_COLORS.keys()), usage[:len(INSTRUMENT_COLORS)],
                   color=colors, alpha=0.85)
    for bar, pct in zip(bars, usage):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("% Frames Present")
    ax2.set_title("Instrument Usage Summary", fontweight="bold")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Timeline saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/timeline.png")
    args = parser.parse_args()

    tracks_data = load_tracks(args.tracks)
    print(f"Loaded {tracks_data['total_frames']} frames, "
          f"{tracks_data['total_tracks']} unique tracks")
    print(f"Duration: {tracks_data['duration_sec']:.1f}s")

    print("\nInstrument usage:")
    for inst, pct in sorted(tracks_data["instrument_usage_pct"].items(), key=lambda x: -x[1]):
        print(f"  {inst:<15} {pct:.1f}%")

    plot_timeline(tracks_data, args.output)


if __name__ == "__main__":
    main()
