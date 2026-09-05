"""
Visualization Demo — Trajectory Prediction Module.

Generates synthetic multi-vehicle scenarios and visualises:
  • Historical positions (solid line with filled dots)
  • Current position (large marker)
  • Kalman-filter predicted future positions (dashed line with open markers)

Run:
    python prediction/visualize_demo.py
"""

import sys
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prediction.kalman_filter import TrajectoryPredictor


# ---------------------------------------------------------------------------
# Synthetic vehicle generators
# ---------------------------------------------------------------------------

def generate_straight(track_id, start, velocity, n_frames, dt):
    """Constant-velocity straight-line motion."""
    frames = []
    for i in range(n_frames):
        t = i * dt
        x = start[0] + velocity[0] * t
        y = start[1] + velocity[1] * t
        frames.append((track_id, x, y, t))
    return frames


def generate_curve(track_id, center, radius, start_angle, angular_vel,
                   n_frames, dt):
    """Circular arc motion (e.g., a vehicle turning at an intersection)."""
    frames = []
    for i in range(n_frames):
        t = i * dt
        angle = start_angle + angular_vel * t
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        frames.append((track_id, x, y, t))
    return frames


def generate_accelerating(track_id, start, velocity, accel, n_frames, dt):
    """Linearly accelerating motion."""
    frames = []
    for i in range(n_frames):
        t = i * dt
        x = start[0] + velocity[0] * t + 0.5 * accel[0] * t * t
        y = start[1] + velocity[1] * t + 0.5 * accel[1] * t * t
        frames.append((track_id, x, y, t))
    return frames


# ---------------------------------------------------------------------------
# Build a TrackedObject dict (same contract as Module 3)
# ---------------------------------------------------------------------------

def build_tracked_object(track_id, history_tuples, cls="car"):
    x, y, _ = history_tuples[-1]
    half = 25
    return {
        "track_id": track_id,
        "class": cls,
        "bbox": (x - half, y - half, x + half, y + half),
        "history": history_tuples,
    }


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    fps = 10
    dt = 1.0 / fps
    n_frames = 30  # 3 seconds of data

    # --- Generate four synthetic vehicles ---
    vehicles = {
        1: {
            "label": "Car A — straight right",
            "color": "#2196F3",
            "frames": generate_straight(1, (50, 300), (80, 0), n_frames, dt),
        },
        2: {
            "label": "Bike B — straight up-left",
            "color": "#FF5722",
            "frames": generate_straight(2, (600, 500), (-50, -40), n_frames, dt),
        },
        3: {
            "label": "Auto C — turning (curve)",
            "color": "#4CAF50",
            "frames": generate_curve(3, (400, 350), 120, 0, 0.8, n_frames, dt),
        },
        4: {
            "label": "Bus D — accelerating",
            "color": "#9C27B0",
            "frames": generate_accelerating(4, (100, 100), (30, 20), (15, 10),
                                            n_frames, dt),
        },
    }

    predictor = TrajectoryPredictor(config_path="prediction/config.yaml")

    # --- Feed frames one at a time, collect results from the last frame ---
    final_results = None
    for frame_idx in range(n_frames):
        t = frame_idx * dt
        tracked_objects = []
        for tid, v in vehicles.items():
            # Build history up to current frame
            history = [(f[1], f[2], f[3]) for f in v["frames"][: frame_idx + 1]]
            tracked_objects.append(
                build_tracked_object(tid, history, cls="car")
            )
        final_results = predictor.update(tracked_objects, t)

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")

    ax.set_title(
        "Module 4 — Trajectory Prediction Demo",
        fontsize=16, fontweight="bold", color="white", pad=15,
    )
    ax.set_xlabel("X (pixels)", color="white", fontsize=12)
    ax.set_ylabel("Y (pixels)", color="white", fontsize=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")

    legend_handles = []

    for pred in final_results:
        tid = pred["track_id"]
        v = vehicles[tid]
        color = v["color"]

        # Full ground-truth history
        all_positions = [(f[1], f[2]) for f in v["frames"]]
        hx = [p[0] for p in all_positions]
        hy = [p[1] for p in all_positions]

        # Historical trail (solid)
        ax.plot(hx, hy, color=color, linewidth=2, alpha=0.7, zorder=2)
        ax.scatter(hx, hy, color=color, s=15, alpha=0.5, zorder=2)

        # Current position (large marker)
        ax.scatter(
            hx[-1], hy[-1], color=color, s=120, edgecolors="white",
            linewidths=2, zorder=4,
        )

        # Predicted future positions (dashed)
        pred_x = [hx[-1]] + [p[0] for p in pred["predictions"]]
        pred_y = [hy[-1]] + [p[1] for p in pred["predictions"]]
        ax.plot(
            pred_x, pred_y, color=color, linewidth=2.5,
            linestyle="--", alpha=0.9, zorder=3,
        )
        # Open markers at prediction points
        ax.scatter(
            pred_x[1:], pred_y[1:], facecolors="none", edgecolors=color,
            s=80, linewidths=2, zorder=3,
        )

        # Label prediction timestamps
        for i, (px, py, pt) in enumerate(pred["predictions"]):
            ax.annotate(
                f"t+{i+1}s",
                (px, py),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                color=color,
                fontweight="bold",
                alpha=0.9,
            )

        # Velocity annotation
        vx, vy = pred["velocity"]
        ax.annotate(
            f"v=({vx:.0f}, {vy:.0f})",
            (hx[-1], hy[-1]),
            textcoords="offset points",
            xytext=(-15, -20),
            fontsize=8,
            color="white",
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.6),
        )

        legend_handles.append(mpatches.Patch(color=color, label=v["label"]))

    ax.legend(
        handles=legend_handles, loc="upper right", fontsize=9,
        facecolor="#16213e", edgecolor="#444", labelcolor="white",
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.15, color="white")

    plt.tight_layout()

    # Save to file
    output_path = os.path.join(
        os.path.dirname(__file__), "trajectory_prediction_demo.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Demo plot saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
