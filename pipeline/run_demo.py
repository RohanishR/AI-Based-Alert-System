"""
End-to-End Pipeline Demo — connects all 6 modules.

    CameraFeed → VehicleDetector → VehicleTracker →
    TrajectoryPredictor → RiskScorer → AlertGenerator →
    Streamlit Dashboard (via JSONL log)

Modes:
    1. Simulated demo (default) — uses synthetic tracking data to
       demonstrate Modules 4→5→6 without requiring a camera/video.
    2. Full pipeline — processes a video file through all 6 modules.

Run simulated demo:
    python pipeline/run_demo.py

Run full pipeline (requires video file):
    python pipeline/run_demo.py --video path/to/video.mp4

Run dashboard (in separate terminal):
    streamlit run alerts/dashboard.py
"""

import argparse
import os
import sys
import time
import math
import json

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prediction.kalman_filter import TrajectoryPredictor
from risk_scoring.ttc import RiskScorer
from alerts.alert_generator import AlertGenerator


# ---------------------------------------------------------------------------
# Simulated tracking data generators
# ---------------------------------------------------------------------------

def generate_collision_scenario(n_frames=100, fps=10):
    """
    Generate two vehicles on a collision course at an intersection.

    Vehicle A: approaching from the left, moving right.
    Vehicle B: approaching from the bottom, moving up.
    They converge near (400, 300) around frame 50-60.
    """
    dt = 1.0 / fps
    frames = []

    for i in range(n_frames):
        t = i * dt
        tracked_objects = []

        # Vehicle A: left → right, slowing slightly
        xa = 100 + 60 * t
        ya = 300 + 5 * math.sin(t * 0.5)  # slight wobble
        history_a = [
            (100 + 60 * j * dt, 300 + 5 * math.sin(j * dt * 0.5), j * dt)
            for j in range(max(0, i - 14), i + 1)
        ]
        tracked_objects.append({
            "track_id": 1,
            "class": "car",
            "bbox": (xa - 30, ya - 20, xa + 30, ya + 20),
            "history": history_a,
        })

        # Vehicle B: bottom → top (decreasing y), constant speed
        xb = 380 + 3 * math.sin(t * 0.3)
        yb = 550 - 50 * t
        history_b = [
            (380 + 3 * math.sin(j * dt * 0.3), 550 - 50 * j * dt, j * dt)
            for j in range(max(0, i - 14), i + 1)
        ]
        tracked_objects.append({
            "track_id": 2,
            "class": "motorcycle",
            "bbox": (xb - 15, yb - 15, xb + 15, yb + 15),
            "history": history_b,
        })

        # Vehicle C: appears at frame 30, moving diagonally
        if i >= 30:
            phase_t = (i - 30) * dt
            xc = 500 - 40 * phase_t
            yc = 200 + 30 * phase_t
            history_c = [
                (500 - 40 * (j - 30) * dt, 200 + 30 * (j - 30) * dt, j * dt)
                for j in range(max(30, i - 14), i + 1)
            ]
            tracked_objects.append({
                "track_id": 3,
                "class": "auto_rickshaw",
                "bbox": (xc - 25, yc - 18, xc + 25, yc + 18),
                "history": history_c,
            })

        frames.append((t, tracked_objects))

    return frames


# ---------------------------------------------------------------------------
# Full pipeline (video mode)
# ---------------------------------------------------------------------------

def run_full_pipeline(video_path):
    """
    Run the complete 6-module pipeline on a video file.
    Requires Modules 1-3 (camera_feed, detection, tracking) to be functional.
    """
    try:
        from camera_feed import CameraFeed
        from detection.detector import VehicleDetector
        from tracking.tracker import VehicleTracker
    except ImportError as e:
        print(f"Error importing pipeline modules: {e}")
        print("Make sure Modules 1-3 are properly installed.")
        sys.exit(1)

    print("=" * 60)
    print("  Full Pipeline — All 6 Modules")
    print("=" * 60)
    print(f"  Video: {video_path}")
    print()

    # Initialize all modules
    camera = CameraFeed(video_path, frame_skip=2)
    camera.open()
    print(f"  Video: {camera.width}x{camera.height} @ {camera.fps:.0f}fps")

    detector = VehicleDetector()
    tracker = VehicleTracker()
    predictor = TrajectoryPredictor()
    scorer = RiskScorer()
    alerter = AlertGenerator()
    alerter.clear_log()

    frame_count = 0
    total_alerts = 0

    print("\n  Processing frames...\n")

    while True:
        frame_data = camera.read()
        if frame_data is None:
            break

        frame_count += 1
        timestamp = frame_data["timestamp"]

        # Module 2: Detection
        detections, _ = detector.detect(frame_data)

        # Module 3: Tracking
        tracked_objects = tracker.update(detections, timestamp)

        if not tracked_objects:
            continue

        # Module 4: Prediction
        predictions = predictor.update(tracked_objects, timestamp)

        # Module 5: Risk Scoring
        risk_events = scorer.evaluate(predictions)

        # Module 6: Alert Generation
        if risk_events:
            new_alerts = alerter.process(risk_events, timestamp)
            total_alerts += len(new_alerts)
            for alert in new_alerts:
                print(f"  [ALERT] | Vehicles {alert['vehicles_involved']} | "
                      f"TTC={alert['time_to_collision']:.2f}s | "
                      f"Severity={alert['severity'].upper()}")

        if frame_count % 50 == 0:
            print(f"  ... processed {frame_count} frames, "
                  f"{len(tracked_objects)} vehicles tracked, "
                  f"{total_alerts} alerts total")

    camera.release()
    stats = alerter.get_stats()
    print(f"\n  Done! {frame_count} frames processed.")
    print(f"  Total alerts generated: {stats['total_alerts']}")
    print(f"\n  Alert log: alerts/alert_log.jsonl")
    print(f"  Run dashboard: streamlit run alerts/dashboard.py")


# ---------------------------------------------------------------------------
# Simulated demo (no video required)
# ---------------------------------------------------------------------------

def run_simulated_demo():
    """
    Run Modules 4→5→6 on synthetic tracking data to demonstrate
    the prediction→risk→alert pipeline without needing camera/detection.
    """
    print("=" * 60)
    print("  Simulated Pipeline Demo — Modules 4, 5, 6")
    print("=" * 60)
    print("  Using synthetic vehicle trajectories (no video required)")
    print()

    predictor = TrajectoryPredictor()
    scorer = RiskScorer()
    alerter = AlertGenerator()
    alerter.clear_log()

    frames = generate_collision_scenario(n_frames=100, fps=10)

    total_alerts = 0
    risk_event_count = 0

    print("  Simulating 100 frames (10 seconds at 10 FPS)...\n")

    for timestamp, tracked_objects in frames:
        # Module 4: Prediction
        predictions = predictor.update(tracked_objects, timestamp)

        # Module 5: Risk Scoring
        risk_events = scorer.evaluate(predictions)
        risk_event_count += len(risk_events)

        # Module 6: Alert Generation
        if risk_events:
            new_alerts = alerter.process(risk_events, timestamp)
            total_alerts += len(new_alerts)

            for alert in new_alerts:
                print(f"  t={timestamp:5.1f}s | [ALERT] "
                      f"| Vehicles {alert['vehicles_involved']} "
                      f"| TTC={alert['time_to_collision']:.2f}s "
                      f"| Severity={alert['severity'].upper()}")

    # Summary
    stats = alerter.get_stats()
    history = alerter.get_alert_history()

    print(f"\n{'=' * 60}")
    print(f"  SIMULATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Frames processed:       {len(frames)}")
    print(f"  Vehicles simulated:     3 (car, motorcycle, auto_rickshaw)")
    print(f"  Risk events detected:   {risk_event_count}")
    print(f"  Alerts generated:       {stats['total_alerts']}")
    print(f"  High severity:          {stats['recent_high']}")
    print(f"  Medium severity:        {stats['recent_medium']}")
    print(f"  Alerts suppressed:      {risk_event_count - stats['total_alerts']} (cooldown dedup)")
    print()

    # Print alert log contents
    log_path = "alerts/alert_log.jsonl"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            lines = f.readlines()
        print(f"  Alert log ({len(lines)} entries): {log_path}")
        print()
        for line in lines[:10]:
            alert = json.loads(line)
            print(f"    {json.dumps(alert, indent=None)}")
        if len(lines) > 10:
            print(f"    ... and {len(lines) - 10} more")

    print(f"\n  To view the dashboard:")
    print(f"    streamlit run alerts/dashboard.py\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Intersection Collision Prediction — Pipeline Demo"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to video file for full pipeline. If omitted, runs simulated demo."
    )
    args = parser.parse_args()

    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        run_full_pipeline(args.video)
    else:
        run_simulated_demo()


if __name__ == "__main__":
    main()
