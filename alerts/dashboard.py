"""
Streamlit Dashboard — V2I Alert Simulation with Live Tracking View.

Provides a real-time monitoring interface for the Intersection Collision
Prediction System. Processes the Demo video through the full pipeline
(detection, tracking, prediction, risk scoring, alerting) and displays
annotated frames with bounding boxes alongside alerts.

Run:
    streamlit run alerts/dashboard.py

This is a SIMULATION dashboard for academic demonstration purposes.
No real V2I/C-V2X communication occurs.
"""

import json
import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime
from collections import deque

import streamlit as st

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="V2I Alert System — Intersection Safety Monitor",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ---- Global ---- */
    .stApp {
        background: #0b0e17;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .stApp > header { background: transparent !important; }
    #MainMenu, footer, .stDeployButton { display: none !important; }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0b0e17; }
    ::-webkit-scrollbar-thumb { background: #2a2f45; border-radius: 3px; }

    /* ---- Top banner ---- */
    .top-banner {
        background: linear-gradient(135deg, #111827 0%, #1e293b 50%, #111827 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .top-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #6366f1, #8b5cf6, transparent);
    }
    .top-banner h1 {
        color: #f1f5f9;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0 0 4px 0;
        letter-spacing: -0.02em;
    }
    .top-banner .subtitle {
        color: #94a3b8;
        font-size: 0.85rem;
        margin: 0;
    }
    .top-banner .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #4ade80;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 100px;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .top-banner .status-dot {
        width: 7px; height: 7px;
        background: #4ade80;
        border-radius: 50%;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.4); }
        50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(74, 222, 128, 0); }
    }

    /* ---- Stat cards ---- */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 24px;
    }
    .stat-card {
        background: linear-gradient(145deg, #131825, #171d2e);
        border: 1px solid #1e2740;
        border-radius: 12px;
        padding: 18px 20px;
        transition: border-color 0.3s, transform 0.2s;
    }
    .stat-card:hover {
        border-color: #334155;
        transform: translateY(-2px);
    }
    .stat-card .stat-label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .stat-card .stat-value {
        color: #f1f5f9;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-card .stat-sub {
        color: #475569;
        font-size: 0.68rem;
        margin-top: 4px;
    }
    .stat-card.accent-red .stat-value { color: #f87171; }
    .stat-card.accent-amber .stat-value { color: #fbbf24; }
    .stat-card.accent-blue .stat-value { color: #60a5fa; }
    .stat-card.accent-emerald .stat-value { color: #34d399; }

    /* ---- Section headers ---- */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 24px 0 12px 0;
    }
    .section-header h2 {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 600;
        margin: 0;
    }
    .section-header .badge {
        background: rgba(99, 102, 241, 0.12);
        color: #818cf8;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 100px;
        letter-spacing: 0.04em;
    }
    .section-line {
        height: 1px;
        background: linear-gradient(90deg, #1e2740, transparent);
        margin-bottom: 14px;
    }

    /* ---- Alert cards ---- */
    .alert-card {
        background: linear-gradient(145deg, #131825, #151c2c);
        border: 1px solid #1e2740;
        border-left: 3px solid;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 10px;
        transition: border-color 0.3s;
    }
    .alert-card:hover { border-color: #334155; }
    .alert-card-high {
        border-left-color: #ef4444;
        box-shadow: inset 0 0 30px rgba(239, 68, 68, 0.03);
    }
    .alert-card-medium {
        border-left-color: #f59e0b;
        box-shadow: inset 0 0 30px rgba(245, 158, 11, 0.03);
    }
    .alert-card .card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .alert-card .card-title {
        color: #e2e8f0;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .severity-badge {
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 100px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .severity-high {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .severity-medium {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    .alert-card .card-body {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }
    .alert-card .field-label {
        color: #475569;
        font-size: 0.65rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
    }
    .alert-card .field-value {
        color: #cbd5e1;
        font-size: 0.82rem;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
    }
    .alert-card .ttc-highlight {
        color: #f1f5f9;
        font-size: 1.05rem;
        font-weight: 700;
    }

    /* ---- Video frame ---- */
    .video-container {
        background: #0d1018;
        border: 1px solid #1e2740;
        border-radius: 12px;
        overflow: hidden;
        position: relative;
    }
    .video-overlay {
        position: absolute;
        top: 10px;
        left: 10px;
        background: rgba(0,0,0,0.6);
        color: #4ade80;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        padding: 4px 10px;
        border-radius: 6px;
        backdrop-filter: blur(4px);
    }

    /* ---- Info box ---- */
    .info-box {
        background: linear-gradient(145deg, #0c1120, #111827);
        border: 1px solid #1e2740;
        border-radius: 10px;
        padding: 16px 20px;
        color: #94a3b8;
        font-size: 0.8rem;
        line-height: 1.5;
    }
    .info-box .info-title {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 6px;
    }

    /* ---- Empty state ---- */
    .empty-state {
        text-align: center;
        padding: 36px 20px;
        color: #475569;
    }
    .empty-state .empty-icon { font-size: 2rem; margin-bottom: 10px; opacity: 0.4; }
    .empty-state p { font-size: 0.85rem; max-width: 320px; margin: 0 auto; line-height: 1.4; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #0d1018 !important;
        border-right: 1px solid #1a1f33;
    }

    /* ---- Hide default metric ---- */
    div[data-testid="stMetric"] { display: none; }

    /* ---- Processing bar ---- */
    .progress-bar-container {
        background: #131825;
        border: 1px solid #1e2740;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
    }
    .progress-bar-container .progress-label {
        color: #94a3b8;
        font-size: 0.78rem;
        margin-bottom: 6px;
    }
    .progress-bar-container .progress-value {
        color: #e2e8f0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

LOG_FILE = "alerts/alert_log.jsonl"
VIDEO_PATH = "data/raw_videos/Demo.mp4"


def load_alerts_from_log(log_path=LOG_FILE):
    """Load all alerts from the JSONL log file."""
    alerts = []
    if not os.path.exists(log_path):
        return alerts
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        alerts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass
    return alerts


def format_timestamp(ts):
    """Convert a Unix timestamp to a readable string."""
    try:
        return datetime.fromtimestamp(ts).strftime("%I:%M:%S %p")
    except (ValueError, TypeError, OSError):
        return f"{ts:.1f}s"


def severity_badge_html(severity):
    """Return HTML for a severity badge."""
    return f'<span class="severity-badge severity-{severity}">{severity}</span>'


def render_alert_card(alert):
    """Render a single alert as a styled card."""
    severity = alert.get("severity", "medium")
    vehicles = alert.get("vehicles_involved", [])
    ttc = alert.get("time_to_collision", 0)
    ts = format_timestamp(alert.get("timestamp", 0))
    location = alert.get("location", "Unknown")

    vid_a = vehicles[0] if len(vehicles) > 0 else "?"
    vid_b = vehicles[1] if len(vehicles) > 1 else "?"

    return f"""
    <div class="alert-card alert-card-{severity}">
        <div class="card-top">
            <span class="card-title">Collision Warning</span>
            {severity_badge_html(severity)}
        </div>
        <div class="card-body">
            <div class="card-field">
                <span class="field-label">Vehicles</span>
                <span class="field-value">ID {vid_a} &harr; ID {vid_b}</span>
            </div>
            <div class="card-field">
                <span class="field-label">Time to Collision</span>
                <span class="field-value ttc-highlight">{ttc:.2f}s</span>
            </div>
            <div class="card-field">
                <span class="field-label">Location</span>
                <span class="field-value">{location}</span>
            </div>
            <div class="card-field">
                <span class="field-label">Detected At</span>
                <span class="field-value">{ts}</span>
            </div>
        </div>
    </div>
    """


def draw_risk_overlay(frame, risk_events, tracked_objects):
    """
    Draw risk event warnings directly on the video frame.
    Draws a red/orange line between vehicles that are at risk.
    """
    if not risk_events or not tracked_objects:
        return frame

    # Build a lookup: track_id -> bbox center
    centers = {}
    for obj in tracked_objects:
        tid = obj["track_id"]
        x1, y1, x2, y2 = obj["bbox"]
        centers[tid] = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    for event in risk_events:
        id_a, id_b = event["vehicle_pair"]
        severity = event["severity"]
        ttc = event["ttc"]

        if id_a not in centers or id_b not in centers:
            continue

        ca = centers[id_a]
        cb = centers[id_b]

        # Color: red for high, orange for medium
        if severity == "high":
            color = (0, 0, 255)
            thickness = 3
        else:
            color = (0, 165, 255)
            thickness = 2

        # Draw dashed-style warning line between vehicles
        cv2.line(frame, ca, cb, color, thickness, cv2.LINE_AA)

        # Draw TTC label at midpoint
        mid = ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2)
        label = f"TTC:{ttc:.1f}s"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (mid[0] - 2, mid[1] - h - 6),
                      (mid[0] + w + 4, mid[1] + 4), color, -1)
        cv2.putText(frame, label, (mid[0], mid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw warning icon near vehicles
        for c in [ca, cb]:
            cv2.circle(frame, c, 8, color, -1)
            cv2.circle(frame, c, 12, color, 2)

    return frame


def draw_prediction_trails(frame, predictions):
    """
    Draw predicted trajectory as dotted future path on the frame.
    """
    for pred in predictions:
        pts = pred.get("predictions", [])
        if len(pts) < 2:
            continue

        for i in range(len(pts) - 1):
            x1, y1 = int(pts[i][0]), int(pts[i][1])
            x2, y2 = int(pts[i + 1][0]), int(pts[i + 1][1])
            # Fading cyan dots for predicted path
            alpha = 1.0 - (i / len(pts)) * 0.6
            color = (int(255 * alpha), int(215 * alpha), 0)  # Gold fading
            cv2.circle(frame, (x2, y2), 4, color, -1)
            cv2.line(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Pipeline processing with live view
# ---------------------------------------------------------------------------

def process_video_live(video_path, frame_placeholder, stats_placeholder,
                       alert_feed_placeholder, progress_bar,
                       frame_skip=3, max_frames=None):
    """
    Process the video through all 6 modules and stream annotated
    frames to the Streamlit dashboard in real time.
    """
    from camera_feed import CameraFeed
    from detection.detector import VehicleDetector
    from tracking.tracker import VehicleTracker
    from prediction.kalman_filter import TrajectoryPredictor
    from risk_scoring.ttc import RiskScorer
    from alerts.alert_generator import AlertGenerator

    # Initialize all modules
    camera = CameraFeed(video_path, frame_skip=frame_skip)
    camera.open()

    detector = VehicleDetector()
    tracker = VehicleTracker()
    predictor = TrajectoryPredictor()
    scorer = RiskScorer()
    alerter = AlertGenerator()
    alerter.clear_log()

    total_frames = camera.total_frames // frame_skip
    frame_count = 0
    total_alerts = 0
    high_count = 0
    medium_count = 0
    vehicles_seen = set()
    recent_alerts = deque(maxlen=8)

    while True:
        frame_data = camera.read()
        if frame_data is None:
            break

        frame_count += 1
        timestamp = frame_data["timestamp"]

        if max_frames and frame_count > max_frames:
            break

        # Module 2: Detection
        detections, inf_time = detector.detect(frame_data)

        # Module 3: Tracking
        tracked_objects = tracker.update(detections, timestamp)

        # Track unique vehicle IDs
        for obj in tracked_objects:
            vehicles_seen.add(obj["track_id"])

        # Module 4: Prediction
        predictions = []
        risk_events = []
        if tracked_objects:
            predictions = predictor.update(tracked_objects, timestamp)

            # Module 5: Risk Scoring
            risk_events = scorer.evaluate(predictions)

            # Module 6: Alert Generation
            if risk_events:
                new_alerts = alerter.process(risk_events, timestamp)
                total_alerts += len(new_alerts)
                for a in new_alerts:
                    if a["severity"] == "high":
                        high_count += 1
                    else:
                        medium_count += 1
                    recent_alerts.append(a)

        # ---- Draw annotations on frame ----
        annotated = frame_data["frame"].copy()

        # Draw tracking boxes (from Module 3)
        annotated = tracker.draw_tracks(annotated, tracked_objects)

        # Draw predicted trajectories
        annotated = draw_prediction_trails(annotated, predictions)

        # Draw risk overlays (lines between at-risk vehicles)
        annotated = draw_risk_overlay(annotated, risk_events, tracked_objects)

        # Draw HUD overlay
        fps_val = 1.0 / inf_time if inf_time > 0 else 0
        cv2.putText(annotated, f"FPS: {fps_val:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        cv2.putText(annotated, f"Vehicles: {len(tracked_objects)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        cv2.putText(annotated, f"Alerts: {total_alerts}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Update the video frame
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update progress bar
        progress = min(frame_count / max(total_frames, 1), 1.0)
        progress_bar.progress(progress, text=f"Processing frame {frame_count}/{total_frames}")

        # Update stats
        stats_placeholder.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card accent-blue">
                <div class="stat-label">Vehicles Tracked</div>
                <div class="stat-value">{len(tracked_objects)}</div>
                <div class="stat-sub">{len(vehicles_seen)} unique total</div>
            </div>
            <div class="stat-card accent-red">
                <div class="stat-label">High Risk</div>
                <div class="stat-value">{high_count}</div>
                <div class="stat-sub">TTC &lt; 1.0s</div>
            </div>
            <div class="stat-card accent-amber">
                <div class="stat-label">Medium Risk</div>
                <div class="stat-value">{medium_count}</div>
                <div class="stat-sub">1.0s &le; TTC &lt; 2.0s</div>
            </div>
            <div class="stat-card accent-emerald">
                <div class="stat-label">Total Alerts</div>
                <div class="stat-value">{total_alerts}</div>
                <div class="stat-sub">After deduplication</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Update alert feed (sidebar-style live feed)
        if recent_alerts:
            cards_html = ""
            for alert in reversed(recent_alerts):
                cards_html += render_alert_card(alert)
            alert_feed_placeholder.markdown(cards_html, unsafe_allow_html=True)

    camera.release()

    return {
        "frames": frame_count,
        "total_alerts": total_alerts,
        "high": high_count,
        "medium": medium_count,
        "vehicles": len(vehicles_seen),
    }


# ---------------------------------------------------------------------------
# Static dashboard (view past results from log)
# ---------------------------------------------------------------------------

def show_static_dashboard():
    """Show the dashboard with previously logged alerts."""
    alerts = load_alerts_from_log()
    alerts_reversed = list(reversed(alerts))[:50]

    high_alerts = [a for a in alerts if a.get("severity") == "high"]
    medium_alerts = [a for a in alerts if a.get("severity") == "medium"]
    unique_pairs = set()
    for a in alerts:
        v = a.get("vehicles_involved", [])
        if len(v) == 2:
            unique_pairs.add(tuple(sorted(v)))

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card accent-blue">
            <div class="stat-label">Total Alerts</div>
            <div class="stat-value">{len(alerts)}</div>
            <div class="stat-sub">{len(alerts)} logged</div>
        </div>
        <div class="stat-card accent-red">
            <div class="stat-label">High Severity</div>
            <div class="stat-value">{len(high_alerts)}</div>
            <div class="stat-sub">TTC &lt; 1.0s</div>
        </div>
        <div class="stat-card accent-amber">
            <div class="stat-label">Medium Severity</div>
            <div class="stat-value">{len(medium_alerts)}</div>
            <div class="stat-sub">1.0s &le; TTC &lt; 2.0s</div>
        </div>
        <div class="stat-card accent-emerald">
            <div class="stat-label">Unique Pairs</div>
            <div class="stat-value">{len(unique_pairs)}</div>
            <div class="stat-sub">Vehicle pairs flagged</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Recent alert cards
    st.markdown(f"""
    <div class="section-header">
        <h2>Recent Alerts</h2>
        <span class="badge">{min(len(alerts_reversed), 6)} most recent</span>
    </div>
    <div class="section-line"></div>
    """, unsafe_allow_html=True)

    if not alerts_reversed:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📡</div>
            <p>No alerts yet. Click <strong>Start Live Tracking</strong>
            in the sidebar to process the video.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        recent = alerts_reversed[:6]
        cols = st.columns(2)
        for idx, alert in enumerate(recent):
            with cols[idx % 2]:
                st.markdown(render_alert_card(alert), unsafe_allow_html=True)

    # Full history table
    if alerts_reversed:
        st.markdown(f"""
        <div class="section-header">
            <h2>Alert History</h2>
            <span class="badge">{len(alerts_reversed)} entries</span>
        </div>
        <div class="section-line"></div>
        """, unsafe_allow_html=True)

        table_data = []
        for a in alerts_reversed:
            vehicles = a.get("vehicles_involved", [])
            table_data.append({
                "Time": format_timestamp(a.get("timestamp", 0)),
                "Vehicle A": vehicles[0] if len(vehicles) > 0 else "-",
                "Vehicle B": vehicles[1] if len(vehicles) > 1 else "-",
                "TTC (sec)": f"{a.get('time_to_collision', 0):.3f}",
                "Severity": a.get("severity", "-").upper(),
                "Location": a.get("location", "-"),
            })
        st.dataframe(table_data, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main():
    # ---- Header ----
    st.markdown("""
    <div class="top-banner">
        <h1>Intersection Collision Prediction System</h1>
        <p class="subtitle">
            V2I Alert Monitor &mdash; Real-time collision risk assessment with live vehicle tracking
        </p>
        <div class="status-chip">
            <div class="status-dot"></div>
            System Active
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("### Controls")

        mode = st.radio(
            "Dashboard Mode",
            ["Live Tracking", "Alert History"],
            index=0,
            help="Live Tracking processes the video with bounding boxes. Alert History shows past results."
        )

        st.markdown("---")

        if mode == "Live Tracking":
            st.markdown("### Video Settings")
            frame_skip = st.slider("Frame Skip", 1, 10, 2,
                                   help="Process every Nth frame. Higher = faster but less smooth.")

            video_file = VIDEO_PATH
            if not os.path.exists(video_file):
                st.error(f"Video not found: {video_file}")
                video_file = None

            start_btn = st.button("Start Live Tracking",
                                  type="primary", use_container_width=True)
        else:
            start_btn = False
            frame_skip = 3

            if st.button("Refresh Data", use_container_width=True):
                st.rerun()

            if st.button("Clear Alert Log", use_container_width=True):
                if os.path.exists(LOG_FILE):
                    with open(LOG_FILE, "w") as f:
                        pass
                st.success("Log cleared!")
                time.sleep(0.5)
                st.rerun()

        st.markdown("---")
        st.markdown("""
        <div class="info-box">
            <div class="info-title">About</div>
            This dashboard processes intersection traffic video through
            6 AI modules: Camera Feed, YOLO Detection, ByteTrack Tracking,
            Kalman Filter Prediction, TTC Risk Scoring, and V2I Alert Generation.
            <br><br>
            <em style="color: #64748b;">
            Simulated V2I &mdash; no real C-V2X hardware.
            </em>
        </div>
        """, unsafe_allow_html=True)

    # ---- Main content ----
    if mode == "Live Tracking" and start_btn:
        # Live tracking mode — process video with full pipeline
        st.markdown("""
        <div class="section-header">
            <h2>Live Vehicle Tracking</h2>
            <span class="badge">PROCESSING</span>
        </div>
        <div class="section-line"></div>
        """, unsafe_allow_html=True)

        # Create layout: video on left, live alerts on right
        video_col, alert_col = st.columns([3, 2])

        with video_col:
            frame_placeholder = st.empty()
            progress_bar = st.progress(0, text="Initializing pipeline...")

        with alert_col:
            st.markdown("""
            <div class="section-header">
                <h2>Live Alert Feed</h2>
                <span class="badge">REAL-TIME</span>
            </div>
            <div class="section-line"></div>
            """, unsafe_allow_html=True)
            alert_feed_placeholder = st.empty()

        # Stats bar below video
        stats_placeholder = st.empty()

        # Process the video
        results = process_video_live(
            VIDEO_PATH,
            frame_placeholder,
            stats_placeholder,
            alert_feed_placeholder,
            progress_bar,
            frame_skip=frame_skip,
        )

        # Show completion message
        progress_bar.progress(1.0, text="Processing complete!")

        st.markdown(f"""
        <div class="info-box" style="margin-top: 16px;">
            <div class="info-title">Processing Complete</div>
            Processed <strong>{results['frames']}</strong> frames &mdash;
            detected <strong>{results['vehicles']}</strong> unique vehicles &mdash;
            generated <strong>{results['total_alerts']}</strong> alerts
            ({results['high']} high, {results['medium']} medium severity).
            <br><br>
            Switch to <strong>Alert History</strong> mode to browse the full log.
        </div>
        """, unsafe_allow_html=True)

    elif mode == "Live Tracking" and not start_btn:
        # Waiting state
        st.markdown("""
        <div class="section-header">
            <h2>Live Vehicle Tracking</h2>
            <span class="badge">STANDBY</span>
        </div>
        <div class="section-line"></div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🎥</div>
            <p>Click <strong>Start Live Tracking</strong> in the sidebar to begin
            processing the Demo video with real-time bounding boxes,
            trajectory predictions, and collision alerts.</p>
        </div>
        """, unsafe_allow_html=True)

        # Show existing alerts if any
        alerts = load_alerts_from_log()
        if alerts:
            st.markdown(f"""
            <div class="section-header">
                <h2>Previous Results</h2>
                <span class="badge">{len(alerts)} alerts from last run</span>
            </div>
            <div class="section-line"></div>
            """, unsafe_allow_html=True)
            show_static_dashboard()

    else:
        # Alert history mode
        show_static_dashboard()

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 28px 0 12px 0; color: #1e293b; font-size: 0.68rem;">
        Intersection Collision Prediction System &mdash; AI-Based Alert System &mdash; Academic Project
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
