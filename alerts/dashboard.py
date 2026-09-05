"""
Streamlit Dashboard — V2I Alert Simulation Display.

Provides a real-time monitoring interface for the Intersection Collision
Prediction System. Reads alert data from the JSONL log file and displays
metrics, active alerts, and alert history.

Run:
    streamlit run alerts/dashboard.py

This is a SIMULATION dashboard for academic demonstration purposes.
No real V2I/C-V2X communication occurs.
"""

import json
import os
import time
import random
from datetime import datetime
from collections import Counter

import streamlit as st


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
# Custom CSS — premium dark dashboard with depth and polish
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ---- Global Reset ---- */
    .stApp {
        background: #0b0e17;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .stApp > header { background: transparent !important; }

    /* ---- Hide default Streamlit branding ---- */
    #MainMenu, footer, .stDeployButton { display: none !important; }

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0b0e17; }
    ::-webkit-scrollbar-thumb { background: #2a2f45; border-radius: 3px; }

    /* ---- Top banner ---- */
    .top-banner {
        background: linear-gradient(135deg, #111827 0%, #1e293b 50%, #111827 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
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
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .top-banner .subtitle {
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 400;
        margin: 0;
    }
    .top-banner .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #4ade80;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 100px;
        margin-top: 10px;
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
        gap: 16px;
        margin-bottom: 28px;
    }
    .stat-card {
        background: linear-gradient(145deg, #131825, #171d2e);
        border: 1px solid #1e2740;
        border-radius: 14px;
        padding: 22px 24px;
        transition: border-color 0.3s, transform 0.2s;
    }
    .stat-card:hover {
        border-color: #334155;
        transform: translateY(-2px);
    }
    .stat-card .stat-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    .stat-card .stat-value {
        color: #f1f5f9;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-card .stat-sub {
        color: #475569;
        font-size: 0.72rem;
        margin-top: 6px;
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
        margin: 28px 0 16px 0;
    }
    .section-header h2 {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
    }
    .section-header .badge {
        background: rgba(99, 102, 241, 0.12);
        color: #818cf8;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 100px;
        letter-spacing: 0.04em;
    }
    .section-line {
        height: 1px;
        background: linear-gradient(90deg, #1e2740, transparent);
        margin-bottom: 18px;
    }

    /* ---- Alert cards ---- */
    .alert-card {
        background: linear-gradient(145deg, #131825, #151c2c);
        border: 1px solid #1e2740;
        border-left: 3px solid;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 14px;
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    .alert-card:hover {
        border-color: #334155;
    }
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
        margin-bottom: 14px;
    }
    .alert-card .card-title {
        color: #e2e8f0;
        font-size: 0.92rem;
        font-weight: 600;
    }
    .severity-badge {
        font-size: 0.68rem;
        font-weight: 700;
        padding: 3px 10px;
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
        gap: 10px;
    }
    .alert-card .card-field {
        display: flex;
        flex-direction: column;
    }
    .alert-card .field-label {
        color: #475569;
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 3px;
    }
    .alert-card .field-value {
        color: #cbd5e1;
        font-size: 0.88rem;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
    }
    .alert-card .ttc-highlight {
        color: #f1f5f9;
        font-size: 1.15rem;
        font-weight: 700;
    }

    /* ---- Info boxes ---- */
    .info-box {
        background: linear-gradient(145deg, #0c1120, #111827);
        border: 1px solid #1e2740;
        border-radius: 12px;
        padding: 20px 24px;
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    .info-box .info-title {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }

    /* ---- Empty state ---- */
    .empty-state {
        text-align: center;
        padding: 48px 24px;
        color: #475569;
    }
    .empty-state .empty-icon {
        font-size: 2.5rem;
        margin-bottom: 12px;
        opacity: 0.4;
    }
    .empty-state p {
        font-size: 0.9rem;
        max-width: 360px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* ---- Table styling ---- */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        border: 1px solid #1e2740 !important;
        border-radius: 12px;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #0d1018 !important;
        border-right: 1px solid #1a1f33;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
        font-size: 0.9rem;
    }

    /* ---- Override Streamlit metric widget ---- */
    div[data-testid="stMetric"] { display: none; }

    /* ---- Timeline bar ---- */
    .timeline-bar {
        display: flex;
        align-items: end;
        gap: 3px;
        height: 60px;
        margin: 8px 0 4px 0;
    }
    .timeline-bar .bar {
        flex: 1;
        border-radius: 3px 3px 0 0;
        min-height: 3px;
        transition: height 0.4s ease;
    }
    .timeline-bar .bar-high { background: linear-gradient(to top, #ef4444, #f87171); }
    .timeline-bar .bar-medium { background: linear-gradient(to top, #f59e0b, #fbbf24); }
    .timeline-bar .bar-none { background: #1e2740; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

LOG_FILE = "alerts/alert_log.jsonl"


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


def format_timestamp_short(ts):
    """Short timestamp for cards."""
    try:
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    except (ValueError, TypeError, OSError):
        return f"{ts:.1f}s"


def severity_badge_html(severity):
    """Return HTML for a severity badge."""
    return f'<span class="severity-badge severity-{severity}">{severity}</span>'


def render_alert_card(alert):
    """Render a single alert as a premium styled card."""
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
                <span class="field-value">ID {vid_a}  &harr;  ID {vid_b}</span>
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


def build_mini_timeline(alerts, num_slots=20):
    """Build an HTML mini-timeline bar chart of recent alert severity."""
    if not alerts:
        return ""

    # Group alerts into time slots
    bars = []
    recent = alerts[-num_slots:] if len(alerts) >= num_slots else alerts
    for a in recent:
        sev = a.get("severity", "none")
        height = 55 if sev == "high" else 35 if sev == "medium" else 6
        css = f"bar-{sev}" if sev in ("high", "medium") else "bar-none"
        bars.append(f'<div class="bar {css}" style="height: {height}px;"></div>')

    # Pad remaining slots
    while len(bars) < num_slots:
        bars.insert(0, '<div class="bar bar-none" style="height: 3px;"></div>')

    return f'<div class="timeline-bar">{"".join(bars)}</div>'


# ---------------------------------------------------------------------------
# Dashboard layout
# ---------------------------------------------------------------------------

def main():
    # ---- Top banner ----
    st.markdown("""
    <div class="top-banner">
        <h1>Intersection Collision Prediction System</h1>
        <p class="subtitle">
            Vehicle-to-Infrastructure (V2I) Alert Monitor &mdash;
            Real-time collision risk assessment using computer vision
        </p>
        <div class="status-chip">
            <div class="status-dot"></div>
            System Active
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("### Settings")
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_rate = st.slider("Refresh interval (sec)", 1, 10, 3)
        max_display = st.slider("Max alerts shown", 10, 100, 50)

        st.markdown("---")

        st.markdown("### About This System")
        st.markdown("""
        <div class="info-box">
            <div class="info-title">Academic Project</div>
            This dashboard monitors a simulated V2I alert pipeline
            for intersection collision prediction. It processes vehicle
            trajectories from a fixed roadside camera and computes
            Time-to-Collision (TTC) for all nearby vehicle pairs.
            <br><br>
            <strong>Pipeline:</strong><br>
            Camera &rarr; Detection &rarr; Tracking &rarr;
            Prediction &rarr; Risk Scoring &rarr; Alerts
            <br><br>
            <em style="color: #64748b;">
            Simulated V2I &mdash; no real C-V2X/AIS-230 hardware.
            </em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        if st.button("Clear Alert Log", use_container_width=True):
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "w") as f:
                    pass
            st.success("Log cleared!")
            time.sleep(0.5)
            st.rerun()

        st.markdown(f"""
        <div style="color: #334155; font-size: 0.72rem; margin-top: 16px; text-align: center;">
            Log: {LOG_FILE}<br>
            Refreshing every {refresh_rate}s
        </div>
        """, unsafe_allow_html=True)

    # ---- Load alert data ----
    alerts = load_alerts_from_log()
    alerts_reversed = list(reversed(alerts))[:max_display]

    high_alerts = [a for a in alerts if a.get("severity") == "high"]
    medium_alerts = [a for a in alerts if a.get("severity") == "medium"]

    unique_pairs = set()
    for a in alerts:
        v = a.get("vehicles_involved", [])
        if len(v) == 2:
            unique_pairs.add(tuple(sorted(v)))

    # Compute dedup ratio
    dedup_text = ""
    if len(alerts) > 0:
        # This is a rough approximation from the log
        dedup_text = f"{len(alerts)} logged"

    # ---- Stats cards (custom HTML for premium look) ----
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card accent-blue">
            <div class="stat-label">Total Alerts</div>
            <div class="stat-value">{len(alerts)}</div>
            <div class="stat-sub">{dedup_text}</div>
        </div>
        <div class="stat-card accent-red">
            <div class="stat-label">High Severity</div>
            <div class="stat-value">{len(high_alerts)}</div>
            <div class="stat-sub">TTC &lt; 1.0 second</div>
        </div>
        <div class="stat-card accent-amber">
            <div class="stat-label">Medium Severity</div>
            <div class="stat-value">{len(medium_alerts)}</div>
            <div class="stat-sub">1.0s &le; TTC &lt; 2.0s</div>
        </div>
        <div class="stat-card accent-emerald">
            <div class="stat-label">Vehicle Pairs</div>
            <div class="stat-value">{len(unique_pairs)}</div>
            <div class="stat-sub">Unique pairs flagged</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Alert timeline mini-bar ----
    if alerts:
        st.markdown("""
        <div class="section-header">
            <h2>Alert Timeline</h2>
            <span class="badge">Last 20 events</span>
        </div>
        <div class="section-line"></div>
        """, unsafe_allow_html=True)

        timeline_html = build_mini_timeline(alerts, num_slots=20)
        st.markdown(timeline_html, unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; gap: 16px; margin: 6px 0 24px 0;">
            <span style="color: #475569; font-size: 0.72rem; display: flex; align-items: center; gap: 5px;">
                <span style="width:10px;height:10px;background:#ef4444;border-radius:2px;display:inline-block;"></span>
                High
            </span>
            <span style="color: #475569; font-size: 0.72rem; display: flex; align-items: center; gap: 5px;">
                <span style="width:10px;height:10px;background:#f59e0b;border-radius:2px;display:inline-block;"></span>
                Medium
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ---- Recent alerts ----
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
            <p>No alerts detected yet. Run the pipeline to start
            monitoring intersection traffic.</p>
            <p style="margin-top: 12px; color: #334155; font-size: 0.8rem;">
                <code>python pipeline/run_demo.py</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show recent alerts as cards in a 2-column grid
        recent = alerts_reversed[:6]
        cols = st.columns(2)
        for idx, alert in enumerate(recent):
            with cols[idx % 2]:
                st.markdown(render_alert_card(alert), unsafe_allow_html=True)

    # ---- Full alert history table ----
    st.markdown(f"""
    <div class="section-header">
        <h2>Alert History</h2>
        <span class="badge">{len(alerts_reversed)} entries</span>
    </div>
    <div class="section-line"></div>
    """, unsafe_allow_html=True)

    if alerts_reversed:
        table_data = []
        for a in alerts_reversed:
            vehicles = a.get("vehicles_involved", [])
            sev = a.get("severity", "—")
            table_data.append({
                "Time": format_timestamp(a.get("timestamp", 0)),
                "Vehicle A": vehicles[0] if len(vehicles) > 0 else "—",
                "Vehicle B": vehicles[1] if len(vehicles) > 1 else "—",
                "TTC (sec)": f"{a.get('time_to_collision', 0):.3f}",
                "Severity": sev.upper(),
                "Location": a.get("location", "—"),
            })

        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No alert history available.")

    # ---- Footer ----
    st.markdown("""
    <div style="text-align: center; padding: 32px 0 16px 0; color: #1e293b; font-size: 0.7rem;">
        Intersection Collision Prediction System &mdash; AI-Based Alert System
        &mdash; Academic Project &copy; 2026
    </div>
    """, unsafe_allow_html=True)

    # ---- Auto-refresh ----
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
