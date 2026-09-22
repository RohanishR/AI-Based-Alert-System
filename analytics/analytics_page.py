"""
Analytics Page — Streamlit page rendering traffic & safety analytics.

Called by the main dashboard when the user navigates to the Analytics tab.
All data comes from AnalyticsEngine (which reads session_data.json + alert_log.jsonl).
"""

import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from analytics.analytics_engine import AnalyticsEngine


# ---------------------------------------------------------------------------
# Chart color palette (matches the dark dashboard theme)
# ---------------------------------------------------------------------------

COLORS = {
    "high": "#FF2A55",
    "medium": "#FFD700",
    "blue": "#00F0FF",
    "emerald": "#00FFCC",
    "purple": "#B537F2",
    "pink": "#FF3399",
    "cyan": "#00FFFF",
    "amber": "#FFB800",
    "slate": "#777777",
}

CLASS_COLORS = {
    "car": "#00F0FF",
    "truck": "#FFB800",
    "bus": "#B537F2",
    "motorcycle": "#00FFCC",
    "bicycle": "#00FFFF",
    "pedestrian": "#FF3399",
    "auto_rickshaw": "#FFD700",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#888888", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
)


# ---------------------------------------------------------------------------
# Render the analytics page
# ---------------------------------------------------------------------------

def render_analytics_page():
    """Main entry point — renders the full analytics page."""

    engine = AnalyticsEngine()

    if not engine.has_data():
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📊</div>
            <p>No analytics data available yet. Run the pipeline via
            <strong>Live Tracking</strong> to generate data, then come back here.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    kpis = engine.get_kpis()

    # ---- Sidebar filters ----
    with st.sidebar:
        st.markdown("### Filters")

        severity_filter = st.selectbox(
            "Severity", ["All", "high", "medium"], index=0
        )
        if severity_filter == "All":
            severity_filter = None

        class_dist = engine.get_class_distribution()
        class_options = ["All"] + [c["class"] for c in class_dist]
        class_filter = st.selectbox("Vehicle Class", class_options, index=0)
        if class_filter == "All":
            class_filter = None

        vehicle_id_input = st.text_input("Vehicle ID", "", help="Enter a track ID to filter")
        vehicle_id = int(vehicle_id_input) if vehicle_id_input.strip().isdigit() else None

    # ==================================================================
    # 1. KPI Overview
    # ==================================================================

    st.markdown("""
    <div class="section-header">
        <h2>Traffic & Safety Analytics</h2>
        <span class="badge">SESSION DATA</span>
    </div>
    <div class="section-line"></div>
    """, unsafe_allow_html=True)

    # Row 1: Primary KPIs
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card accent-blue">
            <div class="stat-label">Total Frames</div>
            <div class="stat-value">{kpis['total_frames']:,}</div>
            <div class="stat-sub">{kpis['processing_fps']} FPS</div>
        </div>
        <div class="stat-card accent-emerald">
            <div class="stat-label">Unique Vehicles</div>
            <div class="stat-value">{kpis['unique_tracks']}</div>
            <div class="stat-sub">{kpis['total_detections']:,} total detections</div>
        </div>
        <div class="stat-card accent-red">
            <div class="stat-label">Risk Events</div>
            <div class="stat-value">{kpis['total_risk_events']}</div>
            <div class="stat-sub">{kpis['high_risk_events']} high, {kpis['medium_risk_events']} medium</div>
        </div>
        <div class="stat-card accent-amber">
            <div class="stat-label">Avg / Min TTC</div>
            <div class="stat-value">{kpis['avg_ttc']:.2f}s</div>
            <div class="stat-sub">Min: {kpis['min_ttc']:.2f}s</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==================================================================
    # 2 & 3. Vehicle Class + Risk Distribution (side by side)
    # ==================================================================

    col_class, col_risk = st.columns(2)

    with col_class:
        st.markdown("""
        <div class="section-header">
            <h2>Vehicle Class Distribution</h2>
        </div>
        <div class="section-line"></div>
        """, unsafe_allow_html=True)

        if class_dist:
            labels = [c["class"] for c in class_dist]
            values = [c["count"] for c in class_dist]
            colors = [CLASS_COLORS.get(l, "#94a3b8") for l in labels]

            fig = go.Figure(data=[go.Pie(
                labels=labels, values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#0A0A0A", width=3)),
                textinfo="label+percent",
                textfont=dict(size=11, color="#FFFFFF", family="Inter"),
            )])
            fig.update_layout(**CHART_LAYOUT, height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Table below
            for c in class_dist:
                st.markdown(
                    f"<span style='color:{CLASS_COLORS.get(c['class'], '#94a3b8')}'>●</span> "
                    f"**{c['class']}**: {c['count']:,} ({c['percentage']}%)",
                    unsafe_allow_html=True
                )

    with col_risk:
        st.markdown("""
        <div class="section-header">
            <h2>Risk Severity Distribution</h2>
        </div>
        <div class="section-line"></div>
        """, unsafe_allow_html=True)

        risk_dist = engine.get_risk_distribution()
        if risk_dist:
            labels = [r["severity"].upper() for r in risk_dist]
            values = [r["count"] for r in risk_dist]
            colors = [COLORS.get(r["severity"], "#94a3b8") for r in risk_dist]

            fig = go.Figure(data=[go.Bar(
                x=labels, y=values,
                marker=dict(color=colors, cornerradius=0),
                text=values, textposition="outside",
                textfont=dict(color="#FFFFFF", size=14, family="Space Grotesk"),
            )])
            fig.update_layout(
                **CHART_LAYOUT, height=300,
                xaxis_title="", yaxis_title="Events",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            for r in risk_dist:
                sev_color = COLORS.get(r["severity"], "#94a3b8")
                st.markdown(
                    f"<span style='color:{sev_color}'>●</span> "
                    f"**{r['severity'].upper()}**: {r['count']} events "
                    f"({r['percentage']}%) — Avg TTC: {r['avg_ttc']:.2f}s",
                    unsafe_allow_html=True
                )

    # ==================================================================
    # 4. TTC Analytics
    # ==================================================================

    st.markdown("""
    <div class="section-header">
        <h2>Time-to-Collision Analytics</h2>
    </div>
    <div class="section-line"></div>
    """, unsafe_allow_html=True)

    ttc_stats = engine.get_ttc_stats()

    col_ttc_stats, col_ttc_hist = st.columns([1, 2])

    with col_ttc_stats:
        st.markdown(f"""
        <div style="background: #0A0A0A;
                    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 0px;
                    padding: 24px;">
            <div style="color: #777777; font-size: 0.75rem; text-transform: uppercase;
                        letter-spacing: 0.08em; margin-bottom: 16px;">TTC Statistics</div>
            <table style="width: 100%; color: #CCCCCC; font-size: 0.9rem;">
                <tr><td style="padding: 8px 0; color: #888888; border-bottom: 1px solid rgba(255,255,255,0.05);">Average</td>
                    <td style="text-align: right; font-family: JetBrains Mono; font-weight: 500; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        {ttc_stats['avg']:.3f}s</td></tr>
                <tr><td style="padding: 8px 0; color: #888888; border-bottom: 1px solid rgba(255,255,255,0.05);">Minimum</td>
                    <td style="text-align: right; font-family: JetBrains Mono; font-weight: 700; color: #FF2A55; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        {ttc_stats['min']:.3f}s</td></tr>
                <tr><td style="padding: 8px 0; color: #888888; border-bottom: 1px solid rgba(255,255,255,0.05);">Maximum</td>
                    <td style="text-align: right; font-family: JetBrains Mono; font-weight: 500; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        {ttc_stats['max']:.3f}s</td></tr>
                <tr><td style="padding: 8px 0; color: #888888; border-bottom: 1px solid rgba(255,255,255,0.05);">Median</td>
                    <td style="text-align: right; font-family: JetBrains Mono; font-weight: 500; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        {ttc_stats['median']:.3f}s</td></tr>
                <tr><td style="padding: 8px 0; color: #888888;">Samples</td>
                    <td style="text-align: right; font-family: JetBrains Mono;">
                        {ttc_stats['count']}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_ttc_hist:
        hist_data = engine.get_ttc_histogram_data(bins=12)
        if hist_data:
            bin_labels = [f"{h['bin_start']:.1f}" for h in hist_data]
            counts = [h["count"] for h in hist_data]
            bar_colors = [
                COLORS["high"] if h["bin_start"] < 1.0
                else COLORS["medium"] if h["bin_start"] < 2.0
                else COLORS["blue"]
                for h in hist_data
            ]

            fig = go.Figure(data=[go.Bar(
                x=bin_labels, y=counts,
                marker=dict(color=bar_colors, cornerradius=0),
                text=counts, textposition="outside",
                textfont=dict(color="#888888", size=10),
            )])
            fig.update_layout(
                **CHART_LAYOUT, height=280,
                title=dict(text="TTC Distribution", font=dict(size=13, color="#FFFFFF", family="Space Grotesk")),
                xaxis_title="TTC (seconds)", yaxis_title="Frequency",
            )
            st.plotly_chart(fig, use_container_width=True)

    # TTC Timeline scatter
    ttc_tl = engine.get_ttc_timeline()
    if ttc_tl:
        fig = go.Figure()
        for sev in ["high", "medium"]:
            pts = [p for p in ttc_tl if p["severity"] == sev]
            if pts:
                fig.add_trace(go.Scatter(
                    x=[p["time"] for p in pts],
                    y=[p["ttc"] for p in pts],
                    mode="markers",
                    name=sev.upper(),
                    marker=dict(
                        color=COLORS[sev], size=8,
                        line=dict(width=1, color="#050505"),
                    ),
                ))
        fig.update_layout(
            **CHART_LAYOUT, height=280,
            title=dict(text="TTC Over Video Time", font=dict(size=13, color="#FFFFFF", family="Space Grotesk")),
            xaxis_title="Video Time (seconds)", yaxis_title="TTC (seconds)",
            legend=dict(font=dict(color="#FFFFFF")),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ==================================================================
    # 5. Alert Timeline
    # ==================================================================

    st.markdown("""
    <div class="section-header">
        <h2>Alert Timeline</h2>
    </div>
    <div class="section-line"></div>
    """, unsafe_allow_html=True)

    alert_tl = engine.get_alert_timeline(severity_filter=severity_filter)

    if alert_tl:
        table_data = []
        for a in alert_tl:
            sev = a["severity"]
            sev_emoji = "🔴" if sev == "high" else "🟠"
            table_data.append({
                "Time": a["time_str"],
                "Vehicle A": f"ID {a['vehicle_a']} ({a['vehicle_a_class']})",
                "Vehicle B": f"ID {a['vehicle_b']} ({a['vehicle_b_class']})",
                "TTC (s)": f"{a['ttc']:.3f}",
                "Severity": f"{sev_emoji} {sev.upper()}",
            })
        st.dataframe(table_data, use_container_width=True, hide_index=True)
    else:
        st.info("No alerts match the current filter.")

    # ==================================================================
    # 6. Most Frequently Involved Vehicles
    # ==================================================================

    st.markdown("""
    <div class="section-header">
        <h2>Most Frequently Involved Vehicles</h2>
    </div>
    <div class="section-line"></div>
    """, unsafe_allow_html=True)

    top_vehicles = engine.get_most_involved_vehicles(top_n=10)
    if top_vehicles:
        veh_data = []
        for v in top_vehicles:
            veh_data.append({
                "Vehicle ID": v["vehicle_id"],
                "Risk Events": v["risk_events"],
                "Min TTC (s)": f"{v['min_ttc']:.3f}",
                "Avg TTC (s)": f"{v['avg_ttc']:.3f}",
            })
        st.dataframe(veh_data, use_container_width=True, hide_index=True)

    # ==================================================================
    # 7 & 8. Traffic Density + Density vs Risk
    # ==================================================================

    col_density, col_dvr = st.columns(2)

    with col_density:
        st.markdown("""
        <div class="section-header">
            <h2>Traffic Density</h2>
        </div>
        <div class="section-line"></div>
        """, unsafe_allow_html=True)

        density_tl = engine.get_density_timeline()
        if density_tl:
            fig = go.Figure(data=[go.Scatter(
                x=[d["time"] for d in density_tl],
                y=[d["vehicle_count"] for d in density_tl],
                mode="lines",
                fill="tozeroy",
                line=dict(color=COLORS["blue"], width=2),
                fillcolor="rgba(96, 165, 250, 0.1)",
            )])
            fig.update_layout(
                **CHART_LAYOUT, height=300,
                title=dict(text="Vehicles Over Time",
                           font=dict(size=13, color="#FFFFFF", family="Space Grotesk")),
                xaxis_title="Video Time (s)", yaxis_title="Vehicle Count",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_dvr:
        st.markdown("""
        <div class="section-header">
            <h2>Density vs Risk Events</h2>
        </div>
        <div class="section-line"></div>
        """, unsafe_allow_html=True)

        dvr = engine.get_density_vs_risk()
        if dvr:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[d["time"] for d in dvr],
                y=[d["risk_count"] for d in dvr],
                name="Risk Events",
                marker=dict(color=COLORS["high"], opacity=0.9, cornerradius=0),
                yaxis="y",
            ))
            fig.add_trace(go.Scatter(
                x=[d["time"] for d in dvr],
                y=[d["avg_density"] for d in dvr],
                name="Avg Density",
                mode="lines+markers",
                line=dict(color=COLORS["blue"], width=2),
                marker=dict(size=5, symbol="square"),
                yaxis="y2",
            ))
            fig.update_layout(
                **CHART_LAYOUT, height=300,
                title=dict(text="Traffic Density vs Observed Risk Events",
                           font=dict(size=13, color="#FFFFFF", family="Space Grotesk")),
                xaxis_title="Time Bucket (s)",
                yaxis=dict(title="Risk Events", side="left",
                           gridcolor="rgba(255,255,255,0.05)"),
                yaxis2=dict(title="Avg Vehicles", side="right",
                            overlaying="y", gridcolor="rgba(255,255,255,0.02)"),
                legend=dict(font=dict(color="#FFFFFF"),
                            x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ==================================================================
    # 10. Full Alert History Table (filtered)
    # ==================================================================

    st.markdown("""
    <div class="section-header">
        <h2>Alert History</h2>
        <span class="badge">SEARCHABLE</span>
    </div>
    <div class="section-line"></div>
    """, unsafe_allow_html=True)

    filtered = engine.get_filtered_alerts(
        severity=severity_filter,
        vehicle_class=class_filter,
        vehicle_id=vehicle_id,
    )

    if filtered:
        full_table = []
        for a in filtered:
            vehicles = a.get("vehicles_involved", [])
            ttc = a.get("time_to_collision", 0)
            full_table.append({
                "Timestamp": AnalyticsEngine._format_video_time(a.get("timestamp", 0)),
                "Vehicle A": vehicles[0] if len(vehicles) > 0 else "-",
                "Vehicle B": vehicles[1] if len(vehicles) > 1 else "-",
                "A Class": a.get("vehicle_a_class", "?"),
                "B Class": a.get("vehicle_b_class", "?"),
                "TTC (s)": f"{ttc:.3f}",
                "Severity": a.get("severity", "-").upper(),
                "Location": a.get("location", "-"),
            })
        st.dataframe(full_table, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(filtered)} of {len(engine.alerts)} total alerts")
    else:
        st.info("No alerts match the current filters.")

    # ==================================================================
    # 11. Export
    # ==================================================================

    st.markdown("""
    <div class="section-header">
        <h2>Export</h2>
    </div>
    <div class="section-line"></div>
    """, unsafe_allow_html=True)

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        if st.button("Export Alert Report (CSV)", use_container_width=True):
            path = engine.export_csv()
            st.success(f"Exported to {path}")

    with col_exp2:
        if st.button("Export Analytics Summary (CSV)", use_container_width=True):
            path = engine.export_analytics_csv()
            st.success(f"Exported to {path}")

    # ==================================================================
    # 12. Session Summary
    # ==================================================================

    summary = engine.get_session_summary()

    st.markdown(f"""
    <div class="section-header">
        <h2>Session Summary</h2>
    </div>
    <div class="section-line"></div>

    <div style="background: #0A0A0A;
                border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 0px;
                padding: 32px; margin-top: 8px;">
        <table style="width: 100%; color: #CCCCCC; font-size: 0.88rem;
                      border-collapse: collapse;">
            <tr><td style="padding: 10px 12px; color: #777777; width: 200px; border-bottom: 1px solid rgba(255,255,255,0.05);">Video</td>
                <td style="font-family: JetBrains Mono; font-weight: 500; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    {summary['video_file']}</td></tr>
            <tr><td style="padding: 10px 12px; color: #777777; border-bottom: 1px solid rgba(255,255,255,0.05);">Frames Processed</td>
                <td style="font-family: JetBrains Mono; font-weight: 500; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    {summary['total_frames']:,}</td></tr>
            <tr><td style="padding: 10px 12px; color: #777777; border-bottom: 1px solid rgba(255,255,255,0.05);">Processing FPS</td>
                <td style="font-family: JetBrains Mono; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    {summary['processing_fps']}</td></tr>
            <tr><td style="padding: 10px 12px; color: #777777; border-bottom: 1px solid rgba(255,255,255,0.05);">Unique Vehicles</td>
                <td style="font-family: JetBrains Mono; font-weight: 500; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    {summary['unique_vehicles']}</td></tr>
            <tr><td style="padding: 10px 12px; color: #777777; border-bottom: 1px solid rgba(255,255,255,0.05);">Total Risk Events</td>
                <td style="font-family: JetBrains Mono; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    {summary['total_risk_events']}</td></tr>
            <tr><td style="padding: 10px 12px; color: #777777; border-bottom: 1px solid rgba(255,255,255,0.05);">High Risk</td>
                <td style="font-family: JetBrains Mono; color: #FF2A55; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    {summary['high_risk']}</td></tr>
            <tr><td style="padding: 10px 12px; color: #777777; border-bottom: 1px solid rgba(255,255,255,0.05);">Medium Risk</td>
                <td style="font-family: JetBrains Mono; color: #FFD700; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    {summary['medium_risk']}</td></tr>
            <tr><td style="padding: 10px 12px; color: #777777; border-bottom: 1px solid rgba(255,255,255,0.05);">Total Alerts</td>
                <td style="font-family: JetBrains Mono; font-weight: 500; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    {summary['total_alerts']}</td></tr>
            <tr><td style="padding: 10px 12px; color: #777777; border-bottom: 1px solid rgba(255,255,255,0.05);">Minimum TTC</td>
                <td style="font-family: JetBrains Mono; color: #FF2A55; font-weight: 700; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    {summary['min_ttc']:.3f} sec</td></tr>
            <tr><td style="padding: 10px 12px; color: #777777; border-bottom: 1px solid rgba(255,255,255,0.05);">Average TTC</td>
                <td style="font-family: JetBrains Mono; font-weight: 500; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    {summary['avg_ttc']:.3f} sec</td></tr>
            <tr><td style="padding: 10px 12px; color: #777777;">Median TTC</td>
                <td style="font-family: JetBrains Mono;">
                    {summary['median_ttc']:.3f} sec</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
