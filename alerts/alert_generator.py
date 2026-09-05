"""
Alert Generator Module — simulated V2I safety alert creation and logging.

Converts RiskEvent objects from Module 5 into structured V2I-style alert
messages, applies cooldown-based deduplication to prevent flooding, and
persists alerts to a JSONL log file.

This is a SIMULATION of V2I alerting — no real broadcast to physical
vehicles occurs. The alerts are displayed on a Streamlit dashboard and
logged for later precision/recall evaluation.

Interface contracts (from architecture.md §7):

    Input — RiskEvent:
        {
            "vehicle_pair": (int, int),
            "ttc": float,
            "severity": "high" | "medium"
        }

    Output — Alert:
        {
            "alert_type": "forward_collision_warning",
            "timestamp": float,
            "vehicles_involved": [int, int],
            "time_to_collision": float,
            "severity": "high" | "medium",
            "location": str
        }
"""

import json
import os
import time
from collections import deque

import yaml


class AlertGenerator:
    """
    Generates deduplicated V2I-style collision warning alerts from
    risk events and logs them to disk.
    """

    def __init__(self, config_path="alerts/config.yaml"):
        """
        Initialize the alert generator.

        :param config_path: Path to the alert config YAML file.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.cooldown_seconds = self.config.get("cooldown_seconds", 5.0)
        self.location = self.config.get("location", "Intersection_A")
        self.log_file = self.config.get("log_file", "alerts/alert_log.jsonl")
        self.max_history = self.config.get("max_history", 200)

        # Deduplication: (track_id_a, track_id_b) → last alert timestamp
        self._active_alerts = {}

        # In-memory alert history for dashboard display
        self._alert_history = deque(maxlen=self.max_history)

        # Counters for dashboard metrics
        self.total_alerts_generated = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, risk_events, timestamp=None):
        """
        Process a batch of risk events from Module 5 and generate
        deduplicated alerts.

        :param risk_events: List of RiskEvent dicts from RiskScorer.
        :param timestamp: Current timestamp (seconds). If None, uses
                          time.time().
        :return: List of newly generated alert dicts (may be empty if
                 all events were deduplicated).
        """
        if timestamp is None:
            timestamp = time.time()

        new_alerts = []

        for event in risk_events:
            pair = tuple(sorted(event["vehicle_pair"]))

            # --- Deduplication check ---
            if self._is_in_cooldown(pair, timestamp):
                continue

            # --- Generate alert ---
            alert = self._create_alert(event, timestamp)

            # --- Record ---
            self._active_alerts[pair] = timestamp
            self._alert_history.append(alert)
            self.total_alerts_generated += 1
            new_alerts.append(alert)

            # --- Log to disk ---
            self._log_alert(alert)

        # --- Cleanup expired cooldowns ---
        self._cleanup_cooldowns(timestamp)

        return new_alerts

    def get_active_alerts(self, current_time=None):
        """
        Return currently active (non-expired) alert pairs and their
        last alert time.

        :param current_time: Current timestamp. If None, uses time.time().
        :return: Dict of {(id_a, id_b): last_alert_timestamp}.
        """
        if current_time is None:
            current_time = time.time()

        return {
            pair: t for pair, t in self._active_alerts.items()
            if (current_time - t) <= self.cooldown_seconds
        }

    def get_alert_history(self):
        """
        Return the in-memory alert history (most recent last).

        :return: List of alert dicts.
        """
        return list(self._alert_history)

    def get_stats(self):
        """
        Return summary statistics for the dashboard.

        :return: Dict with counts.
        """
        recent = list(self._alert_history)
        high_count = sum(1 for a in recent[-20:] if a["severity"] == "high")
        medium_count = sum(1 for a in recent[-20:] if a["severity"] == "medium")

        return {
            "total_alerts": self.total_alerts_generated,
            "active_cooldowns": len(self._active_alerts),
            "recent_high": high_count,
            "recent_medium": medium_count,
        }

    def clear_log(self):
        """Clear the JSONL log file on disk."""
        with open(self.log_file, "w") as f:
            pass  # Truncate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_alert(self, risk_event, timestamp):
        """
        Build a structured V2I alert dict from a risk event.

        :param risk_event: RiskEvent dict.
        :param timestamp: Alert timestamp.
        :return: Alert dict.
        """
        pair = tuple(sorted(risk_event["vehicle_pair"]))
        return {
            "alert_type": "forward_collision_warning",
            "timestamp": round(timestamp, 3),
            "vehicles_involved": list(pair),
            "time_to_collision": risk_event["ttc"],
            "severity": risk_event["severity"],
            "location": self.location,
        }

    def _is_in_cooldown(self, pair, current_time):
        """
        Check if an alert for this vehicle pair is still within the
        cooldown window.

        :param pair: Sorted tuple (track_id_a, track_id_b).
        :param current_time: Current timestamp.
        :return: True if the pair should be suppressed.
        """
        if pair not in self._active_alerts:
            return False

        last_time = self._active_alerts[pair]
        return (current_time - last_time) < self.cooldown_seconds

    def _cleanup_cooldowns(self, current_time):
        """Remove expired entries from the active_alerts dict."""
        expired = [
            pair for pair, t in self._active_alerts.items()
            if (current_time - t) >= self.cooldown_seconds
        ]
        for pair in expired:
            del self._active_alerts[pair]

    def _log_alert(self, alert):
        """
        Append an alert as a JSON line to the log file.

        :param alert: Alert dict to log.
        """
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(alert) + "\n")
