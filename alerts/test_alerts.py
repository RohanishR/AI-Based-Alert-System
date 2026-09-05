"""
Test suite for Module 6 — Alert Generation.

Tests the AlertGenerator class with synthetic RiskEvent data,
verifying alert creation, deduplication/cooldown, logging, and
history management.
"""

import json
import os
import sys
import tempfile

import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alerts.alert_generator import AlertGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary config file pointing to a temp log file."""
    log_path = str(tmp_path / "test_alerts.jsonl").replace("\\", "/")
    config = {
        "cooldown_seconds": 5.0,
        "location": "Test_Intersection",
        "log_file": log_path,
        "max_history": 50,
    }
    config_path = str(tmp_path / "test_config.yaml")
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path, log_path


@pytest.fixture
def generator(temp_config):
    """Create an AlertGenerator with temp config."""
    config_path, _ = temp_config
    return AlertGenerator(config_path=config_path)


def make_risk_event(id_a, id_b, ttc, severity="medium"):
    """Build a RiskEvent dict."""
    return {
        "vehicle_pair": (min(id_a, id_b), max(id_a, id_b)),
        "ttc": ttc,
        "severity": severity,
    }


# ===========================================================================
# Test 1: Basic alert generation
# ===========================================================================

class TestAlertGeneration:
    """Verify that risk events are converted to properly structured alerts."""

    def test_single_event_produces_alert(self, generator):
        events = [make_risk_event(7, 12, 1.4, "high")]
        alerts = generator.process(events, timestamp=1000.0)

        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["alert_type"] == "forward_collision_warning"
        assert alert["timestamp"] == 1000.0
        assert alert["vehicles_involved"] == [7, 12]
        assert alert["time_to_collision"] == 1.4
        assert alert["severity"] == "high"
        assert alert["location"] == "Test_Intersection"

    def test_multiple_events_produce_multiple_alerts(self, generator):
        events = [
            make_risk_event(1, 2, 0.8, "high"),
            make_risk_event(3, 4, 1.5, "medium"),
        ]
        alerts = generator.process(events, timestamp=1000.0)
        assert len(alerts) == 2

    def test_vehicle_pair_always_sorted(self, generator):
        # Pass (12, 7) — should still be stored as [7, 12]
        events = [{"vehicle_pair": (12, 7), "ttc": 1.0, "severity": "high"}]
        alerts = generator.process(events, timestamp=1000.0)

        assert alerts[0]["vehicles_involved"] == [7, 12]


# ===========================================================================
# Test 2: Deduplication / cooldown
# ===========================================================================

class TestDeduplication:
    """Verify that alerts for the same pair are suppressed during cooldown."""

    def test_same_pair_suppressed_within_cooldown(self, generator):
        events = [make_risk_event(1, 2, 1.0, "high")]

        # First call — should generate
        alerts1 = generator.process(events, timestamp=100.0)
        assert len(alerts1) == 1

        # Second call 2s later — within 5s cooldown, should suppress
        alerts2 = generator.process(events, timestamp=102.0)
        assert len(alerts2) == 0

    def test_same_pair_allowed_after_cooldown(self, generator):
        events = [make_risk_event(1, 2, 1.0, "high")]

        alerts1 = generator.process(events, timestamp=100.0)
        assert len(alerts1) == 1

        # 6s later — past 5s cooldown, should generate again
        alerts2 = generator.process(events, timestamp=106.0)
        assert len(alerts2) == 1

    def test_different_pairs_not_affected(self, generator):
        event_a = [make_risk_event(1, 2, 1.0, "high")]
        event_b = [make_risk_event(3, 4, 1.5, "medium")]

        alerts1 = generator.process(event_a, timestamp=100.0)
        assert len(alerts1) == 1

        # Different pair — should not be suppressed
        alerts2 = generator.process(event_b, timestamp=101.0)
        assert len(alerts2) == 1

    def test_cooldown_boundary(self, generator):
        events = [make_risk_event(5, 6, 0.9, "high")]

        generator.process(events, timestamp=100.0)

        # Exactly at 5.0s — cooldown just expires (cleanup uses >=)
        alerts = generator.process(events, timestamp=105.0)
        assert len(alerts) == 1


# ===========================================================================
# Test 3: JSONL logging
# ===========================================================================

class TestLogging:
    """Verify that alerts are correctly persisted to the JSONL log."""

    def test_alert_written_to_log(self, generator, temp_config):
        _, log_path = temp_config
        events = [make_risk_event(7, 12, 1.4, "high")]
        generator.process(events, timestamp=1000.0)

        assert os.path.exists(log_path)
        with open(log_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1

        logged = json.loads(lines[0])
        assert logged["vehicles_involved"] == [7, 12]
        assert logged["time_to_collision"] == 1.4

    def test_multiple_alerts_appended(self, generator, temp_config):
        _, log_path = temp_config

        generator.process([make_risk_event(1, 2, 0.5, "high")], timestamp=100.0)
        generator.process([make_risk_event(3, 4, 1.8, "medium")], timestamp=101.0)

        with open(log_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_clear_log(self, generator, temp_config):
        _, log_path = temp_config

        generator.process([make_risk_event(1, 2, 0.5, "high")], timestamp=100.0)
        generator.clear_log()

        with open(log_path, "r") as f:
            content = f.read()
        assert content == ""


# ===========================================================================
# Test 4: Alert history buffer
# ===========================================================================

class TestAlertHistory:
    """Verify the in-memory alert history buffer."""

    def test_history_accumulates(self, generator):
        for i in range(5):
            generator.process(
                [make_risk_event(i, i + 100, 1.0, "medium")],
                timestamp=100.0 + i * 10,  # spread past cooldown
            )

        history = generator.get_alert_history()
        assert len(history) == 5

    def test_history_capped_at_max(self, generator):
        # max_history is 50 in our temp config
        for i in range(60):
            generator.process(
                [make_risk_event(i, i + 1000, 0.5, "high")],
                timestamp=i * 10.0,  # each far enough apart for cooldown
            )

        history = generator.get_alert_history()
        assert len(history) == 50

    def test_stats_tracking(self, generator):
        generator.process([make_risk_event(1, 2, 0.5, "high")], timestamp=100.0)
        generator.process([make_risk_event(3, 4, 1.5, "medium")], timestamp=101.0)

        stats = generator.get_stats()
        assert stats["total_alerts"] == 2


# ===========================================================================
# Test 5: Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_events(self, generator):
        alerts = generator.process([], timestamp=100.0)
        assert len(alerts) == 0

    def test_active_alerts_query(self, generator):
        generator.process([make_risk_event(1, 2, 1.0, "high")], timestamp=100.0)

        active = generator.get_active_alerts(current_time=102.0)
        assert (1, 2) in active

        # After cooldown
        active_later = generator.get_active_alerts(current_time=106.0)
        assert (1, 2) not in active_later

    def test_no_default_timestamp_crash(self, generator):
        # Calling without explicit timestamp should use time.time()
        events = [make_risk_event(1, 2, 1.0, "high")]
        alerts = generator.process(events)
        assert len(alerts) == 1
        assert alerts[0]["timestamp"] > 0


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
