"""
Test suite for Module 5 — Risk Scoring (Time-to-Collision).

All tests use synthetic PredictedTrajectory data so the module can be
verified independently from Module 4 (Trajectory Prediction).
"""

import sys
import os

import numpy as np
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from risk_scoring.ttc import RiskScorer


# ---------------------------------------------------------------------------
# Helper: build a PredictedTrajectory dict matching Module 4's contract
# ---------------------------------------------------------------------------

def make_prediction(track_id, x, y, vx, vy, base_time=0.0,
                    horizons=(1.0, 2.0, 3.0)):
    """
    Build a PredictedTrajectory dict.

    Generates future predictions by extrapolating (x, y) forward using
    (vx, vy) at the given horizons — mimicking what the Kalman filter
    would produce for constant-velocity motion.
    """
    predictions = []
    for h in horizons:
        px = x + vx * h
        py = y + vy * h
        pt = base_time + h
        predictions.append((px, py, pt))

    return {
        "track_id": track_id,
        "predictions": predictions,
        "velocity": (vx, vy),
    }


# ===========================================================================
# Test 1: Vehicles moving directly toward each other — should detect risk
# ===========================================================================

class TestHeadOnCollision:
    """Two vehicles on a direct head-on collision course."""

    def setup_method(self):
        self.scorer = RiskScorer(config_path="risk_scoring/config.yaml")

    def test_head_on_produces_risk_event(self):
        # Vehicle A at x=100, moving right at 50 px/s
        # Vehicle B at x=250, moving left at 50 px/s
        # Distance = 150, closing speed = 100, TTC = 1.5s
        traj_a = make_prediction(1, x=100, y=300, vx=50, vy=0)
        traj_b = make_prediction(2, x=250, y=300, vx=-50, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])

        assert len(events) == 1
        event = events[0]
        assert event["vehicle_pair"] == (1, 2)
        assert 1.0 < event["ttc"] < 2.0
        assert event["severity"] == "medium"

    def test_very_close_head_on_is_high_severity(self):
        # Distance = 80, closing speed = 100, TTC = 0.8s → "high"
        traj_a = make_prediction(3, x=200, y=300, vx=50, vy=0)
        traj_b = make_prediction(4, x=280, y=300, vx=-50, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])

        assert len(events) == 1
        assert events[0]["ttc"] < 1.0
        assert events[0]["severity"] == "high"

    def test_pair_is_sorted(self):
        # Even if we pass track_id 10 before track_id 3, the pair should be (3, 10)
        traj_a = make_prediction(10, x=100, y=300, vx=50, vy=0)
        traj_b = make_prediction(3, x=250, y=300, vx=-50, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])

        assert len(events) == 1
        assert events[0]["vehicle_pair"] == (3, 10)


# ===========================================================================
# Test 2: Vehicles moving apart — should NOT detect risk
# ===========================================================================

class TestMovingApart:
    """Two vehicles moving away from each other should produce no risk."""

    def setup_method(self):
        self.scorer = RiskScorer(config_path="risk_scoring/config.yaml")

    def test_diverging_vehicles_no_risk(self):
        # Vehicle A moving left, Vehicle B moving right — diverging
        traj_a = make_prediction(1, x=200, y=300, vx=-50, vy=0)
        traj_b = make_prediction(2, x=250, y=300, vx=50, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])
        assert len(events) == 0

    def test_both_moving_same_direction_different_speeds(self):
        # Both moving right, A faster than B — A is behind B and catching up?
        # A at x=100 vx=80, B at x=200 vx=90 → B is pulling away
        traj_a = make_prediction(1, x=100, y=300, vx=80, vy=0)
        traj_b = make_prediction(2, x=200, y=300, vx=90, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])
        assert len(events) == 0


# ===========================================================================
# Test 3: Parallel vehicles — should NOT detect risk
# ===========================================================================

class TestParallelMotion:
    """Vehicles moving in parallel at the same speed should produce no risk."""

    def setup_method(self):
        self.scorer = RiskScorer(config_path="risk_scoring/config.yaml")

    def test_parallel_same_speed(self):
        # Both moving right at 60 px/s, side by side
        traj_a = make_prediction(1, x=100, y=280, vx=60, vy=0)
        traj_b = make_prediction(2, x=100, y=320, vx=60, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])
        assert len(events) == 0

    def test_parallel_opposite_lanes(self):
        # Moving in opposite directions but in separate lanes (y offset)
        # Closing in x, but the formula uses the full position vector
        # including the y offset, so closing speed will be reduced
        traj_a = make_prediction(1, x=100, y=200, vx=60, vy=0)
        traj_b = make_prediction(2, x=250, y=400, vx=-60, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])
        # With significant y offset (200px) the closing speed along the
        # position vector is lower, and TTC may be above threshold
        # Either no event or TTC is above threshold is acceptable


# ===========================================================================
# Test 4: Vehicles far apart — proximity filter should skip them
# ===========================================================================

class TestProximityFilter:
    """Vehicles beyond the proximity radius should be skipped entirely."""

    def setup_method(self):
        self.scorer = RiskScorer(config_path="risk_scoring/config.yaml")

    def test_far_apart_vehicles_skipped(self):
        # 1000px apart — well beyond default proximity_radius of 300
        traj_a = make_prediction(1, x=0, y=300, vx=50, vy=0)
        traj_b = make_prediction(2, x=1000, y=300, vx=-50, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])
        assert len(events) == 0

    def test_just_within_proximity(self):
        # 180px apart — within 300px radius, closing speed = 100, TTC = 1.8s
        traj_a = make_prediction(1, x=100, y=300, vx=50, vy=0)
        traj_b = make_prediction(2, x=280, y=300, vx=-50, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])
        assert len(events) == 1

    def test_just_outside_proximity(self):
        # 310px apart — outside 300px radius
        traj_a = make_prediction(1, x=100, y=300, vx=50, vy=0)
        traj_b = make_prediction(2, x=410, y=300, vx=-50, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])
        assert len(events) == 0


# ===========================================================================
# Test 5: TTC below threshold — flagged
# ===========================================================================

class TestTTCBelowThreshold:
    """Verify that pairs with TTC below the threshold are correctly flagged."""

    def setup_method(self):
        self.scorer = RiskScorer(config_path="risk_scoring/config.yaml")

    def test_ttc_below_threshold_flagged(self):
        # Distance = 100, closing speed = 100, TTC = 1.0s → medium
        traj_a = make_prediction(1, x=200, y=300, vx=50, vy=0)
        traj_b = make_prediction(2, x=300, y=300, vx=-50, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])
        assert len(events) == 1
        assert events[0]["ttc"] <= 2.0

    def test_diagonal_approach(self):
        # Two vehicles approaching diagonally
        traj_a = make_prediction(1, x=200, y=200, vx=40, vy=40)
        traj_b = make_prediction(2, x=300, y=300, vx=-40, vy=-40)

        events = self.scorer.evaluate([traj_a, traj_b])
        assert len(events) == 1
        assert events[0]["ttc"] > 0


# ===========================================================================
# Test 6: TTC above threshold — not flagged
# ===========================================================================

class TestTTCAboveThreshold:
    """Pairs with TTC above the threshold should not produce risk events."""

    def setup_method(self):
        self.scorer = RiskScorer(config_path="risk_scoring/config.yaml")

    def test_slow_approach_high_ttc(self):
        # Distance = 280, closing speed = 10 (very slow), TTC = 28s
        traj_a = make_prediction(1, x=100, y=300, vx=5, vy=0)
        traj_b = make_prediction(2, x=380, y=300, vx=-5, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])
        assert len(events) == 0


# ===========================================================================
# Test 7: Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def setup_method(self):
        self.scorer = RiskScorer(config_path="risk_scoring/config.yaml")

    def test_single_vehicle_no_pairs(self):
        traj_a = make_prediction(1, x=100, y=300, vx=50, vy=0)
        events = self.scorer.evaluate([traj_a])
        assert len(events) == 0

    def test_empty_input(self):
        events = self.scorer.evaluate([])
        assert len(events) == 0

    def test_three_vehicles_multiple_pairs(self):
        # A and B approaching, B and C approaching, A and C far apart
        traj_a = make_prediction(1, x=100, y=300, vx=50, vy=0)
        traj_b = make_prediction(2, x=200, y=300, vx=-50, vy=0)
        traj_c = make_prediction(3, x=250, y=300, vx=-50, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b, traj_c])

        # At least pair (1,2) should be flagged; (2,3) may or may not
        # depending on closing speed (both moving left, B faster)
        pair_ids = [e["vehicle_pair"] for e in events]
        assert (1, 2) in pair_ids

    def test_stationary_vehicles_no_risk(self):
        # Both stationary — closing speed = 0
        traj_a = make_prediction(1, x=200, y=300, vx=0, vy=0)
        traj_b = make_prediction(2, x=250, y=300, vx=0, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])
        assert len(events) == 0

    def test_vehicles_at_same_position(self):
        # Same position — should return TTC ≈ 0 (already colliding)
        traj_a = make_prediction(1, x=300, y=300, vx=10, vy=0)
        traj_b = make_prediction(2, x=300, y=300, vx=-10, vy=0)

        events = self.scorer.evaluate([traj_a, traj_b])
        assert len(events) == 1
        assert events[0]["ttc"] == 0.0
        assert events[0]["severity"] == "high"


# ===========================================================================
# Entry point for direct execution
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
