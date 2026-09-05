"""
Test suite for Module 4 — Trajectory Prediction.

All tests use simulated TrackedObject data so the module can be verified
independently from Module 3 (Detection + Tracking).
"""

import sys
import os
import math

import numpy as np
import pytest

# Ensure project root is on the path so `prediction.*` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prediction.kalman_filter import TrajectoryPredictor
from prediction.calibration import CoordinateTransformer


# ---------------------------------------------------------------------------
# Helper: build a TrackedObject dict matching Module 3's contract
# ---------------------------------------------------------------------------

def make_tracked_object(track_id, history, cls="car"):
    """
    Build a TrackedObject dict from a list of (x, y, t) tuples.
    Generates a dummy bbox centred on the last position.
    """
    x, y, _ = history[-1]
    half_w, half_h = 30, 20
    return {
        "track_id": track_id,
        "class": cls,
        "bbox": (x - half_w, y - half_h, x + half_w, y + half_h),
        "history": history,
    }


# ===========================================================================
# Test 1: Straight-line constant-velocity motion
# ===========================================================================

class TestStraightLineMotion:
    """A vehicle moving at constant velocity should have predictions that
    extrapolate linearly along its direction of travel."""

    def setup_method(self):
        self.predictor = TrajectoryPredictor(
            config_path="prediction/config.yaml"
        )

    def test_predictions_follow_velocity_direction(self):
        # Simulate a car moving right at 100 px/s (constant velocity)
        fps = 10
        dt = 1.0 / fps
        speed_x = 100.0  # px/s
        speed_y = 0.0

        # Build 20 frames of history (2 seconds at 10 fps)
        history = []
        for i in range(20):
            t = i * dt
            x = 200.0 + speed_x * t
            y = 300.0 + speed_y * t
            history.append((x, y, t))

        # Feed frames one at a time to warm up the filter
        for frame_idx in range(1, len(history)):
            frame_history = history[: frame_idx + 1]
            obj = make_tracked_object(1, frame_history)
            current_t = history[frame_idx][2]
            results = self.predictor.update([obj], current_t)

        # Check the final prediction
        assert len(results) == 1
        pred = results[0]
        assert pred["track_id"] == 1
        assert len(pred["predictions"]) == 3  # t+1s, t+2s, t+3s

        # Velocity should be close to (100, 0)
        vx, vy = pred["velocity"]
        assert abs(vx - speed_x) < 15, f"vx={vx}, expected ~{speed_x}"
        assert abs(vy - speed_y) < 15, f"vy={vy}, expected ~{speed_y}"

        # Predicted x positions should increase roughly by 100 px/s
        last_x = history[-1][0]
        for px, py, pt in pred["predictions"]:
            assert px > last_x, "Predictions should be ahead of current pos"

    def test_prediction_timestamps_match_horizons(self):
        history = [(100, 200, 0.0), (110, 200, 0.1)]
        obj = make_tracked_object(42, history)
        results = self.predictor.update([obj], 0.1)

        pred = results[0]
        expected_times = [0.1 + 1.0, 0.1 + 2.0, 0.1 + 3.0]
        for (_, _, pt), et in zip(pred["predictions"], expected_times):
            assert abs(pt - et) < 1e-6, f"Timestamp {pt} != expected {et}"


# ===========================================================================
# Test 2: Newly appearing vehicle with minimal history
# ===========================================================================

class TestNewVehicle:
    """A vehicle with only 1 history point should not crash and should
    return predictions (with near-zero velocity)."""

    def setup_method(self):
        self.predictor = TrajectoryPredictor(
            config_path="prediction/config.yaml"
        )

    def test_single_history_point(self):
        history = [(500.0, 400.0, 0.0)]
        obj = make_tracked_object(99, history)
        results = self.predictor.update([obj], 0.0)

        assert len(results) == 1
        pred = results[0]
        vx, vy = pred["velocity"]

        # Velocity should be ~0 since we can't estimate it from 1 point
        assert abs(vx) < 1e-3
        assert abs(vy) < 1e-3

        # Predictions should be near the current position
        for px, py, _ in pred["predictions"]:
            assert abs(px - 500.0) < 5.0
            assert abs(py - 400.0) < 5.0

    def test_two_history_points_bootstrap_velocity(self):
        # 2 points at 0.1s apart, moving at 50 px/s to the right
        history = [(100.0, 200.0, 0.0), (105.0, 200.0, 0.1)]
        obj = make_tracked_object(100, history)
        results = self.predictor.update([obj], 0.1)

        pred = results[0]
        vx, vy = pred["velocity"]
        # Should have bootstrapped vx ≈ 50 px/s
        assert abs(vx - 50.0) < 10.0


# ===========================================================================
# Test 3: Track disappearance and cleanup
# ===========================================================================

class TestTrackCleanup:
    """When a track disappears from the active list, its Kalman filter
    should be cleaned up after the stale timeout."""

    def setup_method(self):
        self.predictor = TrajectoryPredictor(
            config_path="prediction/config.yaml"
        )

    def test_filter_removed_after_timeout(self):
        # Create and update a track
        history = [(100, 200, 0.0), (110, 200, 0.1)]
        obj = make_tracked_object(5, history)
        self.predictor.update([obj], 0.1)
        assert self.predictor.get_filter_count() == 1

        # Next frame: track 5 disappears, a new track 6 appears
        history2 = [(300, 400, 0.2)]
        obj2 = make_tracked_object(6, history2)
        self.predictor.update([obj2], 0.2)
        # Track 5 is still within stale_timeout (3s), so filter count = 2
        assert self.predictor.get_filter_count() == 2

        # Jump forward past the stale timeout
        history3 = [(310, 400, 4.0)]
        obj3 = make_tracked_object(6, history3)
        self.predictor.update([obj3], 4.0)
        # Track 5 should now be cleaned up
        assert self.predictor.get_filter_count() == 1


# ===========================================================================
# Test 4: Multiple vehicles with independent predictions
# ===========================================================================

class TestMultipleVehicles:
    """Two vehicles approaching each other should produce independent,
    correct predictions for each."""

    def setup_method(self):
        self.predictor = TrajectoryPredictor(
            config_path="prediction/config.yaml"
        )

    def test_two_vehicles_opposite_directions(self):
        fps = 10
        dt = 1.0 / fps

        for frame in range(15):
            t = frame * dt
            # Vehicle A: moving right
            xa = 100.0 + 80.0 * t
            ya = 300.0
            # Vehicle B: moving left
            xb = 600.0 - 60.0 * t
            yb = 300.0

            obj_a = make_tracked_object(
                1, [(100.0 + 80.0 * i * dt, 300.0, i * dt) for i in range(frame + 1)]
            )
            obj_b = make_tracked_object(
                2, [(600.0 - 60.0 * i * dt, 300.0, i * dt) for i in range(frame + 1)]
            )
            results = self.predictor.update([obj_a, obj_b], t)

        assert len(results) == 2

        pred_a = next(r for r in results if r["track_id"] == 1)
        pred_b = next(r for r in results if r["track_id"] == 2)

        # Vehicle A should have positive vx, Vehicle B negative vx
        assert pred_a["velocity"][0] > 0, "Vehicle A should move right"
        assert pred_b["velocity"][0] < 0, "Vehicle B should move left"

        # Their predicted x positions should diverge
        assert pred_a["predictions"][0][0] > 100.0
        assert pred_b["predictions"][0][0] < 600.0


# ===========================================================================
# Test 5: Velocity change — Kalman filter adaptation
# ===========================================================================

class TestVelocityChange:
    """When a vehicle changes velocity, the Kalman filter should adapt
    its state after a few frames."""

    def setup_method(self):
        self.predictor = TrajectoryPredictor(
            config_path="prediction/config.yaml"
        )

    def test_filter_adapts_to_new_velocity(self):
        fps = 10
        dt = 1.0 / fps

        # Phase 1: moving right at 100 px/s for 1 second
        for frame in range(10):
            t = frame * dt
            history = [
                (200.0 + 100.0 * i * dt, 300.0, i * dt)
                for i in range(frame + 1)
            ]
            obj = make_tracked_object(1, history)
            self.predictor.update([obj], t)

        # Phase 2: slow down to 30 px/s for 1 second
        base_x = 200.0 + 100.0 * 1.0  # position at end of phase 1
        for frame in range(10, 20):
            t = frame * dt
            phase2_t = (frame - 10) * dt
            # Build full history (phase 1 + phase 2 so far)
            history = [
                (200.0 + 100.0 * i * dt, 300.0, i * dt)
                for i in range(10)
            ]
            history += [
                (base_x + 30.0 * (i - 10) * dt, 300.0, i * dt)
                for i in range(10, frame + 1)
            ]
            obj = make_tracked_object(1, history)
            results = self.predictor.update([obj], t)

        # After adapting, velocity should be closer to 30 than to 100
        pred = results[0]
        vx = pred["velocity"][0]
        assert vx < 70, f"Filter should adapt: vx={vx}, expected closer to 30"
        assert vx > 10, f"vx too low: {vx}"


# ===========================================================================
# Test 6: Calibration round-trip (pixel → world → pixel)
# ===========================================================================

class TestCalibrationRoundTrip:
    """The CoordinateTransformer should round-trip: converting pixel→world
    and back should return approximately the original pixel coordinates."""

    def test_identity_like_transform(self):
        # Use a simple known transform: scale by 0.1 (10 px = 1 meter)
        src = np.array([
            [0, 0], [1000, 0], [1000, 1000], [0, 1000],
            [500, 500], [250, 750],
        ], dtype=np.float64)
        dst = src * 0.1  # Simple scaling

        transformer = CoordinateTransformer(src, dst)
        assert transformer.is_calibrated()

        # Test several points
        test_points = [(100, 200), (500, 500), (750, 250), (333, 666)]
        for px, py in test_points:
            wx, wy = transformer.pixel_to_world(px, py)
            # World should be ~0.1× pixel
            assert abs(wx - px * 0.1) < 1.0, f"pixel_to_world({px},{py}) → ({wx},{wy})"
            assert abs(wy - py * 0.1) < 1.0

            # Round-trip
            rx, ry = transformer.world_to_pixel(wx, wy)
            assert abs(rx - px) < 2.0, f"Round-trip error: ({px},{py}) → ({rx},{ry})"
            assert abs(ry - py) < 2.0

    def test_save_and_load(self, tmp_path):
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
        dst = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)

        t1 = CoordinateTransformer(src, dst)
        save_path = str(tmp_path / "homography.npz")
        t1.save(save_path)

        t2 = CoordinateTransformer.load(save_path)
        assert t2.is_calibrated()

        # Both should produce the same result
        w1 = t1.pixel_to_world(50, 50)
        w2 = t2.pixel_to_world(50, 50)
        assert abs(w1[0] - w2[0]) < 1e-6
        assert abs(w1[1] - w2[1]) < 1e-6

    def test_insufficient_points_raises(self):
        with pytest.raises(ValueError):
            CoordinateTransformer(
                src_points=[[0, 0], [1, 1]],
                dst_points=[[0, 0], [1, 1]],
            )


# ===========================================================================
# Entry point for direct execution
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
