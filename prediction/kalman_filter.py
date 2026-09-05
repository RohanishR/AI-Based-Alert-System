"""
Trajectory Prediction Module — Kalman-filter-based future position estimation.

Uses a per-vehicle Kalman filter with state [x, y, vx, vy] and a constant-
velocity motion model to predict each tracked vehicle's position 1–3 seconds
into the future.

This module is intentionally independent of Module 3's internals. It consumes
the TrackedObject interface contract and produces PredictedTrajectory dicts
that Module 5 (Risk Scoring) will consume.

Interface contracts (from architecture.md §7):

    Input — TrackedObject:
        {
            "track_id": int,
            "class": str,
            "bbox": (x1, y1, x2, y2),
            "history": [(x_center, y_center, timestamp), ...]
        }

    Output — PredictedTrajectory:
        {
            "track_id": int,
            "predictions": [(x, y, future_timestamp), ...],
            "velocity": (vx, vy)
        }
"""

import copy
import os

import numpy as np
import yaml
from filterpy.kalman import KalmanFilter

from prediction.calibration import CoordinateTransformer


class TrajectoryPredictor:
    """
    Maintains a separate Kalman filter per tracked vehicle and generates
    short-horizon future position predictions.
    """

    def __init__(self, config_path="prediction/config.yaml", transformer=None):
        """
        Initialize the trajectory predictor.

        :param config_path: Path to the prediction config YAML file.
        :param transformer: Optional CoordinateTransformer for pixel-to-world
                            conversion. If None, operates in raw pixel space.
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.measurement_noise = self.config.get("measurement_noise", 4.0)
        self.process_noise = self.config.get("process_noise", 0.5)
        self.initial_covariance = self.config.get("initial_covariance", 100.0)
        self.prediction_horizons = self.config.get(
            "prediction_horizons", [1.0, 2.0, 3.0]
        )
        self.min_history = self.config.get("min_history_for_velocity", 2)
        self.stale_timeout = self.config.get("stale_timeout", 3.0)

        # Optional coordinate transformer (pixel → real-world)
        self.transformer = transformer

        # Load calibration file if specified and no transformer was passed in
        if self.transformer is None:
            cal_path = self.config.get("calibration_file")
            if cal_path and os.path.exists(cal_path):
                self.transformer = CoordinateTransformer.load(cal_path)

        # Per-track Kalman filters: track_id → KalmanFilter
        self._filters = {}

        # Timestamp of last update per track, for stale cleanup
        self._last_seen = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, tracked_objects, timestamp):
        """
        Process a frame's worth of tracked objects and produce predictions.

        :param tracked_objects: List of TrackedObject dicts from Module 3.
        :param timestamp: Current frame timestamp (seconds).
        :return: List of PredictedTrajectory dicts.
        """
        predictions = []
        active_ids = set()

        for obj in tracked_objects:
            track_id = obj["track_id"]
            history = obj["history"]
            active_ids.add(track_id)
            self._last_seen[track_id] = timestamp

            # Get the most recent position
            x, y, t = history[-1]

            # Transform to world coordinates if calibration is available
            if self.transformer and self.transformer.is_calibrated():
                x, y = self.transformer.pixel_to_world(x, y)

            if track_id not in self._filters:
                # ---- New vehicle: initialise a Kalman filter ----
                vx, vy = self._bootstrap_velocity(history)
                self._filters[track_id] = self._create_filter(x, y, vx, vy)
            else:
                # ---- Existing vehicle: predict → update cycle ----
                kf = self._filters[track_id]

                # Compute dt since the filter was last updated
                dt = self._compute_dt(history)
                if dt > 0:
                    kf.F = self._transition_matrix(dt)
                    kf.Q = self._process_noise_matrix(dt)

                kf.predict()
                kf.update(np.array([x, y]))

            # Generate future predictions
            pred = self._predict_future(track_id, timestamp)
            predictions.append(pred)

        # Cleanup stale filters
        self.cleanup(active_ids, timestamp)

        return predictions

    def cleanup(self, active_track_ids, current_time=None):
        """
        Remove Kalman filters for tracks that are no longer active.

        :param active_track_ids: Set of track_ids seen in the current frame.
        :param current_time: Current timestamp for timeout-based cleanup.
        """
        stale_ids = []
        for tid in list(self._filters.keys()):
            if tid not in active_track_ids:
                if current_time is not None:
                    last = self._last_seen.get(tid, 0)
                    if (current_time - last) > self.stale_timeout:
                        stale_ids.append(tid)
                else:
                    stale_ids.append(tid)

        for tid in stale_ids:
            del self._filters[tid]
            self._last_seen.pop(tid, None)

    def get_filter_count(self):
        """Return the number of active Kalman filters (for diagnostics)."""
        return len(self._filters)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_filter(self, x, y, vx, vy):
        """
        Create and initialise a Kalman filter with state [x, y, vx, vy].

        :param x:  Initial x position.
        :param y:  Initial y position.
        :param vx: Initial x velocity (0 if not enough history).
        :param vy: Initial y velocity (0 if not enough history).
        :return: Configured filterpy.kalman.KalmanFilter instance.
        """
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State vector: [x, y, vx, vy]
        kf.x = np.array([x, y, vx, vy], dtype=np.float64)

        # State transition matrix (constant velocity, dt=1 as placeholder)
        kf.F = self._transition_matrix(dt=1.0)

        # Measurement matrix: we observe [x, y] only
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Measurement noise covariance
        kf.R = np.eye(2, dtype=np.float64) * self.measurement_noise

        # Process noise covariance (will be recomputed with actual dt)
        kf.Q = self._process_noise_matrix(dt=1.0)

        # Initial state covariance — high uncertainty
        kf.P = np.eye(4, dtype=np.float64) * self.initial_covariance

        return kf

    def _transition_matrix(self, dt):
        """
        Constant-velocity state transition matrix for a given dt.

            x'  = x + vx * dt
            y'  = y + vy * dt
            vx' = vx
            vy' = vy
        """
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

    def _process_noise_matrix(self, dt):
        """
        Discrete white-noise acceleration model for process noise Q.

        Assumes acceleration is a zero-mean white noise with variance
        `self.process_noise`. This is the standard piecewise-constant
        white-noise model used in constant-velocity trackers.
        """
        q = self.process_noise
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        return np.array([
            [dt4 / 4, 0,       dt3 / 2, 0      ],
            [0,       dt4 / 4, 0,       dt3 / 2],
            [dt3 / 2, 0,       dt2,     0      ],
            [0,       dt3 / 2, 0,       dt2    ],
        ], dtype=np.float64) * q

    def _bootstrap_velocity(self, history):
        """
        Estimate initial velocity from the position history.

        If there are fewer than `min_history` points, returns (0, 0).
        Otherwise, uses the last two points to compute finite-difference velocity.

        :param history: List of (x, y, timestamp) tuples.
        :return: (vx, vy) in units per second.
        """
        if len(history) < self.min_history:
            return 0.0, 0.0

        # Transform positions if calibration is available
        x1, y1, t1 = history[-2]
        x2, y2, t2 = history[-1]

        if self.transformer and self.transformer.is_calibrated():
            x1, y1 = self.transformer.pixel_to_world(x1, y1)
            x2, y2 = self.transformer.pixel_to_world(x2, y2)

        dt = t2 - t1
        if dt <= 0:
            return 0.0, 0.0

        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        return vx, vy

    def _compute_dt(self, history):
        """
        Compute the time delta between the two most recent history entries.
        Falls back to a small positive value if history is too short.

        :param history: List of (x, y, timestamp) tuples.
        :return: dt in seconds (always > 0).
        """
        if len(history) >= 2:
            dt = history[-1][2] - history[-2][2]
            if dt > 0:
                return dt
        # Fallback: assume ~30 fps
        return 1.0 / 30.0

    def _predict_future(self, track_id, current_time):
        """
        Generate multi-step future predictions by projecting the Kalman
        filter state forward without measurements.

        Uses a COPY of the filter so the real filter state is not corrupted.

        :param track_id: Vehicle track ID.
        :param current_time: Current frame timestamp.
        :return: PredictedTrajectory dict.
        """
        kf = self._filters[track_id]

        # Current velocity from filter state
        vx = float(kf.x[2])
        vy = float(kf.x[3])

        # Work on a deep copy so speculative predictions don't pollute
        # the real filter state
        kf_copy = copy.deepcopy(kf)

        predictions = []
        prev_horizon = 0.0

        for horizon in sorted(self.prediction_horizons):
            dt = horizon - prev_horizon
            if dt <= 0:
                continue
            kf_copy.F = self._transition_matrix(dt)
            kf_copy.Q = self._process_noise_matrix(dt)
            kf_copy.predict()

            px = float(kf_copy.x[0])
            py = float(kf_copy.x[1])
            future_t = current_time + horizon

            # Convert back to pixel space if we're working in world coords
            if self.transformer and self.transformer.is_calibrated():
                px, py = self.transformer.world_to_pixel(px, py)

            predictions.append((px, py, future_t))
            prev_horizon = horizon

        return {
            "track_id": track_id,
            "predictions": predictions,
            "velocity": (vx, vy),
        }
