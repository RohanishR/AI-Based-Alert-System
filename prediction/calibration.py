"""
Coordinate Transformer — pixel-to-real-world conversion via homography.

Provides a pluggable interface for the prediction module to operate in
real-world coordinates (meters) instead of raw pixel space. When no
calibration is available, the prediction module works in pixel space and
this module is not used.

Usage:
    1. Collect ≥ 4 reference point pairs (pixel coords ↔ real-world coords)
       from known landmarks in the camera view (e.g., lane markings of known
       width, crosswalk dimensions).
    2. Instantiate CoordinateTransformer(src_points, dst_points).
    3. Pass the transformer to TrajectoryPredictor so positions are converted
       before Kalman filtering, making velocity/TTC values physically meaningful.
"""

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class CoordinateTransformer:
    """
    Computes and applies a homography transform between pixel coordinates
    and real-world (ground-plane) coordinates.
    """

    def __init__(self, src_points=None, dst_points=None):
        """
        Initialize the transformer from point correspondences.

        :param src_points: Nx2 array of pixel coordinates (at least 4 points).
        :param dst_points: Nx2 array of corresponding real-world coordinates
                           (meters), same length as src_points.
        """
        self.homography = None
        self.homography_inv = None

        if src_points is not None and dst_points is not None:
            self._compute_homography(
                np.array(src_points, dtype=np.float64),
                np.array(dst_points, dtype=np.float64),
            )

    def _compute_homography(self, src, dst):
        """
        Compute the homography matrix from src (pixel) → dst (world) using
        OpenCV's findHomography with RANSAC.
        """
        if not _HAS_CV2:
            raise RuntimeError(
                "OpenCV is required for homography computation. "
                "Install it with: pip install opencv-python"
            )
        if len(src) < 4 or len(dst) < 4:
            raise ValueError(
                f"At least 4 point pairs are required, got {len(src)}."
            )
        if len(src) != len(dst):
            raise ValueError(
                f"src ({len(src)}) and dst ({len(dst)}) must have equal length."
            )

        self.homography, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if self.homography is None:
            raise ValueError("Homography computation failed — check point pairs.")

        self.homography_inv, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)

    def pixel_to_world(self, x, y):
        """
        Transform a single pixel coordinate to real-world (ground-plane) coords.

        :param x: pixel x coordinate
        :param y: pixel y coordinate
        :return: (world_x, world_y) in real-world units (e.g., meters)
        """
        if self.homography is None:
            raise RuntimeError("No homography computed. Provide point pairs first.")

        pt = np.array([[[x, y]]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(pt, self.homography)
        wx, wy = transformed[0][0]
        return float(wx), float(wy)

    def world_to_pixel(self, wx, wy):
        """
        Transform a real-world coordinate back to pixel space.

        :param wx: world x coordinate
        :param wy: world y coordinate
        :return: (pixel_x, pixel_y)
        """
        if self.homography_inv is None:
            raise RuntimeError("No inverse homography available.")

        pt = np.array([[[wx, wy]]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(pt, self.homography_inv)
        px, py = transformed[0][0]
        return float(px), float(py)

    def save(self, path):
        """
        Save the homography matrices to a .npz file.

        :param path: File path (e.g., 'data/calibration/cam1_homography.npz')
        """
        if self.homography is None:
            raise RuntimeError("No homography to save.")
        np.savez(
            path,
            homography=self.homography,
            homography_inv=self.homography_inv,
        )

    @classmethod
    def load(cls, path):
        """
        Load a previously saved homography from a .npz file.

        :param path: File path to the .npz file
        :return: CoordinateTransformer instance with loaded homography
        """
        data = np.load(path)
        transformer = cls()
        transformer.homography = data["homography"]
        transformer.homography_inv = data["homography_inv"]
        return transformer

    def is_calibrated(self):
        """Check whether a valid homography has been computed or loaded."""
        return self.homography is not None
