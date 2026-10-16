"""Deterministic contract checks for court-plane calibration quality gates."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from backend.calibration import CalibrationInputError, calibrate_court_homography


class CourtCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.court = np.array([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4], [3.05, 6.7]], dtype=np.float64)
        self.homography = np.array([[45, 8, 320], [5, 32, 90], [0.01, 0.015, 1]], dtype=np.float64)
        self.image = cv2.perspectiveTransform(self.court.reshape(-1, 1, 2), self.homography).reshape(-1, 2)

    def test_metric_calibration_round_trips_when_unit_is_known(self) -> None:
        result = calibrate_court_homography(self.image, self.court, court_unit="m")
        self.assertTrue(result.metric_enabled)
        self.assertEqual(result.status, "metric_ready")
        self.assertLess(result.reprojection_rmse_px, 0.001)
        recovered = result.image_to_court(self.image[:1], require_metric=True)
        self.assertTrue(np.allclose(recovered, self.court[:1], atol=1e-5))

    def test_unknown_unit_keeps_homography_but_disables_metric_claims(self) -> None:
        result = calibrate_court_homography(self.image, self.court, court_unit="unknown")
        self.assertTrue(result.quality_passed)
        self.assertFalse(result.metric_enabled)
        self.assertEqual(result.status, "homography_only")
        self.assertIn("court_unit_is_not_a_recognized_linear_unit", result.metric_disabled_reasons)

    def test_collinear_points_are_rejected(self) -> None:
        with self.assertRaises(CalibrationInputError):
            calibrate_court_homography(
                [[0, 0], [1, 0], [2, 0], [3, 0]],
                [[0, 0], [1, 0], [2, 0], [3, 0]],
                court_unit="m",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
