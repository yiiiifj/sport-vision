"""Model-free contracts for multi-person pose candidates and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import unittest

import numpy as np

from backend.multi_person import MultiPersonPoseAnalyzer, PrimaryTrackSelectionError
from backend.pose_analyzer import PoseAnalyzer
from backend.tracking import TrackManager


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.9


def _landmarks(*, visibility: float = 0.9) -> list[FakeLandmark]:
    """Produce enough landmark-shaped objects for pure candidate conversion."""

    points = [FakeLandmark(0.50, 0.20, visibility=visibility) for _ in range(33)]
    points[11] = FakeLandmark(0.40, 0.30, visibility=visibility)
    points[12] = FakeLandmark(0.60, 0.30, visibility=visibility)
    points[23] = FakeLandmark(0.43, 0.65, visibility=visibility)
    points[24] = FakeLandmark(0.57, 0.65, visibility=visibility)
    points[27] = FakeLandmark(0.40, 0.90, visibility=visibility)
    points[28] = FakeLandmark(0.60, 0.90, visibility=visibility)
    return points


def _candidate(index: int, x: float, y: float) -> dict[str, Any]:
    return {
        "candidate_index": index,
        "center": {"x": x, "y": y},
        "scale": 80.0,
        "confidence": 0.9,
        "keypoints": [
            {"id": 11, "x": x - 10, "y": y - 25},
            {"id": 12, "x": x + 10, "y": y - 25},
            {"id": 23, "x": x - 8, "y": y},
            {"id": 24, "x": x + 8, "y": y},
        ],
    }


class FakeCandidateDetector:
    """Sequence-driven detector that never loads a MediaPipe model."""

    def __init__(self, frames: list[list[dict[str, Any]]]) -> None:
        self.frames = frames
        self.call_count = 0

    def detect_pose_candidates(self, frame_rgb, timestamp_ms=None):
        del frame_rgb, timestamp_ms
        frame = self.frames[self.call_count]
        self.call_count += 1
        return [dict(candidate) for candidate in frame]

    def get_multi_pose_capabilities(self) -> dict[str, Any]:
        return {
            "configured_candidate_limit": 4,
            "multi_person_configured": True,
            "last_media_pipe_pose_count": 2,
            "last_usable_candidate_count": 2,
            "limitations": ["Fake detector; model-free unit test."],
        }


class PoseCandidateConversionTests(unittest.TestCase):
    def test_landmark_conversion_has_trackable_geometry_without_model(self) -> None:
        candidate = PoseAnalyzer.candidate_from_landmarks(
            _landmarks(),
            width=1000,
            height=500,
            candidate_index=2,
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate["candidate_index"], 2)
        self.assertEqual(candidate["center_kind"], "pelvis_midpoint_px")
        self.assertAlmostEqual(candidate["center"]["x"], 500.0)
        self.assertAlmostEqual(candidate["center"]["y"], 325.0)
        self.assertGreater(candidate["scale"], 0.0)
        self.assertEqual(candidate["scale_kind"], "visible_landmark_bbox_max_extent_px")
        self.assertEqual(len(candidate["keypoints"]), len(PoseAnalyzer.LANDMARK_NAMES))

    def test_low_visibility_pose_is_not_promoted_to_track_candidate(self) -> None:
        candidate = PoseAnalyzer.candidate_from_landmarks(
            _landmarks(visibility=0.1),
            width=1000,
            height=500,
            candidate_index=0,
        )
        self.assertIsNone(candidate)


class MultiPersonPoseAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = FakeCandidateDetector([
            [_candidate(0, 100, 120), _candidate(1, 300, 120)],
            [_candidate(7, 302, 120), _candidate(8, 102, 120)],
            [],
        ])
        self.analyzer = MultiPersonPoseAnalyzer(
            detector=self.detector,
            tracker=TrackManager(prefer_hungarian=False),
        )
        self.frame = np.zeros((240, 400, 3), dtype=np.uint8)

    def test_track_ids_survive_detector_reordering_and_primary_is_explicit(self) -> None:
        first = self.analyzer.process_frame(self.frame, frame_index=0)
        self.assertIsNone(first["primary"])
        self.assertTrue(first["selection_required"])
        self.assertEqual([item["track_id"] for item in first["candidates"]], [1, 2])
        self.assertTrue(first["capabilities"]["multi_person_configured"])
        self.assertEqual(first["capabilities"]["configured_candidate_limit"], 4)

        # MediaPipe-like candidate order changes; application track IDs do not.
        second = self.analyzer.process_frame(self.frame, frame_index=1, primary_track_id=1)
        self.assertEqual([item["track_id"] for item in second["candidates"]], [2, 1])
        self.assertEqual(second["primary_track_id"], 1)
        self.assertTrue(second["primary"]["visible"])
        self.assertEqual(second["primary"]["candidate"]["track_id"], 1)
        self.assertEqual(second["primary"]["candidate"]["candidate_index"], 8)

    def test_selected_track_is_not_replaced_during_short_occlusion(self) -> None:
        self.analyzer.process_frame(self.frame, frame_index=0)
        self.analyzer.process_frame(self.frame, frame_index=1, primary_track_id=1)
        occluded = self.analyzer.process_frame(self.frame, frame_index=2, primary_track_id=1)

        self.assertEqual(occluded["candidates"], [])
        self.assertFalse(occluded["primary"]["visible"])
        self.assertIsNone(occluded["primary"]["candidate"])
        self.assertEqual(occluded["primary"]["track"]["track_id"], 1)
        self.assertTrue(occluded["primary"]["track"]["occluded"])

    def test_unknown_primary_id_is_rejected_instead_of_auto_selecting(self) -> None:
        with self.assertRaises(PrimaryTrackSelectionError):
            self.analyzer.process_frame(self.frame, frame_index=0, primary_track_id=99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
