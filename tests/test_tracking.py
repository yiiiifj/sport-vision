"""Contract checks for dependency-light multi-person association."""

from __future__ import annotations

import unittest

from backend.tracking import TrackManager, TrackingInputError


class TrackManagerTests(unittest.TestCase):
    def test_ids_stay_stable_when_detector_order_changes(self) -> None:
        manager = TrackManager(prefer_hungarian=False)
        first = manager.update([
            {"center": [100, 100], "scale": 40, "confidence": 0.9},
            {"center": [300, 100], "scale": 40, "confidence": 0.9},
        ])
        self.assertEqual([(item.track_id, item.detection_index) for item in first], [(1, 0), (2, 1)])

        # Same people arrive in reverse detector order on the next frame.
        second = manager.update([
            {"center": [302, 100], "scale": 40, "confidence": 0.9},
            {"center": [102, 100], "scale": 40, "confidence": 0.9},
        ])
        by_detection = {item.detection_index: item for item in second if item.matched}
        self.assertEqual(by_detection[1].track_id, 1)
        self.assertEqual(by_detection[0].track_id, 2)
        self.assertTrue(all(item.association_cost is not None for item in by_detection.values()))

    def test_short_occlusion_retains_and_recovers_identity(self) -> None:
        manager = TrackManager(prefer_hungarian=False)
        created = manager.update([{"center": [100, 100], "scale": 50, "confidence": 1.0}])[0]
        occluded = manager.update([])[0]
        recovered = manager.update([{"center": [102, 100], "scale": 50, "confidence": 1.0}])[0]

        self.assertEqual(occluded.track_id, created.track_id)
        self.assertTrue(occluded.occluded)
        self.assertEqual(occluded.missed, 1)
        self.assertEqual(recovered.track_id, created.track_id)
        self.assertFalse(recovered.occluded)
        self.assertEqual(recovered.missed, 0)
        self.assertIsNotNone(recovered.association_cost)

    def test_invalid_center_and_frame_index_are_rejected(self) -> None:
        manager = TrackManager(prefer_hungarian=False)
        with self.assertRaises(TrackingInputError):
            manager.update([{"center": [float("nan"), 0]}])
        manager.update([], frame_index=5)
        with self.assertRaises(TrackingInputError):
            manager.update([], frame_index=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
