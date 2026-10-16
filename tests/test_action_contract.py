"""Rule-action output contracts without loading a pose model."""

from __future__ import annotations

import unittest

from backend.action_recognizer import ActionRecognizer


def _keypoints(offset: float = 0.0) -> list[dict]:
    points = {
        "nose": (100, 40),
        "left_shoulder": (80, 80),
        "right_shoulder": (120, 80),
        "left_wrist": (70 + offset, 110),
        "right_wrist": (130 + offset, 110),
        "left_hip": (85, 150),
        "right_hip": (115, 150),
    }
    return [
        {"name": name, "x": x, "y": y, "visibility": 0.9}
        for name, (x, y) in points.items()
    ]


class ActionContractTests(unittest.TestCase):
    def test_rule_score_is_explicitly_not_a_probability(self) -> None:
        recognizer = ActionRecognizer()
        result = None
        for frame in range(5):
            result = recognizer.update(_keypoints(float(frame)), {})
        assert result is not None

        self.assertEqual(result["score_kind"], "rule_match_not_probability")
        self.assertEqual(result["rule_match_score"], result["confidence"])
        self.assertGreaterEqual(result["rule_match_score"], 0)
        self.assertLessEqual(result["rule_match_score"], 1)

    def test_generic_mode_never_emits_a_racket_specific_label(self) -> None:
        recognizer = ActionRecognizer(mode="generic")
        result = None
        for frame in range(5):
            result = recognizer.update(_keypoints(float(frame * 12)), {})
        assert result is not None

        self.assertEqual(result["analysis_mode"], "generic")
        self.assertIn(result["action"], {"ready", "moving", "arm_raise", "arm_swing"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
