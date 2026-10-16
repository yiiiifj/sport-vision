"""Contract tests for frozen-annotation evaluation and artifact evidence."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from backend.artifacts import write_artifact_manifest
from backend.evaluation import (
    EvaluationError,
    build_experiment_report,
    evaluate_calibration_reprojection,
    evaluate_event_detection,
    evaluate_pose_pck,
    validate_split_manifest,
    verify_artifact_manifest,
)


class EventEvaluationTests(unittest.TestCase):
    def test_class_aware_matching_does_not_double_count_predictions(self) -> None:
        result = evaluate_event_detection(
            [
                {"type": "serve", "frame": 10},
                {"type": "serve", "frame": 11},
                {"type": "smash", "frame": 30},
            ],
            [
                {"type": "serve", "frame": 10},
                {"type": "smash", "frame": 32},
            ],
            tolerance_frames=2,
        )

        self.assertEqual(result["true_positive"], 2)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 0)
        self.assertEqual(result["micro_f1"], 0.8)
        self.assertEqual(result["per_class"]["serve"]["false_positive"], 1)

    def test_bad_event_contract_is_rejected(self) -> None:
        with self.assertRaises(EvaluationError):
            evaluate_event_detection([{"type": "serve", "frame": -1}], [])


class PoseEvaluationTests(unittest.TestCase):
    def test_pck_counts_missing_predictions_as_failures(self) -> None:
        ground_truth = {
            "frames": [
                {
                    "frame": 1,
                    "normalization_scale_px": 100,
                    "joints": {
                        "wrist": {"x": 10, "y": 10, "visible": True},
                        "ankle": {"x": 50, "y": 50, "visible": True},
                        "hidden": {"x": 0, "y": 0, "visible": False},
                    },
                }
            ]
        }
        prediction = {
            "frames": [
                {"frame": 1, "joints": {"wrist": {"x": 13, "y": 14}}}]
            }

        result = evaluate_pose_pck(prediction, ground_truth, thresholds=(0.05, 0.1))

        self.assertEqual(result["visible_joint_count"], 2)
        self.assertEqual(result["missing_prediction_count"], 1)
        self.assertEqual(result["pck@0.05"], 0.5)
        self.assertEqual(result["pck@0.1"], 0.5)
        self.assertEqual(result["mean_error_px"], 5.0)

    def test_missing_normalization_scale_is_rejected(self) -> None:
        with self.assertRaises(EvaluationError):
            evaluate_pose_pck(
                {"frames": []},
                {"frames": [{"frame": 0, "joints": {"wrist": {"x": 0, "y": 0}}}]},
            )


class CalibrationEvaluationTests(unittest.TestCase):
    def test_held_out_reprojection_has_expected_rmse(self) -> None:
        result = evaluate_calibration_reprojection(
            [[0, 0], [3, 4]],
            [[0, 0], [0, 0]],
        )

        self.assertEqual(result["point_count"], 2)
        self.assertAlmostEqual(result["rmse_px"], 3.535534, places=5)
        self.assertEqual(result["max_error_px"], 5.0)


class ArtifactEvaluationTests(unittest.TestCase):
    def test_integrity_verification_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            artifact = output_dir / "summary.json"
            artifact.write_text('{"status":"ok"}', encoding="utf-8")
            write_artifact_manifest(
                output_dir,
                run_id="a" * 32,
                engine_id="sportiq",
                execution_mode="local_rules",
            )
            verified = verify_artifact_manifest(output_dir)
            self.assertTrue(verified["valid"])
            self.assertEqual(verified["verified_artifact_count"], 1)

            artifact.write_text('{"status":"changed"}', encoding="utf-8")
            changed = verify_artifact_manifest(output_dir)
            self.assertFalse(changed["valid"])
            self.assertEqual(changed["findings"][0]["status"], "hash_mismatch")

    def test_integrity_verification_rejects_an_unmanifested_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            (output_dir / "summary.json").write_text("{}", encoding="utf-8")
            write_artifact_manifest(
                output_dir,
                run_id="b" * 32,
                engine_id="sportiq",
                execution_mode="local_rules",
            )
            (output_dir / "later-added.txt").write_text("not in the manifest", encoding="utf-8")
            result = verify_artifact_manifest(output_dir)

        self.assertFalse(result["valid"])
        self.assertIn(
            {"name": "later-added.txt", "status": "unmanifested_file"},
            result["findings"],
        )


class BenchmarkSplitTests(unittest.TestCase):
    def test_split_validator_detects_cross_split_leakage(self) -> None:
        result = validate_split_manifest({
            "schema_version": 1,
            "dataset_name": "private-racket-v1",
            "annotation_version": "2026-10-16",
            "splits": {
                "train": ["clip-001", "clip-002"],
                "validation": ["clip-003"],
                "test": ["clip-002", "clip-004"],
            },
        })

        self.assertFalse(result["valid"])
        self.assertIn(
            {"kind": "cross_split_leakage", "clip_id": "clip-002", "splits": ["test", "train"]},
            result["findings"],
        )

    def test_disjoint_nonempty_splits_are_valid(self) -> None:
        result = validate_split_manifest({
            "dataset_name": "private-racket-v1",
            "annotation_version": "2026-10-16",
            "splits": {
                "train": ["clip-001"],
                "validation": ["clip-002"],
                "test": ["clip-003"],
            },
        })

        self.assertTrue(result["valid"])
        self.assertEqual(result["unique_clip_count"], 3)


class ExperimentEvidenceReportTests(unittest.TestCase):
    def test_report_binds_split_run_and_metric_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "outputs" / ("c" * 32)
            output_dir.mkdir(parents=True)
            (output_dir / "summary.json").write_text("{}", encoding="utf-8")
            (output_dir / "run-manifest.json").write_text(json.dumps({
                "source_revision": "abc123",
                "source_tree_dirty": False,
                "rules_version": "2.0.0",
                "runtime": {"python_version": "3.x"},
                "configuration": {"random_seed": 7},
            }), encoding="utf-8")
            write_artifact_manifest(
                output_dir, run_id="c" * 32, engine_id="sportiq", execution_mode="local_rules"
            )
            split_path = root / "split.json"
            split_path.write_text(json.dumps({
                "dataset_name": "private-racket-v1",
                "annotation_version": "2026-10-16",
                "splits": {"train": ["a"], "validation": ["b"], "test": ["c"]},
            }), encoding="utf-8")
            metric_path = root / "events.json"
            metric_path.write_text(json.dumps({"metric": "event_detection_frame_tolerance"}), encoding="utf-8")

            result = build_experiment_report({
                "experiment_id": "baseline-001",
                "split_manifest": "split.json",
                "runs": [{"run_id": "c" * 32, "output_dir": "outputs/" + "c" * 32}],
                "evaluations": [{
                    "name": "events-test",
                    "report": "events.json",
                    "expected_metric": "event_detection_frame_tolerance",
                }],
            }, base_dir=root)

        self.assertTrue(result["valid"])
        self.assertTrue(result["runs"][0]["artifact_integrity"]["valid"])
        self.assertEqual(result["evaluations"][0]["reported_metric"], "event_detection_frame_tolerance")

    def test_report_rejects_metric_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "run"
            output_dir.mkdir()
            (output_dir / "run-manifest.json").write_text("{}", encoding="utf-8")
            write_artifact_manifest(output_dir, run_id="run-1", engine_id="sportiq", execution_mode="local_rules")
            (root / "split.json").write_text(json.dumps({
                "dataset_name": "d", "annotation_version": "v",
                "splits": {"train": ["a"], "validation": ["b"], "test": ["c"]},
            }), encoding="utf-8")
            (root / "metric.json").write_text(json.dumps({"metric": "pose_pck_2d"}), encoding="utf-8")
            result = build_experiment_report({
                "experiment_id": "bad-metric",
                "split_manifest": "split.json",
                "runs": [{"run_id": "run-1", "output_dir": "run"}],
                "evaluations": [{"name": "events", "report": "metric.json", "expected_metric": "event_detection_frame_tolerance"}],
            }, base_dir=root)

        self.assertFalse(result["valid"])
        self.assertFalse(result["evaluations"][0]["metric_matches"])
        self.assertFalse(result["runs"][0]["provenance_valid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
