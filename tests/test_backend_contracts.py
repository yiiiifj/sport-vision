"""Fast, dependency-light contract tests for the Sport Vision backend.

These checks intentionally do not invoke any model inference, download model
weights, or start a web server.  They cover the boundaries that keep the
portal's multi-engine job system safe and predictable:

* integration registry data exposed to the UI;
* specialist, local-rule, and unavailable preflight decisions;
* opaque upload-id validation (including path-traversal rejection); and
* deterministic local-rule event and CSV helpers.

Run from the repository root with:

    .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from backend import external_worker, integrations, jobs, uploads


EXPECTED_INTEGRATION_IDS = {
    "sport-vision",
    "good-badminton",
    "good-tennis",
    "open-hoop-vision",
    "football-analytics",
    "tactiq",
    "football-intelligence-hub",
    "sportiq",
    "sports2d",
    "athlete-pose-3d",
    "paddle-video",
}


class IntegrationRegistryTests(unittest.TestCase):
    """The frontend catalogue must remain complete and serializable."""

    def test_catalogue_has_all_expected_unique_engines(self) -> None:
        catalogue = integrations.list_integrations()
        ids = [item["id"] for item in catalogue]

        self.assertEqual(set(ids), EXPECTED_INTEGRATION_IDS)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(catalogue), len(EXPECTED_INTEGRATION_IDS))

    def test_catalogue_entries_are_safe_ui_records(self) -> None:
        for item in integrations.list_integrations():
            with self.subTest(engine=item["id"]):
                self.assertIn(item["status"], {"available", "incomplete"})
                self.assertIsInstance(item["capabilities"], list)
                self.assertTrue(item["name"])
                self.assertTrue(item["entrypoint"])
                self.assertNotIn("path", item, "Host paths must not be sent to browsers")

    def test_managed_engines_are_registered_and_advertised(self) -> None:
        listed = {item["id"]: item for item in integrations.list_integrations()}
        self.assertTrue(integrations.MANAGED_VIDEO_ENGINES)
        self.assertTrue(integrations.COURT_TEMPLATE_ENGINES <= integrations.MANAGED_VIDEO_ENGINES)

        for engine_id in integrations.MANAGED_VIDEO_ENGINES:
            with self.subTest(engine=engine_id):
                self.assertIn(engine_id, listed)
                self.assertTrue(listed[engine_id]["job_supported"])

    def test_missing_sibling_source_uses_bundled_video_fallback(self) -> None:
        """Cloud deployments must keep managed upload analysis actionable."""
        item = next(engine for engine in integrations.INTEGRATIONS if engine.id == "sportiq")
        with patch.object(integrations, "PORTAL_DIR", Path("/missing-portal")):
            self.assertEqual(item.as_dict()["status"], "incomplete")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models"
            model_dir.mkdir()
            (model_dir / "pose_landmarker_lite.task").write_bytes(b"model")
            with patch.object(integrations, "PORTAL_DIR", Path(temp_dir)):
                self.assertEqual(item.as_dict()["status"], "available")

    def test_unknown_integration_is_not_resolved(self) -> None:
        self.assertIsNone(integrations.get_integration("not-a-real-engine"))
        self.assertIsNone(integrations.get_integration_definition("not-a-real-engine"))


class PreflightModeTests(unittest.TestCase):
    """Preflight decisions should be deterministic without model imports."""

    def _fake_integration(self, root: Path, *, source_ready: bool) -> SimpleNamespace:
        source_dir = root / "engine"
        entrypoint = source_dir / "main.py"
        if source_ready:
            source_dir.mkdir()
            entrypoint.write_text("# test entrypoint\n", encoding="utf-8")
        return SimpleNamespace(
            path=source_dir,
            entrypoint_path=entrypoint,
            required_assets=(),
        )

    def test_specialist_mode_requires_source_python_and_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake = self._fake_integration(root, source_ready=True)
            fallback_model = root / "pose.task"
            fallback_model.write_bytes(b"test")
            with (
                patch.object(jobs, "get_integration_definition", return_value=fake),
                patch.object(jobs, "_python_for", return_value=sys.executable),
                patch.object(jobs, "_missing_packages", return_value=[]),
                patch.object(jobs, "FALLBACK_MODEL", fallback_model),
            ):
                result = jobs.preflight("good-badminton")

        self.assertTrue(result["ready"])
        self.assertTrue(result["specialist_ready"])
        self.assertEqual(result["execution_mode"], "specialist")
        self.assertTrue(result["requires_template"])

    def test_local_rules_mode_handles_missing_specialist_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake = self._fake_integration(root, source_ready=False)
            fallback_model = root / "pose.task"
            fallback_model.write_bytes(b"test")
            with (
                patch.object(jobs, "get_integration_definition", return_value=fake),
                patch.object(jobs, "_python_for", return_value=sys.executable),
                patch.object(jobs, "_missing_packages", return_value=[]),
                patch.object(jobs, "FALLBACK_MODEL", fallback_model),
            ):
                result = jobs.preflight("good-tennis")

        self.assertTrue(result["ready"])
        self.assertFalse(result["specialist_ready"])
        self.assertTrue(result["fallback_ready"])
        self.assertEqual(result["execution_mode"], "local_rules")
        self.assertFalse(result["requires_template"])

    def test_unavailable_mode_is_reported_when_fallback_requirements_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake = self._fake_integration(root, source_ready=False)
            fallback_model = root / "pose.task"
            fallback_model.write_bytes(b"test")
            with (
                patch.object(jobs, "get_integration_definition", return_value=fake),
                patch.object(jobs, "_python_for", return_value=sys.executable),
                patch.object(jobs, "_missing_packages", return_value=["mediapipe"]),
                patch.object(jobs, "FALLBACK_MODEL", fallback_model),
            ):
                result = jobs.preflight("sports2d")

        self.assertFalse(result["ready"])
        self.assertFalse(result["fallback_ready"])
        self.assertEqual(result["execution_mode"], "unavailable")
        self.assertIn("mediapipe", result["fallback_missing_packages"])

    def test_sportiq_is_never_presented_as_a_specialist_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake = self._fake_integration(root, source_ready=True)
            fallback_model = root / "pose.task"
            fallback_model.write_bytes(b"test")
            with (
                patch.object(jobs, "get_integration_definition", return_value=fake),
                patch.object(jobs, "_python_for", return_value=sys.executable),
                patch.object(jobs, "_missing_packages", return_value=[]),
                patch.object(jobs, "FALLBACK_MODEL", fallback_model),
            ):
                result = jobs.preflight("sportiq")

        self.assertFalse(result["specialist_ready"])
        self.assertEqual(result["execution_mode"], "local_rules")

    def test_unknown_engine_returns_not_found(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            jobs.preflight("not-a-real-engine")
        self.assertEqual(raised.exception.status_code, 404)


class UploadIdSecurityTests(unittest.TestCase):
    """Only portal-issued opaque IDs may resolve to uploaded media."""

    @staticmethod
    def _write_upload(upload_dir: Path, upload_id: str, kind: str, extension: str, content: bytes) -> Path:
        path = upload_dir / f"{upload_id}{extension}"
        path.write_bytes(content)
        record = uploads.create_record(upload_id, kind, extension, f"sample{extension}", len(content))
        uploads.save_record(record)
        return path

    def test_job_resolver_accepts_one_valid_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir)
            upload_id = "d" * 32
            with patch.object(uploads, "UPLOAD_DIR", upload_dir):
                expected = self._write_upload(upload_dir, upload_id, "video", ".mp4", b"video")
                self.assertEqual(uploads.resolve_upload(upload_id, "video"), expected)

    def test_job_resolver_rejects_path_traversal_and_uppercase_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir)
            with patch.object(uploads, "UPLOAD_DIR", upload_dir):
                for malicious_id in ("../" + "d" * 32, "d" * 31 + "/x", "D" * 32, "d" * 32 + ".mp4", "deadbeef"):
                    with self.subTest(upload_id=malicious_id):
                        with self.assertRaises(HTTPException) as raised:
                            uploads.resolve_upload(malicious_id, "video")
                        self.assertEqual(raised.exception.status_code, 400)

    def test_job_resolver_rejects_ambiguous_or_missing_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir)
            upload_id = "d" * 32
            with patch.object(uploads, "UPLOAD_DIR", upload_dir):
                self._write_upload(upload_dir, upload_id, "video", ".mp4", b"video")
                # A second similarly named file cannot alter the manifest-selected record.
                (upload_dir / f"{upload_id}.webm").write_bytes(b"another video")
                self.assertTrue(uploads.resolve_upload(upload_id, "video").name.endswith(".mp4"))
                (upload_dir / f"{upload_id}.mp4").write_bytes(b"changed")
                with self.assertRaises(HTTPException) as raised:
                    uploads.resolve_upload(upload_id, "video")
                self.assertEqual(raised.exception.status_code, 404)

    def test_api_resolver_filters_by_expected_media_kind(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir)
            video_id, image_id = "d" * 32, "c" * 32
            with patch.object(uploads, "UPLOAD_DIR", upload_dir):
                video = self._write_upload(upload_dir, video_id, "video", ".mp4", b"video")
                image = self._write_upload(upload_dir, image_id, "template", ".png", b"image")
                self.assertEqual(uploads.resolve_upload(video_id, "video"), video)
                self.assertEqual(uploads.resolve_upload(image_id, "template"), image)
                with self.assertRaises(HTTPException) as raised:
                    uploads.resolve_upload(image_id, "video")
                self.assertEqual(raised.exception.status_code, 404)


class LocalRuleHelperTests(unittest.TestCase):
    """Rule helpers are deterministic and do not need a pose model."""

    @staticmethod
    def _sample(frame: int, value: float, *, metric: str = "wrist_speed_shoulder_widths_s") -> dict:
        sample = {
            "frame": frame,
            "time_s": round(frame / 30, 3),
            "wrist_speed_px_s": 0.0,
            "wrist_speed_shoulder_widths_s": 0.0,
            "hip_speed_px_s": 0.0,
            "hip_speed_shoulder_widths_s": 0.0,
            "ankle_motion_px_s": 0.0,
            "ankle_motion_shoulder_widths_s": 0.0,
            "knee_flexion_proxy_deg": 0.0,
        }
        sample[metric] = value
        return sample

    def test_peak_events_selects_separated_local_maxima(self) -> None:
        samples = [
            self._sample(1, 0),
            self._sample(2, 300),
            self._sample(3, 100),
            self._sample(13, 0),
            self._sample(14, 300),
            self._sample(15, 0),
        ]

        events = external_worker._peak_events(samples, "wrist_speed_shoulder_widths_s", 2.5, "fast_swing_candidate")

        self.assertEqual([event["frame"] for event in events], [2, 14])
        self.assertTrue(all(event["type"] == "fast_swing_candidate" for event in events))
        self.assertTrue(all(0 < event["heuristic_score"] <= 0.95 for event in events))

    def test_sport_event_mapping_keeps_action_events_and_uses_expected_metric(self) -> None:
        action_events = [{"frame": 1, "type": "serve", "heuristic_score": 0.7}]
        tennis_samples = [
            self._sample(1, 0), self._sample(2, 3), self._sample(3, 0),
        ]
        football_samples = [
            self._sample(1, 0, metric="hip_speed_shoulder_widths_s"),
            self._sample(2, 1, metric="hip_speed_shoulder_widths_s"),
            self._sample(3, 0, metric="hip_speed_shoulder_widths_s"),
        ]

        tennis_events = external_worker._sport_events("good-tennis", tennis_samples, action_events)
        football_events = external_worker._sport_events("football-analytics", football_samples, [])

        self.assertEqual(tennis_events[0], action_events[0])
        self.assertEqual(tennis_events[1]["type"], "fast_swing_candidate")
        self.assertEqual(football_events[0]["type"], "sprint_or_run_candidate")

    def test_metrics_writer_emits_stable_column_order(self) -> None:
        sample = {
            "frame": 1,
            "time_s": 0.033,
            "pelvis_x_norm": 0.5,
            "pelvis_y_norm": 0.4,
            "wrist_speed_px_s": 123.0,
            "wrist_speed_shoulder_widths_s": 2.2,
            "hip_speed_px_s": 80.0,
            "hip_speed_shoulder_widths_s": 1.4,
            "ankle_motion_px_s": 70.0,
            "ankle_motion_shoulder_widths_s": 1.2,
            "knee_flexion_proxy_deg": 45.0,
            "image_spine_lean_deg": 12.0,
            "image_levelness_proxy_pct": 90.0,
            "motion_valid": True,
            "action": "ready",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "frame-metrics.csv"
            external_worker._write_metrics_csv(target, [sample])
            with target.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(rows, [{key: str(value) for key, value in sample.items()}])

    def test_data_quality_summary_is_bounded_and_explains_low_coverage(self) -> None:
        sufficient = external_worker._data_quality_summary(
            frame_count=100,
            pose_frames=95,
            visibility_values=[0.8, 0.9],
            longest_gap_frames=5,
            fps=30,
        )
        insufficient = external_worker._data_quality_summary(
            frame_count=100,
            pose_frames=20,
            visibility_values=[0.2],
            longest_gap_frames=150,
            fps=30,
        )

        self.assertEqual(sufficient["verdict"], "sufficient")
        self.assertEqual(insufficient["verdict"], "insufficient_for_technical_comparison")
        for report in (sufficient, insufficient):
            self.assertGreaterEqual(report["data_quality_score"], 0)
            self.assertLessEqual(report["data_quality_score"], 100)
            self.assertIn("not a technique", report["note"])

    def test_html_report_escapes_external_event_and_summary_text(self) -> None:
        summary = {
            "method": "Rule <script>alert(1)</script>",
            "analysis_quality": {"data_quality_score": 90, "verdict": "sufficient"},
            "pose_coverage": 0.9,
            "pose_frames": 90,
            "frames_processed": 100,
            "event_count": 1,
        }
        events = [{
            "frame": 1,
            "time_s": 0.033,
            "type": "<script>alert(1)</script>",
            "heuristic_score": 0.8,
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            external_worker._write_html_report(output_dir, summary, events)
            report = (output_dir / "analysis-report.html").read_text(encoding="utf-8")

        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", report)
        self.assertNotIn("Rule <script>", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
