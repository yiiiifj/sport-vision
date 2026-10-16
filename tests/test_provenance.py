"""Provenance manifest tests that do not load a model or video decoder."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend import external_worker


class RunManifestTests(unittest.TestCase):
    def test_local_rules_manifest_has_reproducibility_and_scope_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            source = output_dir / "athlete-video.mp4"
            source.write_bytes(b"fixture-not-a-real-video")
            external_worker._write_run_manifest(
                output_dir,
                source,
                "sportiq",
                {"frames_processed": 12, "pose_coverage": 0.75},
            )
            manifest = json.loads((output_dir / "run-manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["execution_mode"], "local_rules")
        self.assertEqual(manifest["input"]["filename"], "athlete-video.mp4")
        self.assertEqual(len(manifest["input"]["sha256"]), 64)
        self.assertIn("source_tree_dirty", manifest)
        self.assertIn("python_version", manifest["runtime"])
        self.assertIn("mediapipe", manifest["runtime"]["package_versions"])
        self.assertEqual(manifest["configuration"]["num_poses"], 1)
        self.assertFalse(manifest["measurement_scope"]["calibrated"])
        self.assertIn("metres", manifest["measurement_scope"]["not_reported"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
