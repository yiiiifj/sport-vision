"""Dry-run storage retention tests; no test applies a deletion plan."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.retention import plan_portal_cleanup


class RetentionPlanTests(unittest.TestCase):
    def test_only_old_uuid_owned_files_and_job_directories_are_candidates(self) -> None:
        now = datetime(2026, 10, 16, tzinfo=timezone.utc)
        old_timestamp = (now - timedelta(hours=48)).timestamp()
        fresh_timestamp = (now - timedelta(hours=2)).timestamp()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            uploads = root / "uploads"
            outputs = root / "outputs"
            uploads.mkdir()
            outputs.mkdir()
            old_id = "a" * 32
            fresh_id = "b" * 32
            for name in (f"{old_id}.mp4", f"{old_id}.json"):
                path = uploads / name
                path.write_bytes(b"old")
                os.utime(path, (old_timestamp, old_timestamp))
            for name in (f"{fresh_id}.mp4", f"{fresh_id}.json"):
                path = uploads / name
                path.write_bytes(b"fresh")
                os.utime(path, (fresh_timestamp, fresh_timestamp))
            (uploads / "notes.txt").write_text("do not touch", encoding="utf-8")
            old_job = outputs / ("c" * 32)
            old_job.mkdir()
            (old_job / "summary.json").write_text("{}", encoding="utf-8")
            os.utime(old_job, (old_timestamp, old_timestamp))
            unknown_job = outputs / "manual-results"
            unknown_job.mkdir()
            os.utime(unknown_job, (old_timestamp, old_timestamp))

            plan = plan_portal_cleanup(
                older_than_hours=24,
                now=now,
                upload_dir=uploads,
                output_dir=outputs,
            )

        paths = {candidate.relative_path for candidate in plan.candidates}
        self.assertEqual(paths, {f"{old_id}.mp4", f"{old_id}.json", "c" * 32})
        self.assertTrue(all(candidate.root.name in {"uploads", "outputs"} for candidate in plan.candidates))

    def test_newer_member_keeps_the_whole_upload_group(self) -> None:
        now = datetime(2026, 10, 16, tzinfo=timezone.utc)
        old_timestamp = (now - timedelta(hours=48)).timestamp()
        fresh_timestamp = (now - timedelta(hours=2)).timestamp()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            uploads = root / "uploads"
            outputs = root / "outputs"
            uploads.mkdir()
            outputs.mkdir()
            upload_id = "d" * 32
            old_media = uploads / f"{upload_id}.mp4"
            fresh_manifest = uploads / f"{upload_id}.json"
            old_media.write_bytes(b"media")
            fresh_manifest.write_text("{}", encoding="utf-8")
            os.utime(old_media, (old_timestamp, old_timestamp))
            os.utime(fresh_manifest, (fresh_timestamp, fresh_timestamp))

            plan = plan_portal_cleanup(
                older_than_hours=24,
                now=now,
                upload_dir=uploads,
                output_dir=outputs,
            )

        self.assertEqual(plan.candidates, ())


if __name__ == "__main__":
    unittest.main(verbosity=2)
