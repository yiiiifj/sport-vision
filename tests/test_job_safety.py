"""Pure job-runner safety contracts without launching an engine process."""

from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from backend import jobs
from backend.jobs import MAX_LOG_LINE_CHARACTERS, _public_job_error, _sanitize_log_line


class JobSafetyTests(unittest.TestCase):
    def test_log_lines_drop_ansi_nuls_and_are_bounded(self) -> None:
        cleaned = _sanitize_log_line("\x1b[31mwarning\x1b[0m\x00")
        self.assertEqual(cleaned, "warning")
        long_line = _sanitize_log_line("x" * (MAX_LOG_LINE_CHARACTERS + 40))
        self.assertTrue(long_line.endswith("…[truncated]"))
        self.assertLessEqual(len(long_line), MAX_LOG_LINE_CHARACTERS + len(" …[truncated]"))

    def test_public_error_hides_local_exception_text(self) -> None:
        error = OSError(r"C:\private\model\secret.task is missing")
        message = _public_job_error(error)
        self.assertEqual(message, "Engine process could not be started or monitored")
        self.assertNotIn("private", message)
        self.assertIn("execution failed", _public_job_error(RuntimeError("details")))
        self.assertIn("invalid", _public_job_error(ValueError("details")))
        self.assertIn("could not", _public_job_error(subprocess.SubprocessError("details")))

    def test_shutdown_marks_nonterminal_jobs_interrupted_without_a_real_process(self) -> None:
        class MemoryStore:
            def __init__(self, _path):
                self.saved = []

            def load(self):
                return {}

            def save(self, job):
                self.saved.append(dict(job))

        with patch.object(jobs, "JobStore", MemoryStore):
            manager = jobs.JobManager()
            manager._jobs["x" * 32] = {
                "id": "x" * 32,
                "status": "running",
                "created_at": "fixture",
                "started_at": "fixture",
                "finished_at": None,
                "error": None,
            }
            manager.shutdown()

        interrupted = manager._jobs["x" * 32]
        self.assertEqual(interrupted["status"], "interrupted")
        self.assertIn("shut down", interrupted["error"])

    def test_summary_exposes_only_status_counts(self) -> None:
        manager = object.__new__(jobs.JobManager)
        # A tiny stand-in lock supports the context manager used by summary.
        class Lock:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        manager._lock = Lock()
        manager._jobs = {
            "a": {"status": "queued", "video_path": "must-not-leak"},
            "b": {"status": "running"},
            "c": {"status": "complete"},
        }
        self.assertEqual(
            manager.summary(),
            {"queued_or_running": 2, "complete": 1, "queued": 1, "running": 1},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
