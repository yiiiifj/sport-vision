"""Small SQLite persistence layer for local managed-job state."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class JobStore:
    """Persist public job state across a local server restart.

    Running subprocesses cannot safely be resumed after a process restart, so
    callers mark those records `interrupted` on startup. Completed metadata and
    artifact links remain inspectable.
    """

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path, timeout=10)

    def load(self) -> dict[str, dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id, payload FROM jobs").fetchall()
        result: dict[str, dict] = {}
        for job_id, payload in rows:
            try:
                record = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(record, dict) and record.get("id") == job_id:
                result[job_id] = record
        return result

    def save(self, job: dict) -> None:
        payload = json.dumps(job, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO jobs(id, payload, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at""",
                (job["id"], payload, job.get("finished_at") or job.get("started_at") or job["created_at"]),
            )
