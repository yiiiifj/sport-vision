"""Managed, auditable background jobs for engines exposed by Sport Vision.

Only commands defined in this module can run.  The browser supplies upload
identifiers, never shell fragments or filesystem paths, so the portal remains a
safe common entry point for the sibling projects.
"""

from __future__ import annotations

import json
import os
import re
from shutil import which
import subprocess
import sys
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel, Field

from backend.integrations import WORKSPACE_DIR, get_integration_definition
from backend.artifacts import write_artifact_manifest
from backend.job_store import JobStore
from backend.uploads import resolve_upload


SPORT_VISION_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = SPORT_VISION_DIR / "outputs"
STATE_DIR = SPORT_VISION_DIR / "state"
REQUIRED_PACKAGES: dict[str, tuple[str, ...]] = {
    "good-badminton": ("cv2", "ultralytics", "numpy"),
    "good-tennis": ("cv2", "ultralytics", "numpy"),
    "open-hoop-vision": ("cv2", "torch", "ultralytics", "numpy"),
    # Its loader falls back to pickle when joblib is absent, so joblib is not a
    # startup prerequisite for the bundled classifier artifacts.
    "football-analytics": ("cv2", "mediapipe", "ultralytics"),
    "sportiq": ("cv2", "mediapipe", "numpy", "scipy"),
    "sports2d": ("Pose2Sim",),
}
FALLBACK_VIDEO_ENGINES = frozenset(REQUIRED_PACKAGES)
FALLBACK_PACKAGES = ("cv2", "mediapipe", "numpy", "scipy")
FALLBACK_MODEL = SPORT_VISION_DIR / "models" / "pose_landmarker_lite.task"
# SportIQ's documented checkpoint is not available from its upstream releases,
# so its portal adapter intentionally uses the local fallback path.
FALLBACK_ONLY_ENGINES = frozenset({"sportiq"})
MAX_QUEUED_OR_RUNNING_JOBS = max(1, int(os.environ.get("SPORT_VISION_MAX_JOBS", "3")))
JOB_TIMEOUT_SECONDS = max(60, int(os.environ.get("SPORT_VISION_JOB_TIMEOUT_SECONDS", "1800")))
MAX_LOG_LINE_CHARACTERS = 1_600
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-_][0-?]*[ -/]*[@-~]|\[[0-?]*[ -/]*[@-~])")


class JobRequest(BaseModel):
    engine_id: str = Field(min_length=1, max_length=64)
    video_id: str = Field(min_length=32, max_length=32)
    template_id: str | None = Field(default=None, min_length=32, max_length=32)


def _python_for(engine_id: str) -> str:
    """Allow a project-specific virtualenv without exposing a UI command box."""

    variable = "SPORT_VISION_" + engine_id.upper().replace("-", "_") + "_PYTHON"
    return os.environ.get(variable, sys.executable)


def preflight(engine_id: str) -> dict:
    """Check local files without importing a heavyweight model runtime."""

    integration = get_integration_definition(engine_id)
    if integration is None:
        raise HTTPException(status_code=404, detail="Integration not found")
    python_command = _python_for(engine_id)
    python_path = Path(which(python_command) or python_command)
    source_ready = integration.path.is_dir() and integration.entrypoint_path.is_file()
    missing_assets = [asset for asset in integration.required_assets if not (integration.path / asset).is_file()]
    missing_packages = _missing_packages(str(python_path), REQUIRED_PACKAGES.get(engine_id, ())) if python_path.is_file() else []
    specialist_ready = (
        engine_id in REQUIRED_PACKAGES
        and engine_id not in FALLBACK_ONLY_ENGINES
        and source_ready
        and python_path.is_file()
        and not missing_assets
        and not missing_packages
    )
    fallback_missing_packages = _missing_packages(sys.executable, FALLBACK_PACKAGES)
    fallback_ready = (
        engine_id in FALLBACK_VIDEO_ENGINES
        and FALLBACK_MODEL.is_file()
        and not fallback_missing_packages
    )
    execution_mode = "specialist" if specialist_ready else "local_rules" if fallback_ready else "unavailable"
    return {
        "engine_id": engine_id,
        "source_ready": source_ready,
        "python_ready": python_path.is_file(),
        "missing_assets": missing_assets,
        "missing_packages": missing_packages,
        "specialist_ready": specialist_ready,
        "fallback_ready": fallback_ready,
        "execution_mode": execution_mode,
        "requires_template": execution_mode == "specialist" and engine_id in {"good-badminton", "good-tennis"},
        "ready": execution_mode != "unavailable",
        "fallback_missing_packages": fallback_missing_packages,
    }


def _missing_packages(python_path: str, packages: tuple[str, ...]) -> list[str]:
    """Ask the selected interpreter which imports it can resolve, with a timeout."""

    if not packages:
        return []
    probe = (
        "import importlib.util, json; "
        f"packages={list(packages)!r}; "
        "print(json.dumps([p for p in packages if importlib.util.find_spec(p) is None]))"
    )
    try:
        result = subprocess.run(
            [python_path, "-c", probe],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    # A broken interpreter is actionable in exactly the same way as missing
    # dependencies, but avoid exposing host-specific stderr to the browser.
    return list(packages)


def _command_for(request: JobRequest, output_dir: Path, execution_mode: str) -> tuple[list[str], Path]:
    """Build a fixed argv list for a supported non-interactive engine."""

    video = resolve_upload(request.video_id, "video")
    template = resolve_upload(request.template_id, "template") if request.template_id else None

    if execution_mode == "local_rules":
        return (
            [
                sys.executable, "-m", "backend.external_worker", "local-rules",
                "--video", str(video), "--output", str(output_dir / "summary.json"),
                "--label", request.engine_id,
            ],
            SPORT_VISION_DIR,
        )

    def court_command(project: str, script: str, extra_args: list[str]) -> tuple[list[str], Path]:
        if template is None:
            raise HTTPException(
                status_code=422,
                detail="This engine needs a court-template image. Upload it in the template field.",
            )
        return (
            [
                _python_for(request.engine_id), script,
                "--video-path", str(video),
                "--template-path", str(template),
                "--output-dir", str(output_dir),
                "--display", "false",
                *extra_args,
            ],
            WORKSPACE_DIR / project,
        )

    if request.engine_id == "good-badminton":
        return court_command(
            "Good-Badminton-main", "main.py",
            ["--pose-family", "yolo-pose", "--yolo-pose-model", "yolo11n-pose.pt"],
        )
    if request.engine_id == "good-tennis":
        return court_command(
            "Good-Tennis-main", "main.py",
            [
                "--pose-family", "yolo-pose", "--yolo-pose-model", "yolo11s-pose.pt",
                "--person-model", "yolo26s.pt", "--person-tracker", "none",
            ],
        )
    if request.engine_id == "open-hoop-vision":
        return (
            [
                _python_for(request.engine_id), "main.py", "--video", str(video),
                "--output", str(output_dir / "analysis.mp4"),
                "--tracks", str(output_dir / "tracks.jsonl"),
            ],
            WORKSPACE_DIR / "OpenHoopVision",
        )
    if request.engine_id == "football-analytics":
        return (
            [
                _python_for(request.engine_id), "-m", "backend.external_worker",
                "football-analytics", "--video", str(video), "--output", str(output_dir / "summary.json"),
            ],
            SPORT_VISION_DIR,
        )
    if request.engine_id == "sportiq":
        return (
            [
                _python_for(request.engine_id), "-m", "backend.external_worker",
                "sportiq", "--video", str(video), "--output", str(output_dir / "summary.json"),
            ],
            SPORT_VISION_DIR,
        )
    if request.engine_id == "sports2d":
        return (
            [
                _python_for(request.engine_id), "Sports2D/Sports2D.py",
                "--video_input", str(video), "--result_dir", str(output_dir),
                "--show_graphs", "False",
            ],
            WORKSPACE_DIR / "Sports2D-main",
        )

    raise HTTPException(
        status_code=422,
        detail="This engine is a data, training, match-intelligence or interactive service and has no video-job adapter.",
    )


class JobManager:
    """One-at-a-time ML job runner with bounded, inspectable logs."""

    def __init__(self) -> None:
        self._store = JobStore(STATE_DIR / "jobs.sqlite3")
        self._jobs = self._store.load()
        for job in self._jobs.values():
            if job.get("status") in {"queued", "running"}:
                job.update(
                    status="interrupted",
                    finished_at=_now(),
                    error="Server restarted before this analysis could finish",
                )
                self._store.save(job)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sport-engine")
        self._processes: dict[str, subprocess.Popen] = {}
        self._futures: dict[str, Future] = {}

    def submit(self, request: JobRequest) -> dict:
        readiness = preflight(request.engine_id)
        if not readiness["ready"]:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Engine is not ready to run",
                    "missing_assets": readiness["missing_assets"],
                    "missing_packages": readiness["missing_packages"],
                },
            )
        job_id = uuid.uuid4().hex
        output_dir = OUTPUT_DIR / job_id
        command, working_dir = _command_for(request, output_dir, readiness["execution_mode"])
        job = {
            "id": job_id,
            "engine_id": request.engine_id,
            "execution_mode": readiness["execution_mode"],
            "status": "queued",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "output_dir": f"/api/jobs/{job_id}/files",
            "logs": [],
            "error": None,
        }
        with self._lock:
            active_jobs = sum(item["status"] in {"queued", "running"} for item in self._jobs.values())
            if active_jobs >= MAX_QUEUED_OR_RUNNING_JOBS:
                raise HTTPException(
                    status_code=429,
                    detail=f"Analysis queue is full (maximum {MAX_QUEUED_OR_RUNNING_JOBS} active or queued jobs)",
                )
            output_dir.mkdir(parents=True, exist_ok=False)
            self._jobs[job_id] = job
            self._store.save(job)
            self._futures[job_id] = self._executor.submit(self._run, job_id, command, working_dir)
        return self.public(job_id)

    def public(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            return {**job, "logs": list(job["logs"])}

    def summary(self) -> dict[str, int]:
        """Return aggregate local queue state without exposing job payloads."""

        with self._lock:
            counts: dict[str, int] = {}
            for job in self._jobs.values():
                status = str(job.get("status", "unknown"))
                counts[status] = counts.get(status, 0) + 1
            return {
                "queued_or_running": counts.get("queued", 0) + counts.get("running", 0),
                **dict(sorted(counts.items())),
            }

    def files(self, job_id: str) -> list[dict]:
        self.public(job_id)
        output_dir = OUTPUT_DIR / job_id
        if not output_dir.is_dir():
            return []
        return sorted([
            {"name": item.relative_to(output_dir).as_posix(), "size": item.stat().st_size}
            for item in output_dir.rglob("*") if item.is_file()
        ], key=lambda item: item["name"])

    def cancel(self, job_id: str) -> dict:
        """Cancel a queued job or terminate the subprocess of a running job."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            if job["status"] in {"complete", "failed", "cancelled"}:
                return {**job, "logs": list(job["logs"])}
            job.update(status="cancelled", finished_at=_now(), error="Cancelled by user")
            self._store.save(job)
            process = self._processes.get(job_id)
            future = self._futures.get(job_id)

        if process is not None and process.poll() is None:
            _terminate_process_tree(process)
        elif future is not None:
            future.cancel()
        return self.public(job_id)

    def shutdown(self) -> None:
        """Stop portal-owned engines before the FastAPI process exits.

        A child process cannot be safely resumed after a server shutdown. Mark
        its public record interrupted, then terminate only PIDs created by this
        manager. This prevents a hidden analysis process from surviving a
        development-server restart and writing into a stale result directory.
        """

        with self._lock:
            processes = list(self._processes.values())
            for job in self._jobs.values():
                if job.get("status") in {"queued", "running"}:
                    job.update(
                        status="interrupted",
                        finished_at=_now(),
                        error="Server shut down before this analysis could finish",
                    )
                    self._store.save(job)
        for process in processes:
            _terminate_process_tree(process)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str, command: list[str], working_dir: Path) -> None:
        with self._lock:
            if self._jobs[job_id]["status"] == "cancelled":
                return
            self._jobs[job_id].update(status="running", started_at=_now())
            self._store.save(self._jobs[job_id])
        timeout_timer: threading.Timer | None = None
        timed_out = threading.Event()
        try:
            popen_options: dict[str, object] = {}
            if os.name == "nt":
                popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            process = subprocess.Popen(
                command,
                cwd=working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                **popen_options,
            )
            with self._lock:
                interrupted = self._jobs[job_id]["status"] in {"cancelled", "interrupted"}
                if not interrupted:
                    self._processes[job_id] = process
            if interrupted:
                _terminate_process_tree(process)
                return
            else:
                def end_overdue_process() -> None:
                    timed_out.set()
                    _terminate_process_tree(process)

                timeout_timer = threading.Timer(JOB_TIMEOUT_SECONDS, end_overdue_process)
                timeout_timer.daemon = True
                timeout_timer.start()
            assert process.stdout is not None
            for line in process.stdout:
                self._append_log(job_id, line.rstrip())
            return_code = process.wait()
            if self.public(job_id)["status"] in {"cancelled", "interrupted"}:
                return
            if timed_out.is_set():
                self._update(
                    job_id, status="failed", finished_at=_now(),
                    error=f"Engine exceeded the {JOB_TIMEOUT_SECONDS}-second execution limit",
                )
                return
            if return_code:
                self._update(job_id, status="failed", finished_at=_now(), error=f"Engine exited with code {return_code}")
            else:
                output_dir = OUTPUT_DIR / job_id
                manifest = write_artifact_manifest(
                    output_dir,
                    run_id=job_id,
                    engine_id=self.public(job_id)["engine_id"],
                    execution_mode=self.public(job_id)["execution_mode"],
                )
                if manifest["artifact_count"] == 0:
                    self._update(job_id, status="failed", finished_at=_now(), error="Engine completed without output artifacts")
                elif any(
                    artifact.get("video", {}).get("playable") is False
                    for artifact in manifest["artifacts"]
                    if "video" in artifact
                ):
                    self._update(job_id, status="failed", finished_at=_now(), error="Engine produced an unreadable video artifact")
                else:
                    self._update(job_id, status="complete", finished_at=_now())
        except Exception as exc:  # report external-runtime failures as job data
            if self.public(job_id)["status"] not in {"cancelled", "interrupted"}:
                # Exception messages can contain a local source path, command
                # fragment or a third-party stack detail.  The browser needs a
                # stable actionable category, not host internals.
                self._update(
                    job_id,
                    status="failed",
                    finished_at=_now(),
                    error=_public_job_error(exc),
                )
        finally:
            if timeout_timer is not None:
                timeout_timer.cancel()
            with self._lock:
                self._processes.pop(job_id, None)

    def _update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)
            self._store.save(self._jobs[job_id])

    def _append_log(self, job_id: str, line: str) -> None:
        if not line:
            return
        line = _sanitize_log_line(line)
        if not line:
            return
        with self._lock:
            logs = self._jobs[job_id]["logs"]
            logs.append(line)
            del logs[:-120]
            self._store.save(self._jobs[job_id])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_log_line(line: str) -> str:
    """Bound engine output before it reaches a long-lived browser-visible log."""

    cleaned = ANSI_ESCAPE.sub("", line.replace("\x00", "")).strip()
    if len(cleaned) > MAX_LOG_LINE_CHARACTERS:
        return cleaned[:MAX_LOG_LINE_CHARACTERS] + " …[truncated]"
    return cleaned


def _public_job_error(error: Exception) -> str:
    """Classify an internal failure without leaking exception text to clients."""

    if isinstance(error, (OSError, subprocess.SubprocessError)):
        return "Engine process could not be started or monitored"
    if isinstance(error, (json.JSONDecodeError, ValueError, TypeError)):
        return "Engine returned invalid analysis output"
    return "Engine execution failed; inspect the local server console for details"


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Stop a portal-owned engine and its children without touching other PIDs."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            return
        except (OSError, subprocess.SubprocessError):
            pass
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


manager = JobManager()
