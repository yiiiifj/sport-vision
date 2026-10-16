"""Inspectable artifact manifests for reproducible local analysis runs."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

import cv2


MANIFEST_NAME = "artifact-manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_metadata(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"playable": False}
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, _ = capture.read()
        return {
            "playable": bool(ok),
            "fps": round(fps, 3),
            "frames": frames,
            "width": width,
            "height": height,
            "duration_s": round(frames / fps, 3) if fps else None,
        }
    finally:
        capture.release()


def write_artifact_manifest(output_dir: Path, *, run_id: str, engine_id: str, execution_mode: str) -> dict:
    """Hash output files and probe generated videos before exposing them."""

    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        entry = {
            "name": path.relative_to(output_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        }
        if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            entry["video"] = _video_metadata(path)
        artifacts.append(entry)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "engine_id": engine_id,
        "execution_mode": execution_mode,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
