"""Opaque upload records shared by HTTP, WebSocket and background jobs.

The browser is allowed to hold an opaque UUID ID only. File paths, file
types and validation state stay server-side so a template cannot be substituted
for a video and a client can never ask the process to read an arbitrary path.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import HTTPException


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
# UUID4 hex keeps upload IDs opaque and makes accidental replacement
# infeasible in a long-running local competition deployment.
UPLOAD_ID = re.compile(r"^[a-f0-9]{32}$")
ALLOWED_EXTENSIONS = {
    "video": frozenset({".mp4", ".avi", ".mov", ".webm", ".mkv"}),
    "template": frozenset({".jpg", ".jpeg", ".png", ".webp"}),
}


@dataclass(frozen=True)
class UploadRecord:
    id: str
    kind: str
    extension: str
    original_name: str
    size_bytes: int


def data_path(record: UploadRecord) -> Path:
    return UPLOAD_DIR / f"{record.id}{record.extension}"


def create_record(file_id: str, kind: str, extension: str, original_name: str, size_bytes: int) -> UploadRecord:
    if not UPLOAD_ID.fullmatch(file_id) or kind not in ALLOWED_EXTENSIONS or extension not in ALLOWED_EXTENSIONS[kind]:
        raise ValueError("Invalid upload metadata")
    return UploadRecord(file_id, kind, extension, Path(original_name).name, size_bytes)


def save_record(record: UploadRecord) -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)
    (UPLOAD_DIR / f"{record.id}.json").write_text(
        json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def resolve_upload(file_id: str, expected_kind: str) -> Path:
    """Return only an existing, validated upload of the expected kind."""

    if expected_kind not in ALLOWED_EXTENSIONS or not UPLOAD_ID.fullmatch(file_id):
        raise HTTPException(status_code=400, detail="Invalid upload identifier")
    manifest = UPLOAD_DIR / f"{file_id}.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        record = UploadRecord(**data)
    except (OSError, json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=404, detail="Uploaded file not found") from None
    if record.id != file_id or record.kind != expected_kind or record.extension not in ALLOWED_EXTENSIONS[expected_kind]:
        raise HTTPException(status_code=404, detail="Uploaded file not found")
    path = data_path(record).resolve()
    if UPLOAD_DIR.resolve() not in path.parents or not path.is_file() or path.stat().st_size != record.size_bytes:
        raise HTTPException(status_code=404, detail="Uploaded file not found")
    return path


def discard_record(record: UploadRecord) -> None:
    data_path(record).unlink(missing_ok=True)
    (UPLOAD_DIR / f"{record.id}.json").unlink(missing_ok=True)
