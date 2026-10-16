"""Explicit, conservative retention planning for local portal artifacts.

Jobs and uploads are user-generated local data.  The web server deliberately
does not delete them on a timer.  Instead, this module creates a dry-run plan
that only recognises portal-owned UUID-named upload files and job directories.
Applying a plan requires an explicit CLI ``--apply`` flag, so a stale cleanup
policy can never silently erase an experiment while it is being reviewed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence


SPORT_VISION_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = SPORT_VISION_DIR / "uploads"
OUTPUT_DIR = SPORT_VISION_DIR / "outputs"
PORTAL_ID = re.compile(r"^[a-f0-9]{32}$")
UPLOAD_NAME = re.compile(r"^(?P<id>[a-f0-9]{32})\.(?:json|mp4|avi|mov|webm|mkv|jpg|jpeg|png|webp)$")


@dataclass(frozen=True, slots=True)
class CleanupCandidate:
    """One portal-owned child targeted by an explicit cleanup plan."""

    root: Path
    relative_path: str
    kind: str
    modified_at: datetime

    @property
    def path(self) -> Path:
        return self.root / self.relative_path

    def as_dict(self) -> dict[str, str]:
        return {
            "root": self.root.name,
            "path": self.relative_path,
            "kind": self.kind,
            "modified_at": self.modified_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    """A reviewable set of candidates; it has no side effect until applied."""

    cutoff: datetime
    candidates: tuple[CleanupCandidate, ...]

    def as_dict(self) -> dict:
        return {
            "mode": "dry_run",
            "cutoff": self.cutoff.isoformat(),
            "candidate_count": len(self.candidates),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "safety": [
                "Only UUID-named portal uploads and job-output directories are included.",
                "Unknown files, symlinks and state databases are never candidates.",
                "No deletion occurs unless the caller explicitly applies this plan.",
            ],
        }


def _modified_at(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _upload_groups(root: Path) -> Iterable[tuple[str, list[Path]]]:
    groups: dict[str, list[Path]] = {}
    if not root.is_dir():
        return ()
    for child in root.iterdir():
        if child.is_symlink() or not child.is_file():
            continue
        match = UPLOAD_NAME.fullmatch(child.name)
        if match is None:
            continue
        groups.setdefault(match.group("id"), []).append(child)
    return ((upload_id, paths) for upload_id, paths in sorted(groups.items()))


def plan_upload_cleanup(root: Path, cutoff: datetime) -> tuple[CleanupCandidate, ...]:
    """Plan removal of whole, old UUID upload groups while retaining unknowns."""

    candidates: list[CleanupCandidate] = []
    for _, paths in _upload_groups(root):
        # Keep the full upload group if *any* associated media/manifest file is
        # newer than the cutoff.  This prevents a stale manifest from removing
        # a recently copied video (or vice versa).
        modified = [_modified_at(path) for path in paths]
        if not modified or max(modified) >= cutoff:
            continue
        for path, timestamp in sorted(zip(paths, modified), key=lambda pair: pair[0].name):
            candidates.append(CleanupCandidate(root.resolve(), path.name, "file", timestamp))
    return tuple(candidates)


def plan_output_cleanup(root: Path, cutoff: datetime) -> tuple[CleanupCandidate, ...]:
    """Plan removal of old UUID-named job output directories only."""

    if not root.is_dir():
        return ()
    candidates: list[CleanupCandidate] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.is_symlink() or not child.is_dir() or not PORTAL_ID.fullmatch(child.name):
            continue
        modified = _modified_at(child)
        if modified < cutoff:
            candidates.append(CleanupCandidate(root.resolve(), child.name, "directory", modified))
    return tuple(candidates)


def plan_portal_cleanup(
    *,
    older_than_hours: float = 168,
    now: datetime | None = None,
    upload_dir: Path = UPLOAD_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> CleanupPlan:
    """Build, but do not apply, a safe retention plan for portal-owned data."""

    if isinstance(older_than_hours, bool) or not isinstance(older_than_hours, (int, float)) or older_than_hours <= 0:
        raise ValueError("older_than_hours must be a positive number")
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    cutoff = reference.astimezone(timezone.utc) - timedelta(hours=float(older_than_hours))
    candidates = (*plan_upload_cleanup(upload_dir, cutoff), *plan_output_cleanup(output_dir, cutoff))
    return CleanupPlan(cutoff=cutoff, candidates=tuple(sorted(candidates, key=lambda item: (str(item.root), item.relative_path))))


def apply_cleanup_plan(plan: CleanupPlan) -> dict:
    """Apply an already reviewed plan with path and symlink checks repeated."""

    removed: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for candidate in plan.candidates:
        root = candidate.root.resolve()
        target = candidate.path
        try:
            resolved = target.resolve()
        except OSError:
            skipped.append({"path": candidate.relative_path, "reason": "unresolvable"})
            continue
        if not _inside(root, resolved) or target.is_symlink():
            skipped.append({"path": candidate.relative_path, "reason": "unsafe_path"})
            continue
        if not target.exists():
            skipped.append({"path": candidate.relative_path, "reason": "already_missing"})
            continue
        if candidate.kind == "file" and target.is_file():
            target.unlink()
        elif candidate.kind == "directory" and target.is_dir():
            shutil.rmtree(target)
        else:
            skipped.append({"path": candidate.relative_path, "reason": "type_changed"})
            continue
        removed.append(candidate.as_dict())
    return {
        "mode": "applied",
        "removed_count": len(removed),
        "removed": removed,
        "skipped": skipped,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Offer a local maintenance CLI; dry-run is the unconditional default."""

    parser = argparse.ArgumentParser(description="Plan conservative retention cleanup for Sport Vision local data")
    parser.add_argument("--older-than-hours", type=float, default=168)
    parser.add_argument("--apply", action="store_true", help="actually remove the dry-run candidates")
    args = parser.parse_args(argv)
    plan = plan_portal_cleanup(older_than_hours=args.older_than_hours)
    payload = apply_cleanup_plan(plan) if args.apply else plan.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CleanupCandidate",
    "CleanupPlan",
    "OUTPUT_DIR",
    "UPLOAD_DIR",
    "apply_cleanup_plan",
    "plan_output_cleanup",
    "plan_portal_cleanup",
    "plan_upload_cleanup",
]
