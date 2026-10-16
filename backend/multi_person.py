"""Explicit multi-person pose candidates and primary-athlete selection.

The existing :class:`backend.pose_analyzer.PoseAnalyzer` remains the legacy
single-athlete API.  ``MultiPersonPoseAnalyzer`` is a separate, opt-in layer
that asks MediaPipe for more than one pose, turns each detector result into a
tracking candidate, and associates them through :class:`backend.tracking.TrackManager`.

It deliberately never guesses the "main" athlete.  A caller first inspects
the returned stable ``track_id`` values, then supplies the chosen
``primary_track_id`` to a subsequent ``process_frame`` call.  This keeps a
multi-player sports analysis from silently switching to the nearest or largest
person when the camera composition changes.

Track IDs are short-term geometric associations, not re-identification proof.
Crossing players, long occlusions, and identical-looking athletes need a
validated appearance/re-ID component before identity-sensitive competition
claims are appropriate.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

import numpy as np

from .tracking import TrackManager, TrackingInputError, TrackResult


class PrimaryTrackSelectionError(ValueError):
    """Raised when a requested primary ID is not a live track."""


class CandidateDetector(Protocol):
    """Minimal adapter implemented by ``PoseAnalyzer`` and test doubles."""

    def detect_pose_candidates(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return raw per-frame person candidates."""

    def get_multi_pose_capabilities(self) -> dict[str, Any]:
        """Return configured and observed detector capability metadata."""


class MultiPersonPoseAnalyzer:
    """Track MediaPipe pose candidates without implicitly selecting a person.

    Parameters
    ----------
    num_poses:
        The maximum pose-candidate request passed to a default
        :class:`PoseAnalyzer`.  It is a configuration limit, not a guarantee
        that the model will find that many people in any given frame.
    detector:
        Optional ``PoseAnalyzer``-compatible detector.  Injection makes this
        layer testable without model assets and permits a future detector
        replacement, provided it keeps the candidate contract.
    tracker:
        Optional configured :class:`TrackManager`.  It is intentionally
        separate from MediaPipe's internal temporal pose tracking so that the
        caller can see and audit stable application-level IDs.
    """

    def __init__(
        self,
        *,
        num_poses: int = 4,
        detector: CandidateDetector | None = None,
        tracker: TrackManager | None = None,
    ) -> None:
        self._owns_detector = detector is None
        if detector is None:
            # Lazy import keeps pure tracking/candidate tests independent of
            # model file availability and avoids importing MediaPipe merely to
            # inspect this module.
            from .pose_analyzer import PoseAnalyzer

            detector = PoseAnalyzer(num_poses=num_poses)
        self.detector = detector
        self.tracker = tracker or TrackManager()

    def process_frame(
        self,
        frame_rgb: np.ndarray,
        *,
        timestamp_ms: Optional[int] = None,
        frame_index: Optional[int] = None,
        primary_track_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """Extract, associate and expose every person in one video frame.

        ``primary_track_id`` is intentionally an explicit caller choice.  If
        it is omitted, ``primary`` remains ``None`` even when one candidate is
        visually dominant.  If the ID is retained through a short occlusion,
        ``primary`` includes its predicted track state with ``visible=False``;
        it is never substituted with another player.

        Returns a JSON-ready object with detector ``candidates``, all live
        ``tracks``, capability metadata, and (only when requested) the chosen
        ``primary`` track/candidate pair.
        """

        raw_candidates = self.detector.detect_pose_candidates(frame_rgb, timestamp_ms)
        if not isinstance(raw_candidates, list):
            raise TrackingInputError("detector.detect_pose_candidates must return a list")

        # Pass only the stable tracking contract through the tracker.  Keep the
        # richer detector candidate untouched for client inspection/rendering.
        tracking_inputs = [
            {
                "center": candidate.get("center") if isinstance(candidate, dict) else None,
                "scale": candidate.get("scale") if isinstance(candidate, dict) else None,
                "confidence": candidate.get("confidence") if isinstance(candidate, dict) else None,
                "keypoints": candidate.get("keypoints") if isinstance(candidate, dict) else None,
            }
            for candidate in raw_candidates
        ]
        tracks = self.tracker.update(tracking_inputs, frame_index=frame_index)
        candidate_by_detection_index = {
            index: candidate for index, candidate in enumerate(raw_candidates)
        }
        track_by_detection_index = {
            track.detection_index: track
            for track in tracks
            if track.matched and track.detection_index is not None
        }

        candidates: list[dict[str, Any]] = []
        for detection_index, candidate in enumerate(raw_candidates):
            if not isinstance(candidate, dict):
                # TrackManager already emits a precise contract error for
                # invalid data.  This guard protects type-checker-unaware
                # injected detectors from an accidental mutation here.
                raise TrackingInputError("each detector candidate must be a dictionary")
            track = track_by_detection_index.get(detection_index)
            candidates.append({
                **candidate,
                "detection_index": detection_index,
                "track_id": track.track_id if track is not None else None,
                "track_status": self._track_status(track) if track is not None else "unmatched",
            })

        primary = self._resolve_primary(
            primary_track_id,
            tracks,
            candidate_by_detection_index,
            candidates,
        )
        capabilities = self._capabilities()
        return {
            "frame_index": self.tracker.frame_index,
            "candidates": candidates,
            "tracks": [track.as_dict() for track in tracks],
            "primary_track_id": primary_track_id,
            "primary": primary,
            "selection_required": bool(tracks) and primary_track_id is None,
            "capabilities": capabilities,
        }

    def reset(self) -> None:
        """Clear application track state without changing detector settings."""

        self.tracker.reset()
        reset = getattr(self.detector, "reset", None)
        if callable(reset):
            reset()

    def close(self) -> None:
        """Close only a detector this instance created itself."""

        if not self._owns_detector:
            return
        close = getattr(self.detector, "close", None)
        if callable(close):
            close()

    def _resolve_primary(
        self,
        primary_track_id: Optional[int],
        tracks: Sequence[TrackResult],
        candidate_by_detection_index: dict[int, dict[str, Any]],
        candidates: Sequence[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if primary_track_id is None:
            return None
        if isinstance(primary_track_id, bool) or not isinstance(primary_track_id, int) or primary_track_id < 1:
            raise PrimaryTrackSelectionError("primary_track_id must be a positive integer")

        track = next((item for item in tracks if item.track_id == primary_track_id), None)
        if track is None:
            live_ids = [item.track_id for item in tracks]
            raise PrimaryTrackSelectionError(
                f"primary_track_id {primary_track_id} is not live; available track IDs: {live_ids}"
            )

        candidate: Optional[dict[str, Any]] = None
        if track.detection_index is not None:
            # Return the enriched public candidate rather than a detached
            # detector object, keeping one canonical record for consumers.
            candidate = next(
                (
                    item
                    for item in candidates
                    if item.get("detection_index") == track.detection_index
                ),
                None,
            )
            if candidate is None:
                # Defensive fallback: association state is still useful even
                # if a custom detector handed us unusual candidate objects.
                candidate = candidate_by_detection_index.get(track.detection_index)

        return {
            "track": track.as_dict(),
            "candidate": candidate,
            "visible": bool(track.matched and candidate is not None),
        }

    def _capabilities(self) -> dict[str, Any]:
        get_capabilities = getattr(self.detector, "get_multi_pose_capabilities", None)
        if callable(get_capabilities):
            metadata = get_capabilities()
            if isinstance(metadata, dict):
                return metadata
        return {
            "configured_candidate_limit": None,
            "multi_person_configured": None,
            "limitations": [
                "The injected detector does not expose a pose-count capability contract.",
            ],
        }

    @staticmethod
    def _track_status(track: TrackResult) -> str:
        if track.is_new:
            return "new"
        if track.occluded:
            return "occluded"
        return "tracked"


__all__ = [
    "CandidateDetector",
    "MultiPersonPoseAnalyzer",
    "PrimaryTrackSelectionError",
]
