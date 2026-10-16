"""Lightweight, deterministic multi-person tracking for pose detections.

``TrackManager`` associates per-frame person detections without importing a
detector, a re-identification model, or OpenCV.  It is intended to sit between
any pose detector and sport-specific analysis code:

.. code-block:: python

    tracker = TrackManager()
    tracks = tracker.update([
        {"center": [480, 310], "scale": 210, "confidence": 0.94,
         "keypoints": [{"x": 470, "y": 220}, {"x": 492, "y": 220}]},
    ])

``update`` returns every live track, including recently missed tracks.  A
missed track has ``occluded=True``, ``detection_index=None`` and a monotonically
increasing ``missed`` count.  A visible track has a stable ``track_id`` and the
association cost used for its current-frame match.  ``association_cost`` is
``None`` for a newly born track or an occluded track because no match occurred.

Association is based on predicted centre distance normalized by body scale,
log-scale change, optional pose-keypoint displacement, and detector confidence.
It uses SciPy's Hungarian solver when present; if SciPy is absent or fails, a
stable cost/track-id/detection-index greedy solver is used instead.  The latter
is deliberately deterministic rather than globally optimal.  This is a
tracking primitive, not a substitute for appearance-based re-identification:
two players crossing at the same location can still need team/jersey features
for competition-grade identity preservation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, log
from typing import Any, Callable, Mapping, Sequence


class TrackingError(ValueError):
    """Base error for invalid tracker configuration or detections."""


class TrackingInputError(TrackingError):
    """Raised when a detection is not a finite center/scale/confidence record."""


@dataclass(frozen=True, slots=True)
class Detection:
    """Normalized person detection accepted by :class:`TrackManager`.

    ``center`` is in image pixels (or another consistent 2-D coordinate
    system).  ``scale`` is a positive person-size proxy, normally bounding-box
    height or shoulder width in the same units.  ``keypoints`` is kept opaque
    for callers, but mappings with ``x``/``y`` points and sequences of
    ``[x, y]`` points receive an optional association bonus.
    """

    center: tuple[float, float]
    scale: float = 1.0
    confidence: float = 1.0
    keypoints: Any = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Detection":
        if not isinstance(value, Mapping):
            raise TrackingInputError("a detection must be a mapping or Detection")
        if "center" not in value:
            raise TrackingInputError("detection.center is required")
        return cls(
            center=_point(value["center"], "detection.center"),
            scale=_finite(value.get("scale", 1.0), "detection.scale", positive=True),
            confidence=_finite(value.get("confidence", 1.0), "detection.confidence"),
            keypoints=value.get("keypoints"),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _point(self.center, "detection.center"))
        object.__setattr__(self, "scale", _finite(self.scale, "detection.scale", positive=True))
        confidence = _finite(self.confidence, "detection.confidence")
        if not 0.0 <= confidence <= 1.0:
            raise TrackingInputError("detection.confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Quality gates and relative costs for center/scale/pose association.

    Costs are dimensionless.  A centre distance of one body ``scale`` is 1.0
    before weighting.  ``max_center_distance_scales`` rejects implausible
    jumps before assignment, while ``max_association_cost`` gates the blended
    match score.  Tune both from labelled match footage rather than assuming
    these generic defaults are athlete-identity ground truth.
    """

    max_missed: int = 15
    max_association_cost: float = 2.5
    max_center_distance_scales: float = 2.75
    center_weight: float = 0.80
    scale_weight: float = 0.12
    keypoint_weight: float = 0.05
    confidence_weight: float = 0.03
    velocity_smoothing: float = 0.65
    min_keypoint_overlap: int = 3
    min_scale: float = 1.0e-6
    max_detections: int = 512

    def __post_init__(self) -> None:
        if isinstance(self.max_missed, bool) or not isinstance(self.max_missed, int) or self.max_missed < 0:
            raise TrackingInputError("max_missed must be a non-negative integer")
        if isinstance(self.min_keypoint_overlap, bool) or not isinstance(self.min_keypoint_overlap, int) or self.min_keypoint_overlap < 1:
            raise TrackingInputError("min_keypoint_overlap must be a positive integer")
        if isinstance(self.max_detections, bool) or not isinstance(self.max_detections, int) or self.max_detections < 1:
            raise TrackingInputError("max_detections must be a positive integer")
        for name in (
            "max_association_cost",
            "max_center_distance_scales",
            "center_weight",
            "scale_weight",
            "keypoint_weight",
            "confidence_weight",
            "min_scale",
        ):
            if _finite(getattr(self, name), name, positive=True) <= 0:
                raise TrackingInputError(f"{name} must be positive")
        smoothing = _finite(self.velocity_smoothing, "velocity_smoothing")
        if not 0.0 <= smoothing <= 1.0:
            raise TrackingInputError("velocity_smoothing must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TrackResult:
    """A JSON-friendly live-track snapshot produced by ``update``."""

    track_id: int
    detection_index: int | None
    center: tuple[float, float]
    scale: float
    confidence: float
    keypoints: Any
    matched: bool
    is_new: bool
    occluded: bool
    missed: int
    association_cost: float | None
    age: int
    hits: int
    association_solver: str

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-ready track record without host state."""

        return {
            "track_id": self.track_id,
            "detection_index": self.detection_index,
            "center": [self.center[0], self.center[1]],
            "scale": self.scale,
            "confidence": self.confidence,
            "keypoints": self.keypoints,
            "matched": self.matched,
            "is_new": self.is_new,
            "occluded": self.occluded,
            "missed": self.missed,
            "association_cost": self.association_cost,
            "age": self.age,
            "hits": self.hits,
            "association_solver": self.association_solver,
        }


@dataclass(slots=True)
class _Track:
    track_id: int
    center: tuple[float, float]
    scale: float
    confidence: float
    keypoints: Any
    velocity: tuple[float, float]
    last_frame: int
    last_observed_frame: int
    last_observed_center: tuple[float, float]
    age: int = 1
    hits: int = 1
    missed: int = 0
    association_cost: float | None = None


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackingInputError(f"{name} must be a finite number")
    number = float(value)
    if not isfinite(number) or (positive and number <= 0.0):
        qualifier = "a positive finite number" if positive else "a finite number"
        raise TrackingInputError(f"{name} must be {qualifier}")
    return number


def _point(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, Mapping):
        try:
            return (_finite(value["x"], f"{name}.x"), _finite(value["y"], f"{name}.y"))
        except KeyError as error:
            raise TrackingInputError(f"{name} mapping needs x and y") from error
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise TrackingInputError(f"{name} must be [x, y] or {{'x', 'y'}}")
    return (_finite(value[0], f"{name}[0]"), _finite(value[1], f"{name}[1]"))


def _keypoint_map(value: Any) -> dict[str, tuple[float, float]]:
    """Best-effort finite keypoint extraction; malformed points are ignored.

    Keypoints are optional metadata, so one malformed landmark must not throw
    away an otherwise valid person detection.
    """

    if value is None or isinstance(value, (str, bytes)):
        return {}
    items = value.items() if isinstance(value, Mapping) else enumerate(value) if isinstance(value, Sequence) else ()
    result: dict[str, tuple[float, float]] = {}
    for raw_name, raw_point in items:
        try:
            result[str(raw_name)] = _point(raw_point, f"keypoints[{raw_name!r}]")
        except TrackingInputError:
            continue
    return result


def _canonical_detection_order(detections: Sequence[Detection]) -> list[int]:
    """Make births and equal-cost ties independent of detector output order."""

    return sorted(
        range(len(detections)),
        key=lambda index: (
            detections[index].center[0],
            detections[index].center[1],
            -detections[index].confidence,
            detections[index].scale,
            index,
        ),
    )


def _load_hungarian_solver() -> Callable[[Any], tuple[Sequence[int], Sequence[int]]] | None:
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment
    except Exception:
        # SciPy is intentionally optional: deployments using only the standard
        # library still retain predictable (though non-optimal) associations.
        return None


class TrackManager:
    """Associate person detections across video frames with stable IDs.

    Call :meth:`update` once for each monotonically increasing frame.  The
    returned list is sorted by ``track_id`` and includes visible and occluded
    live tracks.  A track is retired only after it has been missed for more
    than ``config.max_missed`` frame intervals.
    """

    def __init__(self, config: TrackingConfig | None = None, *, prefer_hungarian: bool = True) -> None:
        self.config = config or TrackingConfig()
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._frame_index = -1
        self._hungarian_solver = _load_hungarian_solver() if prefer_hungarian else None
        self.association_solver = "hungarian" if self._hungarian_solver is not None else "greedy"

    @property
    def frame_index(self) -> int:
        """Most recently processed frame index, or ``-1`` before first update."""

        return self._frame_index

    @property
    def track_count(self) -> int:
        """Number of live tracks, including short-term occlusions."""

        return len(self._tracks)

    def reset(self) -> None:
        """Forget all live tracks and restart IDs from one."""

        self._tracks.clear()
        self._next_track_id = 1
        self._frame_index = -1

    def update(
        self,
        detections: Sequence[Detection | Mapping[str, Any]],
        *,
        frame_index: int | None = None,
    ) -> list[TrackResult]:
        """Associate a frame's detections and return all live track snapshots.

        ``frame_index`` is optional; when omitted the manager increments it by
        one.  Explicit indices must strictly increase, and gaps count as missed
        frame intervals for invisible tracks.  New IDs are allocated in a
        canonical spatial order so detector ordering alone does not scramble
        first-frame IDs.
        """

        if isinstance(detections, (str, bytes)) or not isinstance(detections, Sequence):
            raise TrackingInputError("detections must be a sequence")
        if len(detections) > self.config.max_detections:
            raise TrackingInputError(f"too many detections (maximum {self.config.max_detections})")
        normalized = [item if isinstance(item, Detection) else Detection.from_mapping(item) for item in detections]
        current_frame = self._next_frame_index(frame_index)
        frame_delta = current_frame - self._frame_index if self._frame_index >= 0 else 1
        self._frame_index = current_frame

        candidates = [self._tracks[track_id] for track_id in sorted(self._tracks)]
        predictions = {track.track_id: self._predict(track, frame_delta) for track in candidates}
        matches = self._associate(candidates, predictions, normalized)
        matched_track_ids = {track_id for track_id, _, _ in matches}
        matched_detection_indices = {detection_index for _, detection_index, _ in matches}
        results: list[TrackResult] = []

        for track_id, detection_index, cost in matches:
            track = self._tracks[track_id]
            detection = normalized[detection_index]
            self._observe(track, detection, current_frame, predictions[track_id], frame_delta, cost)
            results.append(self._result(track, detection_index, matched=True, is_new=False))

        for track in candidates:
            if track.track_id in matched_track_ids:
                continue
            predicted = predictions[track.track_id]
            track.center = predicted
            track.last_frame = current_frame
            track.age += frame_delta
            track.missed += frame_delta
            track.association_cost = None

        # Keep occluded results only while the configured recovery window is open.
        retired = [track_id for track_id, track in self._tracks.items() if track.missed > self.config.max_missed]
        for track_id in retired:
            del self._tracks[track_id]
        for track in candidates:
            if track.track_id in self._tracks and track.track_id not in matched_track_ids:
                results.append(self._result(track, None, matched=False, is_new=False))

        for detection_index in _canonical_detection_order(normalized):
            if detection_index in matched_detection_indices:
                continue
            track = self._new_track(normalized[detection_index], current_frame)
            self._tracks[track.track_id] = track
            results.append(self._result(track, detection_index, matched=True, is_new=True))

        return sorted(results, key=lambda result: result.track_id)

    def snapshot(self) -> list[TrackResult]:
        """Return current live tracks without advancing a video frame."""

        return [
            self._result(track, None, matched=False, is_new=False)
            for _, track in sorted(self._tracks.items())
        ]

    def _next_frame_index(self, requested: int | None) -> int:
        if requested is None:
            return self._frame_index + 1
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
            raise TrackingInputError("frame_index must be a non-negative integer")
        if requested <= self._frame_index:
            raise TrackingInputError("frame_index must strictly increase")
        return requested

    def _predict(self, track: _Track, frame_delta: int) -> tuple[float, float]:
        return (
            track.center[0] + track.velocity[0] * frame_delta,
            track.center[1] + track.velocity[1] * frame_delta,
        )

    def _association_cost(
        self,
        track: _Track,
        predicted_center: tuple[float, float],
        detection: Detection,
    ) -> float | None:
        body_scale = max(track.scale, detection.scale, self.config.min_scale)
        center_distance = hypot(
            detection.center[0] - predicted_center[0],
            detection.center[1] - predicted_center[1],
        ) / body_scale
        if center_distance > self.config.max_center_distance_scales:
            return None

        scale_cost = abs(log(max(detection.scale, self.config.min_scale) / max(track.scale, self.config.min_scale)))
        keypoint_cost = self._keypoint_cost(track, predicted_center, detection, body_scale)
        confidence_cost = 1.0 - min(track.confidence, detection.confidence)
        terms: list[tuple[float, float]] = [
            (center_distance, self.config.center_weight),
            (scale_cost, self.config.scale_weight),
            (confidence_cost, self.config.confidence_weight),
        ]
        if keypoint_cost is not None:
            terms.append((keypoint_cost, self.config.keypoint_weight))
        total_weight = sum(weight for _, weight in terms)
        cost = sum(value * weight for value, weight in terms) / total_weight
        return cost if cost <= self.config.max_association_cost else None

    def _keypoint_cost(
        self,
        track: _Track,
        predicted_center: tuple[float, float],
        detection: Detection,
        body_scale: float,
    ) -> float | None:
        previous = _keypoint_map(track.keypoints)
        current = _keypoint_map(detection.keypoints)
        common = sorted(set(previous) & set(current))
        if len(common) < self.config.min_keypoint_overlap:
            return None
        # Shift old keypoints by the track's predicted centre movement so
        # keypoints reward pose consistency rather than duplicate centre motion.
        dx = predicted_center[0] - track.center[0]
        dy = predicted_center[1] - track.center[1]
        distances = [
            hypot((previous[name][0] + dx) - current[name][0], (previous[name][1] + dy) - current[name][1]) / body_scale
            for name in common
        ]
        return sum(distances) / len(distances)

    def _associate(
        self,
        tracks: Sequence[_Track],
        predictions: Mapping[int, tuple[float, float]],
        detections: Sequence[Detection],
    ) -> list[tuple[int, int, float]]:
        if not tracks or not detections:
            return []
        detection_order = _canonical_detection_order(detections)
        costs: list[list[float | None]] = [
            [self._association_cost(track, predictions[track.track_id], detections[index]) for index in detection_order]
            for track in tracks
        ]
        if self._hungarian_solver is not None:
            try:
                blocked = self.config.max_association_cost + 1_000_000.0
                rows, columns = self._hungarian_solver([[blocked if cost is None else cost for cost in row] for row in costs])
                pairs = [(int(row), int(column)) for row, column in zip(rows, columns)]
            except Exception:
                # A malformed/partial SciPy install must not stop a live job.
                self._hungarian_solver = None
                self.association_solver = "greedy"
                pairs = self._greedy_pairs(tracks, detection_order, costs)
        else:
            pairs = self._greedy_pairs(tracks, detection_order, costs)

        matches: list[tuple[int, int, float]] = []
        for row, column in pairs:
            if row < 0 or row >= len(tracks) or column < 0 or column >= len(detection_order):
                continue
            cost = costs[row][column]
            if cost is not None:
                matches.append((tracks[row].track_id, detection_order[column], cost))
        return sorted(matches, key=lambda item: item[0])

    @staticmethod
    def _greedy_pairs(
        tracks: Sequence[_Track],
        detection_order: Sequence[int],
        costs: Sequence[Sequence[float | None]],
    ) -> list[tuple[int, int]]:
        candidates = [
            (cost, tracks[row].track_id, detection_order[column], row, column)
            for row, row_costs in enumerate(costs)
            for column, cost in enumerate(row_costs)
            if cost is not None
        ]
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for _, track_id, detection_index, row, column in sorted(candidates):
            if track_id in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(detection_index)
            pairs.append((row, column))
        return pairs

    def _observe(
        self,
        track: _Track,
        detection: Detection,
        current_frame: int,
        predicted_center: tuple[float, float],
        frame_delta: int,
        cost: float,
    ) -> None:
        observed_delta = max(1, current_frame - track.last_observed_frame)
        measured_velocity = (
            (detection.center[0] - track.last_observed_center[0]) / observed_delta,
            (detection.center[1] - track.last_observed_center[1]) / observed_delta,
        )
        alpha = self.config.velocity_smoothing
        track.velocity = (
            alpha * measured_velocity[0] + (1.0 - alpha) * track.velocity[0],
            alpha * measured_velocity[1] + (1.0 - alpha) * track.velocity[1],
        )
        track.center = detection.center
        track.scale = detection.scale
        track.confidence = detection.confidence
        track.keypoints = detection.keypoints
        track.last_frame = current_frame
        track.last_observed_frame = current_frame
        track.last_observed_center = detection.center
        track.age += frame_delta
        track.hits += 1
        track.missed = 0
        track.association_cost = cost

    def _new_track(self, detection: Detection, frame_index: int) -> _Track:
        track = _Track(
            track_id=self._next_track_id,
            center=detection.center,
            scale=detection.scale,
            confidence=detection.confidence,
            keypoints=detection.keypoints,
            velocity=(0.0, 0.0),
            last_frame=frame_index,
            last_observed_frame=frame_index,
            last_observed_center=detection.center,
        )
        self._next_track_id += 1
        return track

    def _result(
        self,
        track: _Track,
        detection_index: int | None,
        *,
        matched: bool,
        is_new: bool,
    ) -> TrackResult:
        return TrackResult(
            track_id=track.track_id,
            detection_index=detection_index,
            center=track.center,
            scale=track.scale,
            confidence=track.confidence,
            keypoints=track.keypoints,
            matched=matched,
            is_new=is_new,
            occluded=not matched,
            missed=track.missed,
            association_cost=track.association_cost if matched and not is_new else None,
            age=track.age,
            hits=track.hits,
            association_solver=self.association_solver,
        )


__all__ = [
    "Detection",
    "TrackManager",
    "TrackResult",
    "TrackingConfig",
    "TrackingError",
    "TrackingInputError",
]
