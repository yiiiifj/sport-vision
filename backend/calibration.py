"""Safe, planar court calibration utilities.

The helpers in this module deliberately have no web, file-system, or model
dependencies.  A request handler can pass decoded JSON points to
``calibrate_court_homography`` and serialize :meth:`CourtCalibration.as_dict`.

The estimated transform maps the *court plane* to image pixels.  Its inverse
maps image pixels back to court coordinates.  Metric output is deliberately
disabled unless callers explicitly provide a recognized length unit and the
RANSAC fit passes the configured validation gates.

Important limitation: a homography is only valid for points on the calibrated
plane (normally shoe/ball ground-contact positions).  It must not be used to
claim metre-accurate positions for airborne balls, heads, hands, or a player
whose contact point is not visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


MIN_CORRESPONDENCES = 4
"""Minimum number of image/court point pairs needed for a projective map."""

MAX_CORRESPONDENCES = 2_048
"""Bound request work before calling OpenCV's RANSAC implementation."""


class CalibrationError(ValueError):
    """Base error for malformed or non-estimable calibration input."""


class CalibrationInputError(CalibrationError):
    """Raised when supplied JSON-like point data is unsafe or incomplete."""


class CalibrationEstimationError(CalibrationError):
    """Raised when OpenCV cannot estimate a stable homography."""


class CalibrationQualityError(CalibrationError):
    """Raised when an operation explicitly requires a metric-ready result."""


# The transform is normalized to metres for recognized linear units.  Values
# with an unknown unit remain useful for drawing a court overlay, but are never
# advertised as metric measurements.
_METERS_PER_COURT_UNIT = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "centimetre": 0.01,
    "centimetres": 0.01,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "millimetre": 0.001,
    "millimetres": 0.001,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
}


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    """Return a finite JSON number, rejecting booleans and string coercion."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise CalibrationInputError(f"{name} must be a finite number")
    number = float(value)
    if not isfinite(number) or (positive and number <= 0):
        comparator = "a positive finite number" if positive else "a finite number"
        raise CalibrationInputError(f"{name} must be {comparator}")
    return number


@dataclass(frozen=True)
class CalibrationConfig:
    """Quality limits for a court-plane homography.

    ``ransac_reprojection_threshold_px`` and both reported reprojection RMSEs
    are in image pixels because RANSAC estimates the court-to-image transform.
    Keeping the tolerance in pixels avoids silently changing acceptance when a
    caller supplies court coordinates in centimetres instead of metres.
    """

    ransac_reprojection_threshold_px: float = 4.0
    max_reprojection_rmse_px: float = 5.0
    min_inliers: int = MIN_CORRESPONDENCES
    min_inlier_ratio: float = 0.75
    max_homography_condition_number: float = 1.0e12

    def __post_init__(self) -> None:
        _finite_number(
            self.ransac_reprojection_threshold_px,
            "ransac_reprojection_threshold_px",
            positive=True,
        )
        _finite_number(
            self.max_reprojection_rmse_px,
            "max_reprojection_rmse_px",
            positive=True,
        )
        _finite_number(
            self.max_homography_condition_number,
            "max_homography_condition_number",
            positive=True,
        )
        if isinstance(self.min_inliers, bool) or not isinstance(self.min_inliers, int):
            raise CalibrationInputError("min_inliers must be an integer")
        if self.min_inliers < MIN_CORRESPONDENCES:
            raise CalibrationInputError(f"min_inliers must be at least {MIN_CORRESPONDENCES}")
        ratio = _finite_number(self.min_inlier_ratio, "min_inlier_ratio")
        if not 0 < ratio <= 1:
            raise CalibrationInputError("min_inlier_ratio must be in (0, 1]")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "CalibrationConfig":
        """Build a config from safe, request-shaped data.

        Unknown keys are rejected so spelling mistakes cannot silently weaken a
        calibration quality gate.
        """

        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise CalibrationInputError("options must be an object")
        allowed = {
            "ransac_reprojection_threshold_px",
            "max_reprojection_rmse_px",
            "min_inliers",
            "min_inlier_ratio",
            "max_homography_condition_number",
        }
        unexpected = sorted(set(values) - allowed)
        if unexpected:
            raise CalibrationInputError(f"unsupported calibration option(s): {', '.join(map(str, unexpected))}")
        return cls(**dict(values))


def _coerce_points(
    points: Sequence[Sequence[float]] | np.ndarray,
    name: str,
    *,
    minimum: int = MIN_CORRESPONDENCES,
) -> np.ndarray:
    """Validate an ``N x 2`` finite coordinate list without lossy coercion."""

    if isinstance(points, (str, bytes)):
        raise CalibrationInputError(f"{name} must be an array of [x, y] points")
    try:
        raw = np.asarray(points)
    except (TypeError, ValueError) as error:
        raise CalibrationInputError(f"{name} must be an array of [x, y] points") from error
    if raw.dtype.kind in {"b", "O", "S", "U"}:
        raise CalibrationInputError(f"{name} coordinates must be JSON numbers, not strings, objects, or booleans")
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CalibrationInputError(f"{name} coordinates must be numeric") from error
    if array.ndim != 2 or array.shape[1] != 2:
        raise CalibrationInputError(f"{name} must have shape N x 2")
    count = array.shape[0]
    if count < minimum:
        raise CalibrationInputError(f"{name} needs at least {minimum} point{'s' if minimum != 1 else ''}")
    if count > MAX_CORRESPONDENCES:
        raise CalibrationInputError(f"{name} has too many points (maximum {MAX_CORRESPONDENCES})")
    if not np.isfinite(array).all():
        raise CalibrationInputError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(array)


def _ensure_non_degenerate(points: np.ndarray, name: str) -> None:
    """Reject duplicate or near-collinear geometry before calling OpenCV."""

    if len(np.unique(points, axis=0)) < MIN_CORRESPONDENCES:
        raise CalibrationInputError(f"{name} needs at least {MIN_CORRESPONDENCES} distinct points")

    centered = points - points.mean(axis=0, keepdims=True)
    extent = float(np.max(np.ptp(points, axis=0)))
    tolerance = max(1.0e-12, extent * 1.0e-9)
    if np.linalg.matrix_rank(centered, tol=tolerance) < 2:
        raise CalibrationInputError(f"{name} points are collinear and cannot define a court plane")

    hull = cv2.convexHull(points.astype(np.float32))
    hull_area = abs(float(cv2.contourArea(hull)))
    # A thin sliver of points produces a wildly unstable transform even if it
    # technically has rank 2.  The threshold scales with the supplied layout.
    if hull_area <= max(1.0e-12, extent * extent * 1.0e-7):
        raise CalibrationInputError(f"{name} points cover too little area for a stable homography")


def _normalise_homography(homography: np.ndarray) -> np.ndarray:
    """Check and normalize a 3x3 homography for stable serialization/use."""

    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise CalibrationEstimationError("OpenCV returned an invalid homography")
    scale = float(matrix[2, 2])
    if abs(scale) < 1.0e-15:
        scale = float(np.linalg.norm(matrix))
    if not isfinite(scale) or abs(scale) < 1.0e-15:
        raise CalibrationEstimationError("homography has no usable scale")
    matrix = matrix / scale
    if np.linalg.matrix_rank(matrix) < 3:
        raise CalibrationEstimationError("homography is singular")
    return matrix


def _project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Project points and turn OpenCV/numerical failures into a clear error."""

    try:
        projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    except cv2.error as error:
        raise CalibrationEstimationError("could not project calibration points") from error
    if not np.isfinite(projected).all():
        raise CalibrationEstimationError("homography projects points to infinity")
    return projected.astype(np.float64, copy=False)


def _unit_scale(court_unit: str | None) -> tuple[str, float | None]:
    """Normalize a unit label and return its conversion to metres if known."""

    if court_unit is None:
        return "unknown", None
    if not isinstance(court_unit, str):
        raise CalibrationInputError("court_unit must be a string such as 'm' or 'cm'")
    normalized = court_unit.strip().lower()
    if not normalized:
        return "unknown", None
    return normalized, _METERS_PER_COURT_UNIT.get(normalized)


@dataclass(frozen=True)
class CourtCalibration:
    """An inspectable, planar court calibration result.

    ``court_to_image_homography`` maps normalized court coordinates (metres
    when ``metric_enabled`` is true) to image pixels.  The inverse maps pixels
    to those court coordinates.  The transform itself can be valid while
    ``metric_enabled`` remains false; consumers must honor that gate before
    labelling distances, speeds, or areas as physical measurements.
    """

    court_to_image_homography: np.ndarray
    image_to_court_homography: np.ndarray
    inlier_mask: np.ndarray
    point_count: int
    inlier_count: int
    inlier_ratio: float
    reprojection_rmse_px: float
    reprojection_rmse_all_px: float
    homography_condition_number: float
    input_court_unit: str
    coordinate_unit: str
    homography_valid: bool
    quality_passed: bool
    metric_enabled: bool
    metric_disabled_reasons: tuple[str, ...] = field(default_factory=tuple)
    metric_scope: str = "court_plane_only"

    @property
    def status(self) -> str:
        """Small stable status for request responses and UI badges."""

        if self.metric_enabled:
            return "metric_ready"
        if self.quality_passed:
            return "homography_only"
        return "quality_rejected"

    def require_metric(self) -> "CourtCalibration":
        """Return self or make accidental use of untrusted metres explicit."""

        if not self.metric_enabled:
            reasons = ", ".join(self.metric_disabled_reasons) or "metric validation did not pass"
            raise CalibrationQualityError(f"metric court coordinates are disabled: {reasons}")
        return self

    def image_to_court(self, image_points: Sequence[Sequence[float]] | np.ndarray, *, require_metric: bool = False) -> np.ndarray:
        """Map image points to the calibrated court plane.

        Set ``require_metric=True`` for distance/speed calculations.  By
        default this remains useful for a non-metric overlay coordinate system.
        """

        if require_metric:
            self.require_metric()
        points = _coerce_points(image_points, "image_points", minimum=1)
        return _project(points, self.image_to_court_homography)

    def court_to_image(self, court_points: Sequence[Sequence[float]] | np.ndarray, *, require_metric: bool = False) -> np.ndarray:
        """Map court-plane coordinates back to image pixels."""

        if require_metric:
            self.require_metric()
        points = _coerce_points(court_points, "court_points", minimum=1)
        return _project(points, self.court_to_image_homography)

    def as_dict(self, *, include_homographies: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable response with explicit quality evidence."""

        payload: dict[str, Any] = {
            "status": self.status,
            "homography_valid": self.homography_valid,
            "quality_passed": self.quality_passed,
            "metric_enabled": self.metric_enabled,
            "metric_scope": self.metric_scope,
            "input_court_unit": self.input_court_unit,
            "coordinate_unit": self.coordinate_unit,
            "point_count": self.point_count,
            "inlier_count": self.inlier_count,
            "inlier_ratio": self.inlier_ratio,
            "inlier_mask": self.inlier_mask.astype(bool).tolist(),
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "reprojection_rmse_all_px": self.reprojection_rmse_all_px,
            "homography_condition_number": self.homography_condition_number,
            "metric_disabled_reasons": list(self.metric_disabled_reasons),
            "limitations": [
                "Valid only on the calibrated court plane.",
                "Do not treat airborne or elevated body keypoints as metre-accurate court positions.",
            ],
        }
        if include_homographies:
            payload["court_to_image_homography"] = self.court_to_image_homography.tolist()
            payload["image_to_court_homography"] = self.image_to_court_homography.tolist()
        return payload


def calibrate_court_homography(
    image_points: Sequence[Sequence[float]] | np.ndarray,
    court_points: Sequence[Sequence[float]] | np.ndarray,
    *,
    court_unit: str | None = "unknown",
    config: CalibrationConfig | Mapping[str, Any] | None = None,
) -> CourtCalibration:
    """Estimate and validate a RANSAC court-plane homography.

    Parameters are JSON-friendly parallel ``[[x, y], ...]`` point arrays.
    ``image_points`` are pixels; ``court_points`` are known locations on the
    same physical plane.  Use ``court_unit='m'`` (or another recognized linear
    unit) only when the court coordinates really encode known dimensions.

    Input/estimation errors raise :class:`CalibrationError`.  A successful
    geometric fit that fails quality thresholds still returns a result, but
    exposes ``metric_enabled=False`` and the exact reasons; this is intentional
    so an API can show calibration feedback without ever publishing unsafe
    physical measurements.
    """

    image = _coerce_points(image_points, "image_points")
    court = _coerce_points(court_points, "court_points")
    if len(image) != len(court):
        raise CalibrationInputError("image_points and court_points must have the same length")
    if len(image) < MIN_CORRESPONDENCES:
        raise CalibrationInputError(f"at least {MIN_CORRESPONDENCES} point pairs are required")
    _ensure_non_degenerate(image, "image_points")
    _ensure_non_degenerate(court, "court_points")

    if config is None:
        active_config = CalibrationConfig()
    elif isinstance(config, CalibrationConfig):
        active_config = config
    elif isinstance(config, Mapping):
        active_config = CalibrationConfig.from_mapping(config)
    else:
        raise CalibrationInputError("config must be a CalibrationConfig or options object")
    if active_config.min_inliers > len(image):
        raise CalibrationInputError("min_inliers cannot exceed the number of supplied point pairs")

    normalized_unit, metres_per_unit = _unit_scale(court_unit)
    # Convert known linear units before fitting so image-to-court output is
    # always metres when the metric gate can be enabled.
    normalized_court = court * metres_per_unit if metres_per_unit is not None else court

    try:
        court_to_image, raw_mask = cv2.findHomography(
            normalized_court,
            image,
            cv2.RANSAC,
            float(active_config.ransac_reprojection_threshold_px),
        )
    except cv2.error as error:
        raise CalibrationEstimationError("OpenCV could not estimate a court homography") from error
    if court_to_image is None or raw_mask is None:
        raise CalibrationEstimationError("RANSAC could not find a homography from the supplied point pairs")

    court_to_image = _normalise_homography(court_to_image)
    try:
        image_to_court = _normalise_homography(np.linalg.inv(court_to_image))
    except np.linalg.LinAlgError as error:
        raise CalibrationEstimationError("homography cannot be inverted for image-to-court mapping") from error

    mask = np.asarray(raw_mask).reshape(-1)
    if mask.shape[0] != len(image):
        raise CalibrationEstimationError("OpenCV returned an invalid RANSAC inlier mask")
    inlier_mask = mask.astype(bool, copy=False)
    inlier_count = int(inlier_mask.sum())
    projected = _project(normalized_court, court_to_image)
    errors_px = np.linalg.norm(projected - image, axis=1)
    if not np.isfinite(errors_px).all():
        raise CalibrationEstimationError("reprojection errors are not finite")

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values)))) if len(values) else float("inf")

    reprojection_rmse_px = rmse(errors_px[inlier_mask])
    reprojection_rmse_all_px = rmse(errors_px)
    inlier_ratio = inlier_count / len(image)
    condition_number = float(np.linalg.cond(court_to_image))

    quality_reasons: list[str] = []
    if inlier_count < active_config.min_inliers:
        quality_reasons.append("insufficient_inliers")
    if inlier_ratio < active_config.min_inlier_ratio:
        quality_reasons.append("low_inlier_ratio")
    if not isfinite(reprojection_rmse_px) or reprojection_rmse_px > active_config.max_reprojection_rmse_px:
        quality_reasons.append("reprojection_rmse_exceeds_limit")
    if not isfinite(condition_number) or condition_number > active_config.max_homography_condition_number:
        quality_reasons.append("homography_ill_conditioned")

    quality_passed = not quality_reasons
    metric_reasons = list(quality_reasons)
    if metres_per_unit is None:
        metric_reasons.append("court_unit_is_not_a_recognized_linear_unit")
    metric_enabled = quality_passed and metres_per_unit is not None

    return CourtCalibration(
        court_to_image_homography=court_to_image,
        image_to_court_homography=image_to_court,
        inlier_mask=inlier_mask,
        point_count=len(image),
        inlier_count=inlier_count,
        inlier_ratio=float(inlier_ratio),
        reprojection_rmse_px=reprojection_rmse_px,
        reprojection_rmse_all_px=reprojection_rmse_all_px,
        homography_condition_number=condition_number,
        input_court_unit=normalized_unit,
        coordinate_unit="m" if metres_per_unit is not None else normalized_unit,
        homography_valid=True,
        quality_passed=quality_passed,
        metric_enabled=metric_enabled,
        metric_disabled_reasons=tuple(metric_reasons),
    )


def calibrate_from_request(payload: Mapping[str, Any]) -> CourtCalibration:
    """Build a calibration from a decoded API request body.

    Expected shape::

        {
          "image_points": [[x, y], ...],
          "court_points": [[x_m, y_m], ...],
          "court_unit": "m",
          "options": {"max_reprojection_rmse_px": 4.0}
        }

    The unit defaults to ``unknown`` rather than metres, keeping metric output
    disabled until a client declares its coordinate system explicitly.
    """

    if not isinstance(payload, Mapping):
        raise CalibrationInputError("calibration request must be an object")
    allowed = {"image_points", "court_points", "court_unit", "options"}
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise CalibrationInputError(f"unsupported calibration field(s): {', '.join(map(str, unexpected))}")
    try:
        image_points = payload["image_points"]
        court_points = payload["court_points"]
    except KeyError as error:
        raise CalibrationInputError(f"missing required calibration field: {error.args[0]}") from error
    return calibrate_court_homography(
        image_points,
        court_points,
        court_unit=payload.get("court_unit", "unknown"),
        config=CalibrationConfig.from_mapping(payload.get("options")),
    )


def calibration_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convenience wrapper for handlers that need a JSON-safe result directly."""

    return calibrate_from_request(payload).as_dict()


# Clear aliases for callers that use "estimate" or "validate" terminology.
estimate_court_homography = calibrate_court_homography
validate_court_calibration = calibrate_from_request


__all__ = [
    "CalibrationConfig",
    "CalibrationError",
    "CalibrationEstimationError",
    "CalibrationInputError",
    "CalibrationQualityError",
    "CourtCalibration",
    "MAX_CORRESPONDENCES",
    "MIN_CORRESPONDENCES",
    "calibrate_court_homography",
    "calibrate_from_request",
    "calibration_response",
    "estimate_court_homography",
    "validate_court_calibration",
]
