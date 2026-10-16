"""Deterministic evaluation utilities for Sport Vision experiments.

The portal's live rule engine is deliberately separate from evaluation.  This
module turns frozen annotations and run artifacts into versioned evidence that
can be reviewed in a competition report.  It provides only metrics whose
definitions are implemented here:

* event precision/recall/F1 with one-to-one, class-aware frame tolerance;
* 2-D pose PCK and image-space landmark error against annotated frames;
* held-out calibration reprojection error; and
* artifact-manifest integrity verification.

It intentionally does **not** label its lightweight tracker scores as HOTA,
IDF1 or MOTA.  Those multi-object tracking metrics need the official TrackEval
protocol and a properly formatted benchmark.  See ``docs/COMPETITION_PROTOCOL.md``.
All functions are dependency-light (standard library only) so they can run in
a clean evaluation environment without loading a vision model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EVALUATION_SCHEMA_VERSION = 1
EXPERIMENT_REPORT_SCHEMA_VERSION = 1


class EvaluationError(ValueError):
    """Raised when an annotation or evaluation input breaks the contract."""


@dataclass(frozen=True, slots=True)
class Event:
    """Canonical event used for deterministic class-aware matching."""

    event_type: str
    frame: int
    source_index: int


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "a positive finite number" if positive else "a finite number"
        raise EvaluationError(f"{name} must be {qualifier}")
    return number


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(f"{name} must be a non-negative integer")
    return value


def _event_items(value: Any, name: str) -> Sequence[Any]:
    """Accept an event list or a JSON object containing an ``events`` list."""

    if isinstance(value, Mapping):
        value = value.get("events")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvaluationError(f"{name} must be an event list or an object with events")
    return value


def canonical_events(value: Any, *, name: str = "events") -> list[Event]:
    """Validate events and normalize their time to an integer frame index.

    Every event needs a non-empty ``type`` and either ``frame`` or
    ``start_frame``.  ``frame`` is preferred because rule-engine artifacts
    record a peak frame.  Start/end spans may be retained in an annotation
    file, but a separate interval metric is required to evaluate duration.
    """

    result: list[Event] = []
    for index, raw in enumerate(_event_items(value, name)):
        if not isinstance(raw, Mapping):
            raise EvaluationError(f"{name}[{index}] must be an object")
        event_type = raw.get("type")
        if not isinstance(event_type, str) or not event_type.strip():
            raise EvaluationError(f"{name}[{index}].type must be a non-empty string")
        frame_value = raw.get("frame", raw.get("start_frame"))
        result.append(Event(event_type.strip(), _non_negative_int(frame_value, f"{name}[{index}].frame"), index))
    return result


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_divide(2.0 * precision * recall, precision + recall)


def _match_one_class(predicted: Sequence[Event], truth: Sequence[Event], tolerance_frames: int) -> list[tuple[Event, Event]]:
    """Return a maximum-cardinality ordered match for one event class.

    With a fixed absolute time tolerance, earliest compatible matching is
    optimal for cardinality on sorted one-dimensional events.  It is
    deterministic, avoids double-counting, and exposes each paired delta for
    audit rather than selecting an arbitrary many-to-one nearest peak.
    """

    ordered_predicted = sorted(predicted, key=lambda item: (item.frame, item.source_index))
    ordered_truth = sorted(truth, key=lambda item: (item.frame, item.source_index))
    pairs: list[tuple[Event, Event]] = []
    pred_index = 0
    truth_index = 0
    while pred_index < len(ordered_predicted) and truth_index < len(ordered_truth):
        prediction = ordered_predicted[pred_index]
        annotation = ordered_truth[truth_index]
        delta = prediction.frame - annotation.frame
        if abs(delta) <= tolerance_frames:
            pairs.append((prediction, annotation))
            pred_index += 1
            truth_index += 1
        elif prediction.frame < annotation.frame - tolerance_frames:
            pred_index += 1
        else:
            truth_index += 1
    return pairs


def evaluate_event_detection(
    predicted: Any,
    ground_truth: Any,
    *,
    tolerance_frames: int = 3,
) -> dict[str, Any]:
    """Evaluate point events with class-aware, one-to-one temporal matching.

    ``tolerance_frames`` must be selected before looking at the held-out test
    results.  The result distinguishes micro-F1 (all events together) from
    macro-F1 (unweighted class average), and returns matched frame deltas as
    reviewable evidence.
    """

    tolerance_frames = _non_negative_int(tolerance_frames, "tolerance_frames")
    predicted_events = canonical_events(predicted, name="predicted")
    truth_events = canonical_events(ground_truth, name="ground_truth")
    predicted_by_type: dict[str, list[Event]] = defaultdict(list)
    truth_by_type: dict[str, list[Event]] = defaultdict(list)
    for event in predicted_events:
        predicted_by_type[event.event_type].append(event)
    for event in truth_events:
        truth_by_type[event.event_type].append(event)

    per_class: dict[str, dict[str, Any]] = {}
    all_pairs: list[dict[str, Any]] = []
    total_tp = 0
    for event_type in sorted(set(predicted_by_type) | set(truth_by_type)):
        pairs = _match_one_class(
            predicted_by_type[event_type], truth_by_type[event_type], tolerance_frames
        )
        tp = len(pairs)
        predicted_count = len(predicted_by_type[event_type])
        truth_count = len(truth_by_type[event_type])
        fp = predicted_count - tp
        fn = truth_count - tp
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        per_class[event_type] = {
            "predicted": predicted_count,
            "ground_truth": truth_count,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(_f1(precision, recall), 6),
        }
        total_tp += tp
        all_pairs.extend(
            {
                "type": event_type,
                "prediction_frame": prediction.frame,
                "ground_truth_frame": annotation.frame,
                "delta_frames": prediction.frame - annotation.frame,
            }
            for prediction, annotation in pairs
        )

    total_predicted = len(predicted_events)
    total_truth = len(truth_events)
    micro_precision = _safe_divide(total_tp, total_predicted)
    micro_recall = _safe_divide(total_tp, total_truth)
    macro_f1 = _safe_divide(sum(metric["f1"] for metric in per_class.values()), len(per_class))
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "metric": "event_detection_frame_tolerance",
        "tolerance_frames": tolerance_frames,
        "predicted_events": total_predicted,
        "ground_truth_events": total_truth,
        "true_positive": total_tp,
        "false_positive": total_predicted - total_tp,
        "false_negative": total_truth - total_tp,
        "micro_precision": round(micro_precision, 6),
        "micro_recall": round(micro_recall, 6),
        "micro_f1": round(_f1(micro_precision, micro_recall), 6),
        "macro_f1": round(macro_f1, 6),
        "per_class": per_class,
        "matched_pairs": sorted(all_pairs, key=lambda item: (item["type"], item["ground_truth_frame"], item["prediction_frame"])),
        "limitations": [
            "Scores event-point timing only; event duration requires an interval metric.",
            "The tolerance must be frozen before evaluating the final test split.",
        ],
    }


def _frame_records(value: Any, name: str) -> dict[int, Mapping[str, Any]]:
    """Normalize pose records from ``[{frame, joints, ...}]`` or a frame map."""

    records = value.get("frames") if isinstance(value, Mapping) and "frames" in value else value
    if isinstance(records, Mapping):
        iterable: Iterable[tuple[Any, Any]] = records.items()
    elif isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        iterable = ((item.get("frame") if isinstance(item, Mapping) else None, item) for item in records)
    else:
        raise EvaluationError(f"{name} must be a frame list or an object with frames")
    normalized: dict[int, Mapping[str, Any]] = {}
    for raw_frame, raw_record in iterable:
        frame = _non_negative_int(int(raw_frame) if isinstance(raw_frame, str) and raw_frame.isdigit() else raw_frame, f"{name}.frame")
        if not isinstance(raw_record, Mapping):
            raise EvaluationError(f"{name}[{frame}] must be an object")
        if "joints" not in raw_record or not isinstance(raw_record["joints"], Mapping):
            raise EvaluationError(f"{name}[{frame}].joints must be an object")
        if frame in normalized:
            raise EvaluationError(f"{name} contains duplicate frame {frame}")
        normalized[frame] = raw_record
    return normalized


def _joint_point(value: Any, name: str) -> tuple[float, float, bool]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{name} must be an object with x and y")
    visible = value.get("visible", value.get("visibility", True))
    if isinstance(visible, (int, float)) and not isinstance(visible, bool):
        visible = float(visible) > 0
    if not isinstance(visible, bool):
        raise EvaluationError(f"{name}.visible must be boolean or numeric")
    return _finite(value.get("x"), f"{name}.x"), _finite(value.get("y"), f"{name}.y"), visible


def evaluate_pose_pck(
    predicted: Any,
    ground_truth: Any,
    *,
    thresholds: Sequence[float] = (0.05, 0.1),
) -> dict[str, Any]:
    """Evaluate 2-D pose against frozen annotations with declared pixel scale.

    Each ground-truth frame needs positive ``normalization_scale_px`` (usually
    the annotated person bounding-box diagonal or torso scale) plus a ``joints``
    map.  Invisible ground-truth joints are excluded; a visible joint missing
    from a prediction counts as a PCK failure.  This makes the denominator
    explicit and avoids presenting detector coverage as pose accuracy.
    """

    checked_thresholds = sorted({_finite(item, "threshold", positive=True) for item in thresholds})
    if not checked_thresholds:
        raise EvaluationError("at least one positive PCK threshold is required")
    predicted_frames = _frame_records(predicted, "predicted")
    truth_frames = _frame_records(ground_truth, "ground_truth")
    errors_px: list[float] = []
    normalized_errors: list[float] = []
    per_joint_errors: dict[str, list[float]] = defaultdict(list)
    visible_joint_count = 0
    missing_prediction_count = 0
    evaluated_frames = 0

    for frame, truth_record in truth_frames.items():
        scale = _finite(
            truth_record.get("normalization_scale_px"),
            f"ground_truth[{frame}].normalization_scale_px",
            positive=True,
        )
        truth_joints = truth_record["joints"]
        prediction_record = predicted_frames.get(frame)
        if prediction_record is not None:
            evaluated_frames += 1
        predicted_joints = prediction_record["joints"] if prediction_record is not None else {}
        for joint_name, raw_truth_joint in truth_joints.items():
            truth_x, truth_y, visible = _joint_point(raw_truth_joint, f"ground_truth[{frame}].joints[{joint_name!r}]")
            if not visible:
                continue
            visible_joint_count += 1
            raw_prediction_joint = predicted_joints.get(joint_name)
            if raw_prediction_joint is None:
                missing_prediction_count += 1
                continue
            prediction_x, prediction_y, _ = _joint_point(raw_prediction_joint, f"predicted[{frame}].joints[{joint_name!r}]")
            error = math.hypot(prediction_x - truth_x, prediction_y - truth_y)
            normalized = error / scale
            errors_px.append(error)
            normalized_errors.append(normalized)
            per_joint_errors[str(joint_name)].append(normalized)

    if not visible_joint_count:
        raise EvaluationError("ground_truth contains no visible joints")
    threshold_scores = {
        f"pck@{threshold:g}": round(
            _safe_divide(sum(error <= threshold for error in normalized_errors), visible_joint_count), 6
        )
        for threshold in checked_thresholds
    }
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "metric": "pose_pck_2d",
        "ground_truth_frames": len(truth_frames),
        "prediction_frames": len(predicted_frames),
        "frames_with_prediction": evaluated_frames,
        "visible_joint_count": visible_joint_count,
        "predicted_visible_joint_count": len(errors_px),
        "missing_prediction_count": missing_prediction_count,
        "mean_error_px": round(statistics.fmean(errors_px), 6) if errors_px else None,
        "median_error_px": round(statistics.median(errors_px), 6) if errors_px else None,
        "mean_normalized_error": round(statistics.fmean(normalized_errors), 6) if normalized_errors else None,
        **threshold_scores,
        "per_joint_mean_normalized_error": {
            joint: round(statistics.fmean(values), 6)
            for joint, values in sorted(per_joint_errors.items())
        },
        "limitations": [
            "PCK depends on the frozen annotation scale definition in normalization_scale_px.",
            "This is image-space 2-D accuracy, not 3-D or biomechanical validity.",
        ],
    }


def _point(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, Mapping):
        return _finite(value.get("x"), f"{name}.x"), _finite(value.get("y"), f"{name}.y")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise EvaluationError(f"{name} must be [x, y] or an object with x and y")
    return _finite(value[0], f"{name}[0]"), _finite(value[1], f"{name}[1]")


def evaluate_calibration_reprojection(
    projected_image_points: Sequence[Any],
    held_out_image_points: Sequence[Any],
) -> dict[str, Any]:
    """Report held-out image reprojection error from paired pixel points.

    The caller must project held-out court coordinates with a calibration fit
    that did not use these image points.  Keeping the projection outside this
    helper prevents accidental evaluation on the fit correspondences.
    """

    if len(projected_image_points) != len(held_out_image_points) or not projected_image_points:
        raise EvaluationError("projected and held-out image points must be non-empty parallel sequences")
    errors = [
        math.hypot(predicted[0] - truth[0], predicted[1] - truth[1])
        for index, (raw_predicted, raw_truth) in enumerate(zip(projected_image_points, held_out_image_points))
        for predicted, truth in [(_point(raw_predicted, f"projected[{index}]"), _point(raw_truth, f"held_out[{index}]"))]
    ]
    rmse = math.sqrt(statistics.fmean(error * error for error in errors))
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "metric": "held_out_calibration_reprojection_px",
        "point_count": len(errors),
        "rmse_px": round(rmse, 6),
        "mean_error_px": round(statistics.fmean(errors), 6),
        "median_error_px": round(statistics.median(errors), 6),
        "max_error_px": round(max(errors), 6),
        "limitations": ["Use only points that were excluded from homography fitting."],
    }


def validate_split_manifest(value: Any) -> dict[str, Any]:
    """Validate a frozen train/validation/test clip split before evaluation.

    The manifest is deliberately small and data-independent::

        {
          "schema_version": 1,
          "dataset_name": "racket-private-v1",
          "annotation_version": "2026-10-16",
          "splits": {"train": ["clip-001"], "validation": [], "test": ["clip-101"]}
        }

    It does not prove labels are correct, but it catches test leakage and
    accidental duplicate clips before any threshold or model comparison is
    reported.  Malformed JSON contracts raise :class:`EvaluationError`; a
    structurally valid but non-disjoint manifest returns ``valid: false`` with
    actionable findings so a CI job can retain its evidence.
    """

    if not isinstance(value, Mapping):
        raise EvaluationError("benchmark manifest must be an object")
    dataset_name = value.get("dataset_name")
    annotation_version = value.get("annotation_version")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise EvaluationError("benchmark manifest dataset_name must be a non-empty string")
    if not isinstance(annotation_version, str) or not annotation_version.strip():
        raise EvaluationError("benchmark manifest annotation_version must be a non-empty string")
    splits = value.get("splits")
    if not isinstance(splits, Mapping):
        raise EvaluationError("benchmark manifest splits must be an object")

    normalized: dict[str, list[str]] = {}
    issues: list[dict[str, Any]] = []
    owners: dict[str, list[str]] = defaultdict(list)
    for split_name, clip_ids in splits.items():
        if not isinstance(split_name, str) or not split_name.strip():
            raise EvaluationError("split names must be non-empty strings")
        if isinstance(clip_ids, (str, bytes)) or not isinstance(clip_ids, Sequence):
            raise EvaluationError(f"splits.{split_name} must be a list of clip IDs")
        seen_within_split: set[str] = set()
        normalized_ids: list[str] = []
        for index, clip_id in enumerate(clip_ids):
            if not isinstance(clip_id, str) or not clip_id.strip():
                raise EvaluationError(f"splits.{split_name}[{index}] must be a non-empty string")
            canonical_id = clip_id.strip()
            if canonical_id in seen_within_split:
                issues.append({"kind": "duplicate_within_split", "split": split_name, "clip_id": canonical_id})
            else:
                seen_within_split.add(canonical_id)
                normalized_ids.append(canonical_id)
                owners[canonical_id].append(split_name)
        normalized[split_name] = normalized_ids

    required = ("train", "validation", "test")
    missing_splits = [name for name in required if name not in normalized]
    if missing_splits:
        issues.append({"kind": "missing_required_splits", "splits": missing_splits})
    for split_name in required:
        if split_name in normalized and not normalized[split_name]:
            issues.append({"kind": "empty_required_split", "split": split_name})
    for clip_id, split_names in sorted(owners.items()):
        if len(split_names) > 1:
            issues.append({"kind": "cross_split_leakage", "clip_id": clip_id, "splits": sorted(split_names)})

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "metric": "benchmark_split_integrity",
        "dataset_name": dataset_name.strip(),
        "annotation_version": annotation_version.strip(),
        "split_counts": {name: len(ids) for name, ids in sorted(normalized.items())},
        "unique_clip_count": len(owners),
        "valid": not issues,
        "findings": issues,
        "limitations": [
            "This verifies declared clip-ID separation, not annotation quality or athlete-level leakage.",
            "Use anonymized stable IDs and document camera/athlete grouping when those are leakage risks.",
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Verify every declared artifact hash without traversing outside a run.

    Missing, modified, unexpected duplicate and unsafe paths are returned as
    evidence rather than silently ignored.  This makes a recorded experiment
    auditable after it leaves the job runner.
    """

    root = Path(output_dir).resolve()
    manifest_path = root / "artifact-manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise EvaluationError("output_dir must contain artifact-manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError("artifact manifest is not valid JSON") from error
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("artifacts"), list):
        raise EvaluationError("artifact manifest must contain an artifacts list")
    declared: set[str] = set()
    findings: list[dict[str, Any]] = []
    for index, artifact in enumerate(manifest["artifacts"]):
        if not isinstance(artifact, Mapping):
            findings.append({"index": index, "status": "invalid_record"})
            continue
        name = artifact.get("name")
        expected_hash = artifact.get("sha256")
        if not isinstance(name, str) or not name or not isinstance(expected_hash, str) or len(expected_hash) != 64:
            findings.append({"index": index, "status": "invalid_record"})
            continue
        if name in declared:
            findings.append({"name": name, "status": "duplicate_manifest_entry"})
            continue
        declared.add(name)
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            findings.append({"name": name, "status": "missing_or_unsafe"})
            continue
        actual_hash = _sha256(path)
        actual_size = path.stat().st_size
        if actual_hash != expected_hash:
            findings.append({"name": name, "status": "hash_mismatch", "actual_sha256": actual_hash})
        elif artifact.get("size_bytes") != actual_size:
            findings.append({"name": name, "status": "size_mismatch", "actual_size_bytes": actual_size})
        else:
            findings.append({"name": name, "status": "verified"})
    # A file added after manifest creation is also an integrity failure.  Do
    # not silently accept a report that hashes a subset of the published run.
    observed: set[str] = set()
    for path in root.rglob("*"):
        if path.name == "artifact-manifest.json":
            continue
        relative_name = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append({"name": relative_name, "status": "unsafe_symlink"})
        elif path.is_file():
            observed.add(relative_name)
    for name in sorted(observed - declared):
        findings.append({"name": name, "status": "unmanifested_file"})
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "metric": "artifact_manifest_integrity",
        "output_dir": str(root),
        "declared_artifact_count": len(manifest["artifacts"]),
        "verified_artifact_count": sum(item["status"] == "verified" for item in findings),
        "valid": bool(findings) and all(item["status"] == "verified" for item in findings),
        "findings": findings,
    }


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{name} must be a non-empty string")
    return value.strip()


def _resolve_evidence_file(base_dir: Path, raw_path: Any, name: str) -> Path:
    """Resolve a manifest-relative evidence file without accepting folders."""

    candidate = raw_path if isinstance(raw_path, Path) else Path(_required_text(raw_path, name))
    path = candidate if candidate.is_absolute() else base_dir / candidate
    path = path.resolve()
    if not path.is_file():
        raise EvaluationError(f"{name} must name an existing JSON file")
    return path


def _read_evidence_json(path: Path, name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"{name} is not valid JSON") from error


def build_experiment_report(
    manifest: Any,
    *,
    base_dir: str | Path,
) -> dict[str, Any]:
    """Build a hash-linked, reviewable report for one frozen experiment.

    The report deliberately does not calculate a new score.  It binds results
    that were already produced by the explicit evaluators to a frozen split and
    a verified run.  Keeping this assembly step deterministic prevents a PDF or
    presentation from silently mixing a result from one run with another run's
    environment or annotation revision.

    ``manifest`` is a small, checked-in JSON document.  Its required shape is::

        {
          "experiment_id": "racket-rules-baseline-001",
          "split_manifest": "split-manifest.json",
          "runs": [{"run_id": "...", "output_dir": "../outputs/..."}],
          "evaluations": [{
            "name": "held-out events",
            "report": "events-test.json",
            "expected_metric": "event_detection_frame_tolerance"
          }]
        }

    Paths are resolved relative to the experiment manifest.  The resulting
    report includes SHA-256 receipts for every JSON input and validates every
    referenced artifact directory before it can be marked valid.
    """

    if not isinstance(manifest, Mapping):
        raise EvaluationError("experiment manifest must be an object")
    root = Path(base_dir).resolve()
    experiment_id = _required_text(manifest.get("experiment_id"), "experiment_id")
    declared_schema = manifest.get("schema_version", EXPERIMENT_REPORT_SCHEMA_VERSION)
    if declared_schema != EXPERIMENT_REPORT_SCHEMA_VERSION:
        raise EvaluationError(
            f"experiment manifest schema_version must be {EXPERIMENT_REPORT_SCHEMA_VERSION}"
        )

    split_path = _resolve_evidence_file(root, manifest.get("split_manifest"), "split_manifest")
    split_result = validate_split_manifest(_read_evidence_json(split_path, "split_manifest"))

    raw_runs = manifest.get("runs")
    if isinstance(raw_runs, (str, bytes)) or not isinstance(raw_runs, Sequence) or not raw_runs:
        raise EvaluationError("runs must be a non-empty list")
    run_reports: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, Mapping):
            raise EvaluationError(f"runs[{index}] must be an object")
        run_id = _required_text(raw_run.get("run_id"), f"runs[{index}].run_id")
        if run_id in seen_run_ids:
            raise EvaluationError(f"runs contains duplicate run_id {run_id!r}")
        seen_run_ids.add(run_id)
        output_dir = _resolve_evidence_file(
            root, Path(_required_text(raw_run.get("output_dir"), f"runs[{index}].output_dir")) / "artifact-manifest.json",
            f"runs[{index}].output_dir/artifact-manifest.json",
        ).parent
        integrity = verify_artifact_manifest(output_dir)
        artifact_manifest = _read_evidence_json(output_dir / "artifact-manifest.json", "artifact manifest")
        declared_run_id = artifact_manifest.get("run_id") if isinstance(artifact_manifest, Mapping) else None
        run_manifest_path = output_dir / "run-manifest.json"
        run_manifest = _read_evidence_json(run_manifest_path, "run manifest") if run_manifest_path.is_file() else None
        provenance_valid = bool(
            isinstance(run_manifest, Mapping)
            and isinstance(run_manifest.get("source_revision"), str)
            and run_manifest["source_revision"].strip()
            and isinstance(run_manifest.get("source_tree_dirty"), bool)
            and isinstance(run_manifest.get("runtime"), Mapping)
            and isinstance(run_manifest.get("configuration"), Mapping)
        )
        run_reports.append({
            "run_id": run_id,
            "artifact_manifest_sha256": _sha256(output_dir / "artifact-manifest.json"),
            "run_manifest_sha256": _sha256(run_manifest_path) if run_manifest_path.is_file() else None,
            "artifact_integrity": integrity,
            "run_id_matches_artifact_manifest": declared_run_id == run_id,
            "provenance_valid": provenance_valid,
            "provenance": {
                "source_revision": run_manifest.get("source_revision") if isinstance(run_manifest, Mapping) else None,
                "source_tree_dirty": run_manifest.get("source_tree_dirty") if isinstance(run_manifest, Mapping) else None,
                "rules_version": run_manifest.get("rules_version") if isinstance(run_manifest, Mapping) else None,
                "runtime": run_manifest.get("runtime") if isinstance(run_manifest, Mapping) else None,
                "configuration": run_manifest.get("configuration") if isinstance(run_manifest, Mapping) else None,
            },
        })

    raw_evaluations = manifest.get("evaluations")
    if isinstance(raw_evaluations, (str, bytes)) or not isinstance(raw_evaluations, Sequence) or not raw_evaluations:
        raise EvaluationError("evaluations must be a non-empty list")
    evaluation_reports: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, raw_evaluation in enumerate(raw_evaluations):
        if not isinstance(raw_evaluation, Mapping):
            raise EvaluationError(f"evaluations[{index}] must be an object")
        name = _required_text(raw_evaluation.get("name"), f"evaluations[{index}].name")
        if name in seen_names:
            raise EvaluationError(f"evaluations contains duplicate name {name!r}")
        seen_names.add(name)
        expected_metric = _required_text(raw_evaluation.get("expected_metric"), f"evaluations[{index}].expected_metric")
        report_path = _resolve_evidence_file(root, raw_evaluation.get("report"), f"evaluations[{index}].report")
        report = _read_evidence_json(report_path, f"evaluations[{index}].report")
        report_metric = report.get("metric") if isinstance(report, Mapping) else None
        evaluation_reports.append({
            "name": name,
            "expected_metric": expected_metric,
            "reported_metric": report_metric,
            "metric_matches": report_metric == expected_metric,
            "report_sha256": _sha256(report_path),
            "report_valid": report.get("valid", True) if isinstance(report, Mapping) else False,
        })

    invalid_integrity = any(
        not item["artifact_integrity"].get("valid", False)
        or not item["run_id_matches_artifact_manifest"]
        or not item["provenance_valid"]
        for item in run_reports
    )
    invalid_evaluations = any(
        not item["metric_matches"] or not item["report_valid"]
        for item in evaluation_reports
    )
    findings: list[dict[str, Any]] = []
    if not split_result["valid"]:
        findings.append({"kind": "invalid_split_manifest", "details": split_result["findings"]})
    if invalid_integrity:
        findings.append({"kind": "invalid_run_artifacts_or_identity"})
    if invalid_evaluations:
        findings.append({"kind": "invalid_or_mismatched_evaluation_receipt"})
    return {
        "schema_version": EXPERIMENT_REPORT_SCHEMA_VERSION,
        "metric": "experiment_evidence_bundle",
        "experiment_id": experiment_id,
        "experiment_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "split_manifest_sha256": _sha256(split_path),
        "split_integrity": split_result,
        "runs": run_reports,
        "evaluations": evaluation_reports,
        "valid": not findings,
        "findings": findings,
        "limitations": [
            "This binds declared evidence files; it does not establish annotation quality, statistical significance, or external validity.",
            "A valid bundle is not an accuracy, clinical, production, or competition-award claim without the independent evaluation protocol.",
        ],
    }


def _load_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"unable to read JSON: {path}") from error


def _write_json(path: str | None, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(serialized, end="")
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as temporary:
        temporary.write(serialized)
        temporary_path = Path(temporary.name)
    temporary_path.replace(target)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate frozen Sport Vision annotations and artifacts")
    subcommands = parser.add_subparsers(dest="command", required=True)
    events = subcommands.add_parser("events", help="score point events with a frame tolerance")
    events.add_argument("--predicted", required=True)
    events.add_argument("--ground-truth", required=True)
    events.add_argument("--tolerance-frames", type=int, default=3)
    events.add_argument("--output")
    pose = subcommands.add_parser("pose-pck", help="score 2-D annotated joints")
    pose.add_argument("--predicted", required=True)
    pose.add_argument("--ground-truth", required=True)
    pose.add_argument("--threshold", type=float, action="append", default=None)
    pose.add_argument("--output")
    calibration = subcommands.add_parser("calibration", help="score held-out projected pixel points")
    calibration.add_argument("--projected", required=True, help="JSON list of projected points")
    calibration.add_argument("--held-out", required=True, help="JSON list of annotated points")
    calibration.add_argument("--output")
    split = subcommands.add_parser("validate-splits", help="reject overlapping frozen benchmark splits")
    split.add_argument("--manifest", required=True)
    split.add_argument("--output")
    artifacts = subcommands.add_parser("verify-artifacts", help="verify a job artifact manifest")
    artifacts.add_argument("--output-dir", required=True)
    artifacts.add_argument("--output")
    experiment = subcommands.add_parser("experiment-report", help="assemble a hash-linked frozen-experiment evidence report")
    experiment.add_argument("--manifest", required=True)
    experiment.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit JSON evaluation CLI; return a shell-safe exit code."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "events":
            payload = evaluate_event_detection(
                _load_json(args.predicted), _load_json(args.ground_truth), tolerance_frames=args.tolerance_frames
            )
        elif args.command == "pose-pck":
            payload = evaluate_pose_pck(
                _load_json(args.predicted), _load_json(args.ground_truth),
                thresholds=tuple(args.threshold) if args.threshold else (0.05, 0.1),
            )
        elif args.command == "calibration":
            payload = evaluate_calibration_reprojection(_load_json(args.projected), _load_json(args.held_out))
        elif args.command == "validate-splits":
            payload = validate_split_manifest(_load_json(args.manifest))
        elif args.command == "verify-artifacts":
            payload = verify_artifact_manifest(args.output_dir)
        else:
            manifest_path = Path(args.manifest).resolve()
            payload = build_experiment_report(
                _load_json(str(manifest_path)), base_dir=manifest_path.parent
            )
        _write_json(args.output, payload)
        return 0 if payload.get("valid", True) else 2
    except EvaluationError as error:
        _parser().error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "EXPERIMENT_REPORT_SCHEMA_VERSION",
    "EvaluationError",
    "Event",
    "canonical_events",
    "evaluate_calibration_reprojection",
    "evaluate_event_detection",
    "evaluate_pose_pck",
    "validate_split_manifest",
    "verify_artifact_manifest",
    "build_experiment_report",
]
