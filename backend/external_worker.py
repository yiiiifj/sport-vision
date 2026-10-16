"""Small adapters for sibling projects that expose Python APIs but no CLI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import importlib.util
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.integrations import WORKSPACE_DIR


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def football_analytics(video: Path) -> dict:
    project = WORKSPACE_DIR / "football-analytics-main"
    sys.path.insert(0, str(project))
    app_module = _load_module("football_analytics_app", project / "app.py")
    return app_module.process_video_short(video, "sport_vision")


LOCAL_ENGINE_LABELS = {
    "good-badminton": "Badminton local rules",
    "good-tennis": "Tennis local rules",
    "open-hoop-vision": "Basketball local rules",
    "football-analytics": "Football local rules",
    "sportiq": "SportIQ local biomechanics",
    "sports2d": "Sports2D local biomechanics",
}
LOCAL_RULES_VERSION = "2.0.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions(packages: tuple[str, ...] = ("mediapipe", "numpy", "opencv-python-headless")) -> dict[str, str | None]:
    """Record relevant runtime versions without exporting environment secrets."""

    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _source_revision() -> str | None:
    """Return the checked-out source revision when this is a Git worktree."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and len(revision) == 40 else None


def _source_tree_dirty() -> bool | None:
    """Report whether Git says the run differed from its recorded revision."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _write_run_manifest(output_dir: Path, video: Path, label: str, summary: dict) -> None:
    """Record enough provenance to reproduce and audit a local-rules result."""

    model_path = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine_id": label,
        "execution_mode": "local_rules",
        "rules_version": LOCAL_RULES_VERSION,
        "source_revision": _source_revision(),
        "source_tree_dirty": _source_tree_dirty(),
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(aliased=True, terse=True),
            "package_versions": _package_versions(),
        },
        "configuration": {
            "pose_model_running_mode": "VIDEO",
            "num_poses": 1,
            "smoothing": "ema_alpha_0.6",
            "event_rules_version": LOCAL_RULES_VERSION,
            "random_seed": None,
        },
        "input": {"filename": video.name, "sha256": _sha256(video)},
        "pose_model": {
            "name": model_path.name,
            "sha256": _sha256(model_path) if model_path.is_file() else None,
            "running_mode": "VIDEO",
            "num_poses": 1,
        },
        "measurement_scope": {
            "calibrated": False,
            "allowed_units": ["pixels", "pixels_per_second", "shoulder_widths_per_second"],
            "not_reported": ["metres", "metres_per_second", "ball_trajectory", "player_identity"],
        },
        "input_summary": {
            "frames_processed": summary.get("frames_processed"),
            "pose_coverage": summary.get("pose_coverage"),
        },
    }
    (output_dir / "run-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _peak_events(samples: list[dict], key: str, threshold: float, event_name: str) -> list[dict]:
    """Select separated local maxima without pretending they are ML detections."""

    events: list[dict] = []
    last_frame = -10_000
    for index in range(1, len(samples) - 1):
        current = samples[index]
        value = current[key]
        if (
            value >= threshold
            and value >= samples[index - 1][key]
            and value >= samples[index + 1][key]
            and current["frame"] - last_frame >= 12
        ):
            events.append({
                "frame": current["frame"],
                "time_s": current["time_s"],
                "type": event_name,
                "heuristic_score": round(min(0.95, value / max(threshold * 1.8, 1)), 2),
                "score_kind": "rule_match_not_probability",
                "evidence": {"metric": key, "value": round(value, 3), "threshold": threshold},
            })
            last_frame = current["frame"]
    return events


def _sport_events(label: str, samples: list[dict], action_events: list[dict]) -> list[dict]:
    """Derive transparent sport-specific cues from body kinematics only."""

    if label in {"good-badminton", "good-tennis"}:
        return action_events + _peak_events(samples, "wrist_speed_shoulder_widths_s", 2.5, "fast_swing_candidate")
    if label == "football-analytics":
        return _peak_events(samples, "hip_speed_shoulder_widths_s", 0.8, "sprint_or_run_candidate")
    if label == "open-hoop-vision":
        return _peak_events(samples, "hip_speed_shoulder_widths_s", 1.0, "explosive_movement_candidate")
    if label == "sports2d":
        return _peak_events(samples, "knee_flexion_proxy_deg", 45, "deep_knee_flexion_candidate")
    return _peak_events(samples, "ankle_motion_shoulder_widths_s", 1.2, "step_candidate")


def _draw_pose(frame, pose: dict) -> None:
    """Draw the locally detected skeleton so the generated video is auditable."""

    import cv2

    points = {point["id"]: (int(point["x"]), int(point["y"])) for point in pose["keypoints"]}
    for start, end in pose["skeleton"]:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], (0, 230, 255), 2, cv2.LINE_AA)
    for x, y in points.values():
        cv2.circle(frame, (x, y), 3, (255, 80, 120), -1, cv2.LINE_AA)


def _write_metrics_csv(path: Path, samples: list[dict]) -> None:
    fieldnames = (
        "frame", "time_s", "pelvis_x_norm", "pelvis_y_norm", "wrist_speed_px_s",
        "wrist_speed_shoulder_widths_s", "hip_speed_px_s", "hip_speed_shoulder_widths_s",
        "ankle_motion_px_s", "ankle_motion_shoulder_widths_s", "knee_flexion_proxy_deg", "image_spine_lean_deg",
        "image_levelness_proxy_pct", "motion_valid", "action",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def _data_quality_summary(
    frame_count: int,
    pose_frames: int,
    visibility_values: list[float],
    longest_gap_frames: int,
    fps: float,
) -> dict:
    """Return interpretable data sufficiency, never an athletic-performance score."""

    coverage = pose_frames / frame_count if frame_count else 0.0
    mean_visibility = float(np.mean(visibility_values)) if visibility_values else 0.0
    gap_seconds = longest_gap_frames / fps if fps else 0.0
    # Coverage is the dominant term; landmark visibility and continuous tracking make a
    # result easier to audit.  This only scores input/measurement sufficiency.
    score = round(max(0.0, min(100.0, 70 * coverage + 20 * mean_visibility + 10 * max(0, 1 - gap_seconds / 3))))
    if coverage >= 0.9 and mean_visibility >= 0.6 and gap_seconds <= 1:
        verdict = "sufficient"
    elif coverage >= 0.6:
        verdict = "review_recommended"
    else:
        verdict = "insufficient_for_technical_comparison"
    return {
        "data_quality_score": score,
        "verdict": verdict,
        "pose_coverage": round(coverage, 3),
        "mean_pose_visibility": round(mean_visibility, 3),
        "longest_tracking_gap_s": round(gap_seconds, 3),
        "note": "Input/measurement sufficiency only; it is not a technique or athlete score.",
    }


def _write_html_report(output_dir: Path, summary: dict, events: list[dict]) -> None:
    """Create a standalone, offline-readable result summary for coaches/judges."""

    quality = summary.get("analysis_quality", {})
    rows = "".join(
        "<tr>"
        f"<td>{event['frame']}</td><td>{event['time_s']:.3f}s</td>"
        f"<td>{html.escape(str(event['type']))}</td><td>{event.get('heuristic_score', 0):.0%}</td>"
        "</tr>"
        for event in events
    ) or "<tr><td colspan=\"4\">No rule-based event candidates detected.</td></tr>"
    summary_json = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    document = f"""<!doctype html>
<html lang=\"en\"><meta charset=\"utf-8\"><title>Sport Vision Analysis Report</title>
<style>body{{font:16px system-ui,sans-serif;max-width:960px;margin:32px auto;padding:0 20px;color:#19202a}}h1{{color:#056b93}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px}}.card{{border:1px solid #d9e3e9;border-radius:10px;padding:14px;background:#f8fbfd}}.label{{color:#586877;font-size:.82rem}}.value{{font-size:1.45rem;font-weight:700}}table{{width:100%;border-collapse:collapse;margin-top:16px}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #d9e3e9}}pre{{overflow:auto;background:#0d1720;color:#d9edf7;padding:14px;border-radius:10px;font-size:12px}}.note{{color:#586877}}</style>
<body><h1>Sport Vision local-rules report</h1><p class=\"note\">{html.escape(summary['method'])}</p>
<div class=\"grid\"><div class=\"card\"><div class=\"label\">Data quality</div><div class=\"value\">{quality.get('data_quality_score', 0)}/100</div><div>{html.escape(quality.get('verdict', 'unknown'))}</div></div><div class=\"card\"><div class=\"label\">Pose coverage</div><div class=\"value\">{summary.get('pose_coverage', 0):.1%}</div><div>{summary.get('pose_frames', 0)} / {summary.get('frames_processed', 0)} frames</div></div><div class=\"card\"><div class=\"label\">Rule-event candidates</div><div class=\"value\">{summary.get('event_count', 0)}</div><div>See table below</div></div></div>
<h2>Event timeline</h2><p class=\"note\">Scores are rule-match scores, not calibrated probabilities.</p><table><thead><tr><th>Frame</th><th>Time</th><th>Rule candidate</th><th>Heuristic score</th></tr></thead><tbody>{rows}</tbody></table><h2>Machine-readable summary</h2><pre>{summary_json}</pre></body></html>"""
    (output_dir / "analysis-report.html").write_text(document, encoding="utf-8")


def generic_biomechanics(video: Path, label: str, output_dir: Path) -> dict:
    """Run a self-contained, explainable local video-analysis engine.

    This is intentionally a kinematics engine, not an imitation of the absent
    third-party weights.  It produces only measurements derived from the local
    MediaPipe landmarks and labels each event as a rule-based cue.
    """

    import cv2
    from backend.action_recognizer import ActionRecognizer
    from backend.pose_analyzer import PoseAnalyzer

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / "local-analysis.mp4"
    writer = cv2.VideoWriter(
        str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (frame_width, frame_height),
    )
    analyzer = PoseAnalyzer()
    recognizer = ActionRecognizer()
    frame_count = pose_frames = 0
    wrist_speeds: list[float] = []
    angle_samples: dict[str, list[float]] = {}
    samples: list[dict] = []
    action_events: list[dict] = []
    visibility_samples: list[float] = []
    previous_center: tuple[float, float, float] | None = None
    previous_wrist: tuple[float, float, float] | None = None
    previous_ankles: tuple[tuple[float, float], tuple[float, float], float] | None = None
    current_gap = longest_gap = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            pose = analyzer.process_frame(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                round((frame_count - 1) * 1000 / fps),
            )
            if not pose:
                current_gap += 1
                longest_gap = max(longest_gap, current_gap)
                writer.write(frame)
                continue
            current_gap = 0
            pose_frames += 1
            visibility_samples.append(float(pose.get("mean_keypoint_visibility", pose["confidence"])))
            action = recognizer.update(pose["keypoints"], pose["joint_angles"])
            if action["is_new_action"]:
                action_events.append({
                    "frame": frame_count,
                    "time_s": round(frame_count / fps, 3),
                    "type": action["action"],
                    "heuristic_score": action["rule_match_score"],
                    "score_kind": "rule_match_not_probability",
                    "evidence": {"source": "racket_action_rules_v2"},
                })
            keypoints = {point["name"]: point for point in pose["keypoints"]}
            shoulder_points = [keypoints.get(name) for name in ("left_shoulder", "right_shoulder")]
            body_scale = (
                math.dist(
                    (shoulder_points[0]["x"], shoulder_points[0]["y"]),
                    (shoulder_points[1]["x"], shoulder_points[1]["y"]),
                )
                if all(shoulder_points) else max(frame_width, frame_height) * 0.15
            )
            body_scale = max(body_scale, 1.0)
            center = pose["pelvis_center"]
            center_xy = (center["x"], center["y"])
            time_s = frame_count / fps
            hip_speed = 0.0
            if previous_center:
                previous_xy = previous_center[:2]
                delta_time = time_s - previous_center[2]
                if 0 < delta_time <= 0.25:
                    hip_speed = math.dist(center_xy, previous_xy) / delta_time
            previous_center = (*center_xy, time_s)
            ankles = tuple(
                (keypoints[name]["x"], keypoints[name]["y"])
                for name in ("left_ankle", "right_ankle") if name in keypoints
            )
            ankle_motion = 0.0
            if previous_ankles and len(ankles) == len(previous_ankles[:2]) == 2:
                delta_time = time_s - previous_ankles[2]
                if 0 < delta_time <= 0.25:
                    ankle_motion = sum(math.dist(now, then) for now, then in zip(ankles, previous_ankles[:2])) / (2 * delta_time)
            if len(ankles) == 2:
                previous_ankles = (*ankles, time_s)
            active_wrist = max(
                (keypoints[name] for name in ("left_wrist", "right_wrist") if name in keypoints),
                key=lambda point: point["visibility"],
                default=None,
            )
            wrist_speed = 0.0
            if active_wrist and previous_wrist:
                delta_time = time_s - previous_wrist[2]
                if 0 < delta_time <= 0.25:
                    wrist_speed = math.dist((active_wrist["x"], active_wrist["y"]), previous_wrist[:2]) / delta_time
            if active_wrist:
                previous_wrist = (active_wrist["x"], active_wrist["y"], time_s)
            # A single-frame movement exceeding 30 shoulder widths/s is a
            # landmark jump, not a credible event. Retain the frame but mark
            # speed-dependent measurements unavailable.
            motion_valid = max(wrist_speed, hip_speed, ankle_motion) / body_scale <= 30
            if not motion_valid:
                wrist_speed = hip_speed = ankle_motion = 0.0
            sample = {
                "frame": frame_count,
                "time_s": round(time_s, 3),
                "pelvis_x_norm": round(center["x"] / max(frame_width, 1), 4),
                "pelvis_y_norm": round(center["y"] / max(frame_height, 1), 4),
                "wrist_speed_px_s": round(wrist_speed, 2),
                "wrist_speed_shoulder_widths_s": round(wrist_speed / body_scale, 3),
                "hip_speed_px_s": round(hip_speed, 2),
                "hip_speed_shoulder_widths_s": round(hip_speed / body_scale, 3),
                "ankle_motion_px_s": round(ankle_motion, 2),
                "ankle_motion_shoulder_widths_s": round(ankle_motion / body_scale, 3),
                "knee_flexion_proxy_deg": float(pose["biomechanics"]["knee_flexion_proxy_deg"]),
                "image_spine_lean_deg": float(pose["biomechanics"]["image_spine_lean_deg"]),
                "image_levelness_proxy_pct": float(pose["biomechanics"]["image_levelness_proxy_pct"]),
                "motion_valid": motion_valid,
                "action": action["action"],
            }
            samples.append(sample)
            wrist_speeds.append(wrist_speed)
            for name, value in pose["joint_angles"].items():
                angle_samples.setdefault(name, []).append(float(value))
            _draw_pose(frame, pose)
            cv2.putText(frame, LOCAL_ENGINE_LABELS.get(label, "Local rules"), (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{action['action']} | wrist {wrist_speed:.0f}px/s", (16, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
        analyzer.close()

    longest_gap = max(longest_gap, current_gap)
    events = _sport_events(label, samples, action_events)
    _write_metrics_csv(output_dir / "frame-metrics.csv", samples)
    (output_dir / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    event_types: dict[str, int] = {}
    for event in events:
        event_types[event["type"]] = event_types.get(event["type"], 0) + 1
    movement = {
        "duration_s": round(frame_count / fps, 3) if fps else 0.0,
        "mean_hip_speed_px_s": round(float(np.mean([item["hip_speed_px_s"] for item in samples])), 2) if samples else 0.0,
        "peak_hip_speed_px_s": round(max((item["hip_speed_px_s"] for item in samples), default=0.0), 2),
        "mean_image_levelness_proxy_pct": round(float(np.mean([item["image_levelness_proxy_pct"] for item in samples])), 2) if samples else 0.0,
        "motion_valid_coverage": round(sum(item["motion_valid"] for item in samples) / len(samples), 3) if samples else 0.0,
    }
    summary = {
        "engine": LOCAL_ENGINE_LABELS.get(label, "Sport Vision local rules"),
        "execution_mode": "local_rules",
        "method": "Local MediaPipe landmarks plus transparent kinematic rules; no external sport checkpoint.",
        "frames_processed": frame_count,
        "pose_frames": pose_frames,
        "pose_coverage": round(pose_frames / frame_count, 3) if frame_count else 0.0,
        "mean_wrist_speed_px_s": round(float(np.mean(wrist_speeds)), 2) if wrist_speeds else 0.0,
        "mean_joint_angles": {
            name: round(float(np.mean(values)), 1) for name, values in sorted(angle_samples.items())
        },
        "analysis_quality": _data_quality_summary(
            frame_count, pose_frames, visibility_samples, longest_gap, fps,
        ),
        "movement": movement,
        "event_count": len(events),
        "event_breakdown": event_types,
        "artifacts": [
            "summary.json", "frame-metrics.csv", "events.json", "local-analysis.mp4", "analysis-report.html",
            "run-manifest.json",
        ],
    }
    _write_html_report(output_dir, summary, events)
    return summary


def sportiq(video: Path, output_dir: Path) -> dict:
    return generic_biomechanics(video, "sportiq", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sport Vision external-engine adapter")
    parser.add_argument("engine", choices=("football-analytics", "sportiq", "local-rules"))
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", default="Sport Vision")
    args = parser.parse_args()

    if args.engine == "football-analytics":
        result = football_analytics(args.video)
    elif args.engine == "sportiq":
        result = sportiq(args.video, args.output.parent)
    else:  # local-rules
        result = generic_biomechanics(args.video, args.label, args.output.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if args.engine == "local-rules":
        _write_run_manifest(args.output.parent, args.video, args.label, result)
    print(f"Wrote result: {args.output}")


if __name__ == "__main__":
    main()
