"""Workspace integration registry for the Sport Vision portal.

The projects in ``D:\\AIC`` use different frameworks and model stacks.  They
remain isolated engines (so a Paddle, PyTorch or Streamlit dependency cannot
break the lightweight Sport Vision server), while this registry gives the main
frontend one truthful catalogue and one extension point for every engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Integration:
    """A locally checked sports-analysis capability exposed by the portal."""

    id: str
    directory: str
    name: str
    sport: str
    icon: str
    category: str
    summary: str
    capabilities: tuple[str, ...]
    entrypoint: str
    runtime: str
    notes: str = ""
    required_assets: tuple[str, ...] = ()

    @property
    def path(self) -> Path:
        return WORKSPACE_DIR / self.directory

    @property
    def entrypoint_path(self) -> Path:
        return self.path / self.entrypoint

    def as_dict(self) -> dict:
        present = self.path.is_dir()
        entrypoint_present = self.entrypoint_path.exists()
        return {
            "id": self.id,
            "name": self.name,
            "sport": self.sport,
            "icon": self.icon,
            "category": self.category,
            "summary": self.summary,
            "capabilities": list(self.capabilities),
            "runtime": self.runtime,
            "notes": self.notes,
            # Do not expose host filesystem paths to the browser.  The portal
            # only reports a usable source location and launch convention.
            "status": "available" if present and entrypoint_present else "incomplete",
            "entrypoint": self.entrypoint,
            "job_supported": self.id in MANAGED_VIDEO_ENGINES,
            "requires_template": self.id in COURT_TEMPLATE_ENGINES,
        }


INTEGRATIONS: tuple[Integration, ...] = (
    Integration(
        "sport-vision", "sport-vision", "Sport Vision", "all", "⚡",
        "Unified portal", "Real-time pose, action and biomechanics analysis.",
        ("Pose tracking", "Action recognition", "Biomechanics", "Heatmap"),
        "backend/main.py", "FastAPI + WebSocket", "Runs in this page.",
    ),
    Integration(
        "good-badminton", "Good-Badminton-main", "Good Badminton", "badminton", "🏸",
        "Video analysis", "Court-aware shuttlecock, player and trajectory analysis.",
        ("Court calibration", "Shuttle tracking", "Player trajectories", "Heatmaps"),
        "webui/app.py", "Gradio + Python",
        required_assets=("weights/yolo11s-ball.pt",),
    ),
    Integration(
        "good-tennis", "Good-Tennis-main", "Good Tennis", "tennis", "🎾",
        "Video analysis", "Court-aware tennis tracking and bounce analysis.",
        ("Ball tracking", "Court mapping", "Bounce detection", "Player statistics"),
        "main.py", "Python CLI",
        required_assets=("weights/tennis-ball.pt",),
    ),
    Integration(
        "open-hoop-vision", "OpenHoopVision", "OpenHoopVision", "basketball", "🏀",
        "Video analysis", "Basketball players, ball, rim, court and event analysis.",
        ("Multi-object tracking", "Possession", "Shot events", "Jersey re-ID"),
        "main.py", "Python CLI", "Jersey-number OCR degrades gracefully when its optional checkpoint is absent.",
        required_assets=("weights/rf-detr-basketball.pth", "weights/yolov8n-pose.pt"),
    ),
    Integration(
        "football-analytics", "football-analytics-main", "Football Analytics", "football", "⚽",
        "Video analysis", "Video-based player, team, ball and pose analytics.",
        ("YOLO detection", "Team classification", "Ball tracking", "Formation"),
        "app.py", "Flask",
        required_assets=("models/yolov8n.pt", "models/team_classifier.pkl", "models/football_pose_classifier.pkl"),
    ),
    Integration(
        "tactiq", "TactiQ-AI", "TactiQ AI", "football", "📈",
        "Match intelligence", "StatsBomb match intelligence and tactical visualizations.",
        ("Passing networks", "Shot maps", "Match timeline", "AI reports"),
        "backend/app/main.py", "FastAPI + Next.js",
    ),
    Integration(
        "football-intelligence-hub", "Football-Intelligence-Hub", "Football Intelligence Hub", "football", "🧠",
        "Match intelligence", "Team reports, role fit and match visualizations.",
        ("Pressing maps", "Player roles", "Team reports", "LLM coach"),
        "app.py", "Streamlit",
    ),
    Integration(
        "sportiq", "SportIQ", "SportIQ", "all", "📊",
        "Performance analytics", "Athlete performance, tactical and biomechanics services.",
        ("Motion metrics", "Gait analysis", "Performance reports", "Tactics"),
        "api/server.py", "MediaPipe fallback", "Uses Sport Vision's local pose engine because the upstream checkpoint is not published.",
    ),
    Integration(
        "sports2d", "Sports2D-main", "Sports2D", "all", "🦴",
        "Biomechanics", "Markerless 2D pose and joint-angle extraction.",
        ("2D pose", "Joint angles", "Video processing"),
        "Sports2D/Sports2D.py", "Python package",
    ),
    Integration(
        "athlete-pose-3d", "AthletePose3D-main", "AthletePose3D", "all", "🧍",
        "Dataset & research", "Athletic 3D-pose benchmark and training resources.",
        ("3D pose data", "Kinematic validation", "Fine-tuning resources"),
        "README.md", "Dataset + research code",
    ),
    Integration(
        "paddle-video", "PaddleVideo-develop", "PaddleVideo", "all", "🎞️",
        "Video intelligence", "Action recognition, detection and video tagging toolkit.",
        ("Action recognition", "Video tagging", "Detection", "Model zoo"),
        "main.py", "PaddlePaddle",
    ),
)

# Only engines whose required inputs can be represented as uploaded files are
# run by the portal. Research, training and match-data services remain visible
# without being incorrectly presented as a generic video analyser.
MANAGED_VIDEO_ENGINES = frozenset({
    "good-badminton", "good-tennis", "open-hoop-vision", "football-analytics", "sportiq", "sports2d",
})
COURT_TEMPLATE_ENGINES = frozenset({"good-badminton", "good-tennis"})


def list_integrations() -> list[dict]:
    """Return only serializable data, recomputing availability on each call."""

    return [integration.as_dict() for integration in INTEGRATIONS]


def get_integration(integration_id: str) -> dict | None:
    """Find an integration without importing its optional runtime."""

    return next(
        (integration.as_dict() for integration in INTEGRATIONS if integration.id == integration_id),
        None,
    )


def get_integration_definition(integration_id: str) -> Integration | None:
    """Return the internal definition for backend-only health checks."""

    return next((item for item in INTEGRATIONS if item.id == integration_id), None)


