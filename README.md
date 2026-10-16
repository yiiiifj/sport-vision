<div align="center">

# ⚡ Sport Vision

**Unified AI-powered sports video and performance analysis portal**

实时骨骼追踪 · 击球动作识别 · 运动生物力学分析

[![License: MIT](https://img.shields.io/badge/License-MIT-cyan.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-PoseLandmarker-green.svg)](https://ai.google.dev/edge/mediapipe)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🦴 **Real-time Pose Tracking** | MediaPipe PoseLandmarker with 13 keypoints |
| 🎯 **Action Recognition** | Serve, Smash, Forehand, Backhand, Lob, Drop |
| 📐 **Biomechanics Proxies** | Image-space joint angles, wrist motion and body lean |
| 🔥 **Movement Heatmap** | Spatial visualization of player movement |
| ⚡ **Action Timeline** | Sequential recording of all detected actions |
| 🌐 **Real-time WebSocket** | Streaming analysis at ~20 FPS |
| 🎨 **Stunning UI** | Dark neon theme with particle animation |

> **Zero cloud dependency** — All AI inference runs locally on CPU. No API keys needed.

## 🧩 Unified workspace integration

`sport-vision` is now the primary frontend for the complete `D:\AIC` sports
workspace.  Its Engine Hub dynamically checks the local source folders through
`GET /api/integrations` and groups their capabilities by sport.  Each engine
keeps its own runtime instead of being imported into the FastAPI process, which
prevents incompatible PaddlePaddle, PyTorch, Gradio, Streamlit and Node.js
dependencies from breaking the real-time Sport Vision pipeline.

| Sport / capability | Integrated source projects |
|---|---|
| Racket sports | Good Badminton, Good Tennis, Sport Vision |
| Basketball | OpenHoopVision |
| Football video | Football Analytics |
| Football intelligence | TactiQ AI, Football Intelligence Hub |
| General biomechanics | SportIQ, Sports2D, AthletePose3D |
| General video AI | PaddleVideo |

The portal reports only locally verified source availability; it does not claim
that an engine's optional models or third-party dependencies are installed.
Start a specialised engine from its own project environment when its deeper
workflow is required. This keeps results reproducible and preserves each
project's license and dependency boundary.

### Managed jobs

The Engine Hub can submit uploaded videos to Good Badminton, Good Tennis,
OpenHoopVision, Football Analytics, SportIQ, and Sports2D. Jobs run one at a
time, expose their current state and recent logs through `GET /api/jobs/{job_id}`
and save artifacts in `outputs/<job_id>/`. Badminton and tennis also require a
court-template image. Configure a project's own Python environment with an
optional variable such as `SPORT_VISION_GOOD_TENNIS_PYTHON`; otherwise the
Sport Vision Python interpreter is used. Projects requiring match IDs, LLM
credentials, training configuration or datasets are shown in the hub but are
not incorrectly presented as generic video jobs.

For the racket engines, the specialised shuttlecock/tennis-ball checkpoints
must come from the authors' release assets; generic YOLO pose and person models
are requested by their official model names and may be fetched by Ultralytics.
OpenHoopVision can run without its jersey-number OCR checkpoint. SportIQ uses a
local MediaPipe biomechanics fallback because the checkpoint URL in its README
does not currently resolve to a published release asset.

When a managed engine's specialist runtime is incomplete, Sport Vision runs a
clearly labelled local rules engine instead of silently failing. It uses the
checked-in MediaPipe pose model plus transparent kinematic rules and creates:

- `summary.json` — pose coverage, joint-angle and execution-method summary
- `frame-metrics.csv` — frame-level movement, wrist, hip, ankle and image-space proxy metrics
- `events.json` — rule-based actions or movement cues with frame/time and a labelled heuristic score
- `local-analysis.mp4` — an auditable video with the detected skeleton and rule state
- `run-manifest.json` — input/model hashes, rule version and measurement scope
- `artifact-manifest.json` — output hashes, sizes and generated-video probe results

The local engine does **not** claim ball/shuttle tracking, jersey recognition,
court calibration or learned event detection. Its job response is marked
`execution_mode: "local_rules"`; downstream reports can therefore distinguish
it from a ready specialist engine.

### Measurement and safety boundary

The portal is local-only by default (`127.0.0.1`). Uploads use opaque UUIDs,
are validated as the declared media type, and cannot be exchanged for arbitrary
filesystem paths. Managed jobs have a bounded queue, execution timeout,
subprocess-tree cancellation and an artifact-integrity manifest. These controls
make local experimentation safer; they are not a replacement for user accounts,
tenant isolation or a production deployment review.

Image-space pelvic trajectories, lean, levelness and joint values are labelled
as image-space proxies. They are not calibrated centre-of-mass, symmetry,
metres/second or clinical measurements. Rule-event scores are rule-match
scores, not learned probabilities. See the [competition evaluation protocol](docs/COMPETITION_PROTOCOL.md)
before making accuracy or award-level claims.

Before a job is enabled, the portal checks the source entrypoint, configured
Python executable, required import packages and known model files. Uploads are streamed to disk (up to
2 GB for video and 50 MB for a template) rather than buffered in memory. A
queued or running task can be cancelled with `DELETE /api/jobs/{job_id}`.

### Calibration and evidence workflow

`POST /api/calibration/validate` accepts four or more image/court point pairs
and estimates a RANSAC plane homography. A result only sets
`metric_enabled: true` when the supplied court coordinates use a recognized
linear unit and all inlier, reprojection-error and conditioning gates pass.
It remains a court-plane transform: do not apply it to airborne body points.

```json
{
  "image_points": [[318, 91], [622, 120], [690, 626], [228, 588]],
  "court_points": [[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]],
  "court_unit": "m"
}
```

For a frozen, independently annotated test split, use the built-in evaluator
instead of manually selecting favorable clips:

```bash
# Class-aware event F1; commit the tolerance before final test evaluation.
python -m backend.evaluation events \
  --predicted outputs/<job-id>/events.json \
  --ground-truth benchmarks/events-test.json \
  --tolerance-frames 3 --output benchmark-results/<job-id>-event-evaluation.json

# 2-D pose PCK and error; truth frames declare normalization_scale_px.
python -m backend.evaluation pose-pck \
  --predicted benchmarks/pose-predictions.json \
  --ground-truth benchmarks/pose-test.json

# Confirm generated output has not changed since the job completed.
python -m backend.evaluation verify-artifacts --output-dir outputs/<job-id>

# Reject train/validation/test clip overlap before tuning or reporting results.
python -m backend.evaluation validate-splits --manifest benchmarks/split-manifest.json

# Assemble a hash-linked experiment report from frozen split, job outputs and
# individual metric receipts. It exits non-zero if any declared evidence fails.
python -m backend.evaluation experiment-report \
  --manifest benchmarks/experiment-manifest.json \
  --output benchmark-results/experiment-evidence.json
```

The evaluator deliberately does not call a lightweight association result
HOTA, IDF1 or MOTA. Run the official [TrackEval](https://github.com/JonathonLuiten/TrackEval)
workflow on benchmark-formatted multi-person annotations for those tracking
claims. The full annotation and quality-gate contract is in
[docs/COMPETITION_PROTOCOL.md](docs/COMPETITION_PROTOCOL.md); reusable JSON
schemas and commands are in the [benchmark starter kit](benchmarks/README.md).

### Local retention maintenance

Uploads and job artifacts are never auto-deleted by the server. Review an
explicit dry-run plan first; it only recognises UUID-named portal uploads and
job folders, never arbitrary files or the state database:

```bash
python -m backend.retention --older-than-hours 168
```

Only after reviewing that JSON plan should a local operator add `--apply`.
Keep experiment outputs until their artifact manifests and evaluation reports
have been archived.

`GET /api/health` is a path-free local readiness check: it reports the portal
version, integration count and aggregate job states, suitable for a launch
script or competition demo checklist.

### Multi-person foundation

The repository now includes an opt-in MediaPipe candidate layer and deterministic
short-term `TrackManager`. It requires an explicit `primary_track_id` rather
than switching athletes based on detector order or apparent size. This is a
foundation API, not an unvalidated re-identification claim; see the
[multi-person tracking contract](docs/MULTI_PERSON_TRACKING.md) for use and
benchmark gates.

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/MindDock/sport-vision.git
cd sport-vision

# One-click start (creates venv, installs deps, downloads model, launches server)
chmod +x run.sh
./run.sh

# Open browser
open http://localhost:8000
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or reproduce the direct dependency versions used in the validated local run
pip install -r requirements-verified.txt

# Download pose model
mkdir -p models
curl -sL -o models/pose_landmarker_lite.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

# Start server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

## 🎬 Usage

1. **Open camera recognition** — Click “打开摄像头识别”, allow the browser prompt, and see local real-time pose and rule analysis. Frames are processed in memory, not saved as a camera recording.
2. **Upload a video** — Click the upload button and select any sports, fitness or training video. The default mode uses sport-neutral motion candidates.
3. **Use demo videos** — Place `.mp4` files in the `demo_videos/` directory.
4. **Choose a specialty only when appropriate** — “全部运动” is the default. Switch to badminton or tennis only when their racket-motion rules are relevant. “规则匹配度” is a heuristic score, not a learned probability.

Camera access requires a browser-secure context. `http://127.0.0.1:8000` and
`http://localhost:8000` are treated as secure local contexts by mainstream
browsers; a LAN deployment should use HTTPS before requesting camera access.

### Full-stack free deployment

The repository includes a `Dockerfile` and `render.yaml` for a full FastAPI
deployment rather than a static-site substitute. It retains the upload,
WebSocket, camera, pose and managed-job endpoints. Render's free web-service
plan is suitable for demonstrations, but it sleeps after inactivity and has an
ephemeral filesystem; do not rely on it to retain uploads, jobs or experiment
evidence after a restart. Connect the repository in Render and create the
Blueprint from `render.yaml`; the health check is `/api/health`.

## 🏗️ Architecture

```
Video Input
    ↓
┌─────────────────────────────────────┐
│  FastAPI Backend                    │
│  ┌──────────────┐                   │
│  │ MediaPipe     │→ 13 Keypoints    │
│  │ PoseLandmarker│                  │
│  └──────────────┘                   │
│         ↓                           │
│  ┌──────────────┐  ┌─────────────┐  │
│  │ Action        │  │ Biomechanics│  │
│  │ Recognizer    │  │ Analyzer    │  │
│  └──────────────┘  └─────────────┘  │
│         ↓                ↓          │
│     WebSocket (JSON + Base64 Frame) │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Frontend (Canvas + Dashboard)      │
│  Skeleton Overlay · Gauges · Heatmap│
└─────────────────────────────────────┘
```

## 📁 Project Structure

```
sport-vision/
├── backend/
│   ├── main.py              # FastAPI server + WebSocket
│   ├── pipeline.py           # Video processing pipeline
│   ├── pose_analyzer.py      # MediaPipe pose estimation + biomechanics
│   ├── action_recognizer.py  # Rule-based action recognition engine
│   ├── calibration.py        # RANSAC court-plane quality gates
│   ├── evaluation.py         # Frozen-set metrics and artifact verification
│   ├── retention.py          # Explicit dry-run storage maintenance
│   ├── tracking.py           # Deterministic multi-person association primitive
│   └── visualizer.py         # OpenCV skeleton rendering
├── frontend/
│   ├── index.html            # Single-page application
│   ├── css/style.css         # Dark neon design system
│   └── js/
│       ├── app.js            # WebSocket client + UI orchestration
│       ├── dashboard.js      # Gauges, stats, heatmap rendering
│       ├── skeleton-renderer.js  # Canvas skeleton overlay
│       └── particles.js      # Background particle animation
├── models/                   # MediaPipe model (auto-downloaded)
├── demo_videos/              # Place your .mp4 files here
├── requirements.txt
├── run.sh                    # One-click launcher
├── LICENSE
└── README.md
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Pose Estimation | [MediaPipe PoseLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) |
| Action Recognition | Rule-based keypoint temporal analysis |
| Video Processing | OpenCV |
| Backend | FastAPI + WebSocket |
| Frontend | Vanilla JS + Canvas API |
| Styling | CSS (dark neon + glassmorphism) |

For the validated dependency snapshot and competition archive guidance, see
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

## 🤝 Contributing

Contributions are welcome! Some ideas:

- [ ] **Improve action recognition** — Add ML-based classifier (ST-GCN)
- [ ] **Multi-person tracking** — ByteTrack / BoT-SORT integration
- [ ] **3D pose lifting** — MotionBERT for 2D→3D reconstruction
- [ ] **Action quality assessment** — Score technique quality
- [ ] **Mobile deployment** — CoreML / TFLite export

## 📄 License

[MIT](LICENSE) © [MindDock](https://github.com/MindDock)
