"""
Sport Vision — FastAPI 应用入口
提供 REST API + WebSocket 实时分析流
"""

import json
import os
import re
import uuid
import asyncio
import base64
import binascii
from contextlib import suppress
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.pipeline import Pipeline
from backend.integrations import get_integration, list_integrations
from backend.jobs import JobRequest, manager as job_manager, preflight
from backend.uploads import ALLOWED_EXTENSIONS, UPLOAD_DIR, create_record, discard_record, resolve_upload, save_record
from backend.calibration import CalibrationError, calibration_response

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DEMO_DIR = BASE_DIR / "demo_videos"
DEMO_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
MAX_CAMERA_FRAME_BYTES = 1_000_000
ANALYSIS_MODES = frozenset({"generic", "badminton", "tennis"})

UPLOAD_DIR.mkdir(exist_ok=True)
DEMO_DIR.mkdir(exist_ok=True)

# FastAPI 应用
app = FastAPI(title="Sport Vision", version="1.1.0")

allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "SPORT_VISION_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 活跃的处理流水线
active_pipelines: dict[str, Pipeline] = {}


def decode_camera_frame(encoded_image: object) -> np.ndarray:
    """Decode one bounded JPEG/WebP camera frame without persisting it."""

    if not isinstance(encoded_image, str) or not encoded_image:
        raise ValueError("Camera frame is required")
    # Base64 increases size by about a third; reject before allocating a large
    # decoded buffer. Browser capture sends only the bare data payload.
    if len(encoded_image) > (MAX_CAMERA_FRAME_BYTES * 4 // 3) + 8:
        raise ValueError("Camera frame exceeds 1 MB limit")
    try:
        payload = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Camera frame is not valid base64") from error
    if not payload or len(payload) > MAX_CAMERA_FRAME_BYTES:
        raise ValueError("Camera frame exceeds 1 MB limit")
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape[0] < 16 or frame.shape[1] < 16:
        raise ValueError("Camera frame is not a supported image")
    return frame


@app.on_event("shutdown")
async def shutdown_portal() -> None:
    """Release pose sessions and portal-owned engine processes on shutdown."""

    for pipeline in list(active_pipelines.values()):
        try:
            pipeline.close()
        except Exception:
            # Shutdown must keep releasing the remaining local resources.
            pass
    active_pipelines.clear()
    await asyncio.to_thread(job_manager.shutdown)


async def _stream_pipeline(websocket: WebSocket, session_id: str, pipeline: Pipeline, video_path: str) -> None:
    """Send video frames in a task so the owner can still receive stop messages."""

    try:
        async for result in pipeline.process_video(video_path, target_fps=20, skip_frames=1):
            if "error" in result:
                await websocket.send_json({"type": "error", "message": result["error"]})
                return
            if pipeline.stop_requested:
                return
            await websocket.send_json({"type": "frame", "data": result})
        if not pipeline.stop_requested:
            await websocket.send_json({"type": "complete", "session_id": session_id})
    except asyncio.CancelledError:
        pipeline.stop()
        raise
    except Exception:
        await websocket.send_json({"type": "error", "message": "Analysis stream failed"})
    finally:
        pipeline.close()
        active_pipelines.pop(session_id, None)


# ============ REST API ============

@app.get("/api/health")
async def health() -> dict:
    """Provide a local readiness signal without leaking filesystem details."""

    integrations = list_integrations()
    return {
        "status": "ok",
        "service": "sport-vision",
        "version": app.version,
        "integrations": {
            "available": sum(item["status"] == "available" for item in integrations),
            "total": len(integrations),
        },
        "jobs": job_manager.summary(),
    }

@app.get("/api/integrations")
async def integrations():
    """Catalogue every local sports-analysis engine behind this frontend.

    The engines deliberately are not imported here: several have incompatible
    model runtimes.  This endpoint lets the unified UI safely discover which
    source projects are present without eagerly loading their ML dependencies.
    """
    items = list_integrations()
    return {
        "integrations": items,
        "available": sum(item["status"] == "available" for item in items),
        "total": len(items),
    }


@app.get("/api/integrations/{integration_id}")
async def integration_detail(integration_id: str):
    """Return capabilities and runtime information for one registered engine."""
    integration = get_integration(integration_id)
    if integration is None:
        return JSONResponse(status_code=404, content={"error": "integration not found"})
    return integration


@app.get("/api/integrations/{integration_id}/preflight")
async def integration_preflight(integration_id: str):
    """Check whether a managed engine has its local source, Python and known assets."""
    # Package probes launch a short-lived subprocess.  Keep that blocking work
    # off FastAPI's event loop so WebSocket stop messages remain responsive
    # while a user opens the Engine Hub.
    return await asyncio.to_thread(preflight, integration_id)


@app.post("/api/jobs")
async def create_job(request: JobRequest):
    """Queue a supported specialist engine against files uploaded to this portal."""
    # ``submit`` validates an upload manifest and may run a preflight probe.
    # Do it in a worker thread rather than pausing live streaming clients.
    return await asyncio.to_thread(job_manager.submit, request)


@app.post("/api/calibration/validate")
async def validate_calibration(payload: dict):
    """Validate court-plane point pairs before any metric coordinate is used.

    This endpoint only estimates a homography; it never converts a pose metric
    into metres by itself.  Consumers must check ``metric_enabled`` in the
    response and persist the accepted calibration with their experiment.
    """

    try:
        return calibration_response(payload)
    except CalibrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    """Read a queued/running/completed job and its most recent engine output."""
    return job_manager.public(job_id)


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a queued job or request termination of the active engine process."""
    return job_manager.cancel(job_id)


@app.get("/api/jobs/{job_id}/files")
async def job_files(job_id: str):
    """List output artifacts.  Files remain in the dedicated job directory."""
    return {"files": job_manager.files(job_id)}


@app.get("/api/jobs/{job_id}/files/{filename:path}")
async def serve_job_file(job_id: str, filename: str):
    """Serve an artifact only when it resolves inside its job directory."""
    job_manager.public(job_id)
    output_dir = (BASE_DIR / "outputs" / job_id).resolve()
    target = (output_dir / filename).resolve()
    if output_dir not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")
    headers = {"X-Content-Type-Options": "nosniff"}
    # Specialist engines can generate arbitrary output types.  Treat every
    # artifact as a download rather than executing/rendering it in the
    # portal's origin; frontend fetches can still read JSON evidence safely.
    return FileResponse(
        target,
        filename=target.name,
        content_disposition_type="attachment",
        headers=headers,
    )

@app.get("/api/demos")
async def list_demos():
    """列出所有可用的 Demo 视频"""
    demos = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.webm"]:
        for f in DEMO_DIR.glob(ext):
            demos.append({
                "id": f.stem,
                "name": f.stem.replace("_", " ").replace("-", " ").title(),
                "filename": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
            })
    return {"demos": demos}


def _validate_uploaded_media(path: Path, kind: str) -> bool:
    """Reject a renamed text/binary file before it enters an analysis queue."""

    if kind == "video":
        capture = cv2.VideoCapture(str(path))
        try:
            ok, _ = capture.read()
            return bool(capture.isOpened() and ok)
        finally:
            capture.release()
    return cv2.imread(str(path), cv2.IMREAD_COLOR) is not None


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), kind: str = Form("video")):
    """Upload a video or a court-template image for a managed job."""
    if kind not in ALLOWED_EXTENSIONS:
        return JSONResponse(status_code=400, content={"error": "Unsupported upload kind"})
    allowed = ALLOWED_EXTENSIONS[kind]
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported format: {ext}. Allowed: {allowed}"}
        )

    max_size = 2 * 1024 * 1024 * 1024 if kind == "video" else 50 * 1024 * 1024
    file_id = uuid.uuid4().hex
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    record = create_record(file_id, kind, ext, filename, 0)
    size = 0
    try:
        with open(save_path, "xb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_size:
                    raise ValueError("File exceeds the upload size limit")
                f.write(chunk)
        # OpenCV decoding can be noticeably slow for a large video.  The
        # upload endpoint is asynchronous, so keep validation off the event
        # loop and retain the same strict content check.
        if not await asyncio.to_thread(_validate_uploaded_media, save_path, kind):
            raise ValueError("File contents do not match the selected media type")
        record = create_record(file_id, kind, ext, filename, size)
        save_record(record)
    except (OSError, ValueError) as exc:
        discard_record(record)
        status_code = 413 if isinstance(exc, ValueError) else 500
        return JSONResponse(status_code=status_code, content={"error": str(exc)})
    finally:
        await file.close()

    return {
        "id": file_id,
        "filename": file.filename,
        "kind": kind,
        "size_mb": round(size / (1024 * 1024), 1),
    }


# ============ WebSocket ============

@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """
    WebSocket 实时分析端点

    客户端发送:
        {"type": "start", "source": "demo", "id": "badminton_rally"}
        {"type": "start", "source": "upload", "id": "portal-upload-id"}
        {"type": "start", "source": "camera"}
        {"type": "camera_frame", "image": "base64-jpeg", "timestamp_ms": 123}
        {"type": "stop"}

    Camera frames are decoded in memory only, limited to 1 MB, and the browser
    sends the next frame only after receiving its prior result. This keeps the
    local real-time path bounded without buffering a webcam recording.

    服务端推送:
        {"type": "frame", "data": {...}}
        {"type": "complete", "summary": {...}}
        {"type": "error", "message": "..."}
    """
    await websocket.accept()
    session_id = uuid.uuid4().hex
    pipeline: Optional[Pipeline] = None
    stream_task: asyncio.Task | None = None
    source: str | None = None
    camera_frame_number = 0
    last_camera_timestamp_ms = -1

    async def stop_stream() -> None:
        nonlocal stream_task, pipeline, source, camera_frame_number, last_camera_timestamp_ms
        current_pipeline = pipeline
        if current_pipeline is not None:
            current_pipeline.stop()
        if stream_task is not None and not stream_task.done():
            stream_task.cancel()
            with suppress(asyncio.CancelledError):
                await stream_task
        elif current_pipeline is not None:
            # File-stream tasks close themselves in _stream_pipeline.finally.
            # Camera sessions have no producer task, so release MediaPipe here.
            current_pipeline.close()
        stream_task = None
        pipeline = None
        source = None
        camera_frame_number = 0
        last_camera_timestamp_ms = -1

    try:
        while True:
            # 接收客户端消息
            msg = await websocket.receive_text()
            data = json.loads(msg)

            if data.get("type") == "start":
                # Stop and await the previous stream before a new request uses
                # this WebSocket. This prevents interleaved frame messages.
                await stop_stream()

                source = data.get("source")
                # 确定视频路径
                video_path = None
                if source == "demo":
                    demo_id = data.get("id", "")
                    if not isinstance(demo_id, str) or not DEMO_ID.fullmatch(demo_id):
                        await websocket.send_json({"type": "error", "message": "Invalid demo identifier"})
                        continue
                    # 搜索 demo 视频
                    for ext in [".mp4", ".avi", ".mov", ".webm"]:
                        candidate = DEMO_DIR / f"{demo_id}{ext}"
                        if candidate.exists():
                            video_path = str(candidate)
                            break
                elif source == "upload":
                    upload_id = data.get("id", "")
                    if not isinstance(upload_id, str):
                        await websocket.send_json({"type": "error", "message": "Invalid upload identifier"})
                        continue
                    try:
                        video_path = str(resolve_upload(upload_id, "video"))
                    except HTTPException:
                        await websocket.send_json({"type": "error", "message": "Uploaded video not found"})
                        continue
                elif source != "camera":
                    await websocket.send_json({"type": "error", "message": "Unsupported video source"})
                    continue

                if source != "camera" and (not video_path or not Path(video_path).exists()):
                    await websocket.send_json({
                        "type": "error",
                        "message": "Requested video is unavailable",
                    })
                    continue

                # Create an independent producer task.  The receive loop stays
                # responsive so the browser's Stop button is meaningful.
                requested_sport = data.get("sport", "generic")
                analysis_mode = requested_sport if requested_sport in ANALYSIS_MODES else "generic"
                pipeline = Pipeline(analysis_mode=analysis_mode)
                active_pipelines[session_id] = pipeline

                await websocket.send_json({
                    "type": "started",
                    "session_id": session_id,
                    "source": source,
                })

                if source != "camera":
                    stream_task = asyncio.create_task(
                        _stream_pipeline(websocket, session_id, pipeline, video_path),
                        name=f"sport-vision-stream-{session_id}",
                    )

            elif data.get("type") == "camera_frame":
                if source != "camera" or pipeline is None:
                    await websocket.send_json({"type": "error", "message": "Camera session is not active"})
                    continue
                timestamp_ms = data.get("timestamp_ms")
                if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int) or timestamp_ms < 0:
                    await websocket.send_json({"type": "error", "message": "Invalid camera timestamp"})
                    continue
                # Browser timers may share a millisecond on very fast devices;
                # MediaPipe VIDEO mode requires strictly increasing timestamps.
                timestamp_ms = max(timestamp_ms, last_camera_timestamp_ms + 1)
                try:
                    camera_frame = await asyncio.to_thread(decode_camera_frame, data.get("image"))
                    camera_frame_number += 1
                    result = await asyncio.to_thread(
                        pipeline.process_camera_frame,
                        camera_frame,
                        timestamp_ms,
                        camera_frame_number,
                    )
                    last_camera_timestamp_ms = timestamp_ms
                except ValueError as error:
                    await websocket.send_json({"type": "error", "message": str(error)})
                    continue
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Camera frame analysis failed"})
                    continue
                if "error" in result:
                    await websocket.send_json({"type": "error", "message": result["error"]})
                    continue
                await websocket.send_json({"type": "frame", "data": result})

            elif data.get("type") == "stop":
                if pipeline or stream_task:
                    await stop_stream()
                    await websocket.send_json({
                        "type": "stopped",
                        "session_id": session_id,
                    })

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.send_json({"type": "error", "message": "Invalid analysis session request"})
        except Exception:
            pass
    finally:
        await stop_stream()
        active_pipelines.pop(session_id, None)


# ============ 静态文件 ============

# 前端静态文件
app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

# Demo 视频访问
@app.get("/demo_videos/{filename}")
async def serve_demo_video(filename: str):
    path = (DEMO_DIR / filename).resolve()
    if DEMO_DIR.resolve() in path.parents and path.is_file():
        return FileResponse(path)
    return JSONResponse(status_code=404, content={"error": "not found"})

# 首页
@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
