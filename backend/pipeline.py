"""
Sport Vision — CV 处理流水线
串联姿态分析、动作识别、可视化的核心流水线
"""

import cv2
import base64
import numpy as np
import time
from typing import Optional, AsyncGenerator

from backend.pose_analyzer import PoseAnalyzer
from backend.action_recognizer import ActionRecognizer
from backend.visualizer import Visualizer


class Pipeline:
    """视频分析流水线"""

    def __init__(self, analysis_mode: str = "generic"):
        self.pose_analyzer = PoseAnalyzer()
        self.action_recognizer = ActionRecognizer(mode=analysis_mode)
        self.visualizer = Visualizer()
        self.is_running = False
        self.stop_requested = False

    def _begin_session(self) -> None:
        """Reset state shared by file and browser-camera analysis sessions."""

        self.is_running = True
        self.stop_requested = False
        self.pose_analyzer.reset()
        self.action_recognizer.reset()
        self.visualizer.reset()

    def _process_bgr_frame(
        self,
        frame: np.ndarray,
        *,
        timestamp_ms: int,
        frame_number: int,
        total_frames: int,
        source: str,
    ) -> dict:
        """Run one BGR frame through the local pipeline and return safe JSON."""

        frame_height, frame_width = frame.shape[:2]
        max_width = 960
        if frame_width > max_width:
            scale = max_width / frame_width
            target_w = max_width
            target_h = int(frame_height * scale)
            frame = cv2.resize(frame, (target_w, target_h))
        else:
            target_w, target_h = frame_width, frame_height

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = self.pose_analyzer.process_frame(frame_rgb, timestamp_ms)
        action_result = None
        if pose_result:
            action_result = self.action_recognizer.update(
                pose_result["keypoints"], pose_result["joint_angles"]
            )
        rendered = self.visualizer.render_frame(frame, pose_result, action_result)
        encoded, buffer = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not encoded:
            return {"error": "Could not encode analysis frame"}
        return {
            "frame_base64": base64.b64encode(buffer).decode("utf-8"),
            "frame_number": frame_number,
            "total_frames": total_frames,
            "width": target_w,
            "height": target_h,
            "pose": self._sanitize_pose(pose_result),
            "action": action_result,
            "progress": round(frame_number / total_frames, 3) if total_frames > 0 else None,
            "heatmap_data": self.pose_analyzer.get_trajectory(),
            "source": source,
        }

    def process_camera_frame(self, frame: np.ndarray, timestamp_ms: int, frame_number: int) -> dict:
        """Process one browser-camera image without writing the source frame to disk."""

        if frame.ndim != 3 or frame.shape[2] != 3:
            return {"error": "Invalid camera image"}
        if not self.is_running:
            self._begin_session()
        return self._process_bgr_frame(
            frame,
            timestamp_ms=timestamp_ms,
            frame_number=frame_number,
            total_frames=0,
            source="camera",
        )

    async def process_video(self, video_path: str,
                            target_fps: int = 24,
                            skip_frames: int = 1) -> AsyncGenerator[dict, None]:
        """
        处理视频并逐帧 yield 分析结果（异步生成器）

        Yields:
            {
                "frame_base64": str,       # 渲染后的帧（JPEG base64）
                "frame_number": int,
                "total_frames": int,
                "fps": float,
                "pose": {...} or None,     # 姿态分析结果
                "action": {...} or None,   # 动作识别结果
                "progress": float,         # 0.0 ~ 1.0
                "heatmap_data": [...],     # 热力图数据点
            }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield {"error": "Cannot open requested video"}
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self._begin_session()

        frame_count = 0
        # 帧间隔控制
        frame_interval = 1.0 / target_fps

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % skip_frames != 0:
                    continue

                start_time = time.time()

                timestamp_ms = round((frame_count - 1) * 1000 / video_fps)
                result = self._process_bgr_frame(
                    frame,
                    timestamp_ms=timestamp_ms,
                    frame_number=frame_count,
                    total_frames=total_frames,
                    source="video",
                )
                result["fps"] = round(video_fps, 1)
                yield result

                # 控制帧率
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    import asyncio
                    await asyncio.sleep(sleep_time)

        finally:
            cap.release()
            self.is_running = False

    def _sanitize_pose(self, pose_result: Optional[dict]) -> Optional[dict]:
        """清理姿态数据以便 JSON 序列化"""
        if not pose_result:
            return None
        # 移除大体积的骨骼连接信息（前端已有）
        return {
            "keypoints": pose_result["keypoints"],
            "joint_angles": pose_result["joint_angles"],
            "biomechanics": pose_result["biomechanics"],
            "pelvis_center": pose_result["pelvis_center"],
            "mean_keypoint_visibility": pose_result.get("mean_keypoint_visibility", pose_result["confidence"]),
            "confidence": pose_result["confidence"],
        }

    def stop(self):
        """停止处理"""
        self.stop_requested = True
        self.is_running = False

    def close(self):
        """释放所有资源"""
        self.stop()
        self.pose_analyzer.close()
