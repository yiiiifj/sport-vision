"""
Sport Vision — 姿态分析模块
基于 MediaPipe PoseLandmarker (Tasks API) 的人体关键点检测与生物力学分析
"""

import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from pathlib import Path
from typing import Any, Optional, Sequence


class PoseAnalyzer:
    """封装 MediaPipe PoseLandmarker，提供关键点提取和生物力学分析"""

    # This is an application-side resource guard, not a claim about a
    # particular MediaPipe model's ability to find this many people.  A
    # detector still returns zero to ``num_poses`` observations for each
    # frame, depending on visibility, occlusion and the model runtime.
    MAX_CONFIGURED_POSES = 16
    MIN_CANDIDATE_CONFIDENCE = 0.30

    # 骨骼连接定义 (使用 PoseLandmarker 的索引)
    SKELETON_CONNECTIONS = [
        # 躯干
        (11, 12),  # 左肩-右肩
        (11, 23),  # 左肩-左髋
        (12, 24),  # 右肩-右髋
        (23, 24),  # 左髋-右髋
        # 左臂
        (11, 13),  # 左肩-左肘
        (13, 15),  # 左肘-左腕
        # 右臂
        (12, 14),  # 右肩-右肘
        (14, 16),  # 右肘-右腕
        # 左腿
        (23, 25),  # 左髋-左膝
        (25, 27),  # 左膝-左踝
        # 右腿
        (24, 26),  # 右髋-右膝
        (26, 28),  # 右膝-右踝
    ]

    # 关键点名称映射
    LANDMARK_NAMES = {
        0: "nose", 11: "left_shoulder", 12: "right_shoulder",
        13: "left_elbow", 14: "right_elbow",
        15: "left_wrist", 16: "right_wrist",
        23: "left_hip", 24: "right_hip",
        25: "left_knee", 26: "right_knee",
        27: "left_ankle", 28: "right_ankle",
    }

    # 要分析的关键关节角度
    JOINT_ANGLES = {
        "left_elbow": (11, 13, 15),
        "right_elbow": (12, 14, 16),
        "left_shoulder": (13, 11, 23),
        "right_shoulder": (14, 12, 24),
        "left_knee": (23, 25, 27),
        "right_knee": (24, 26, 28),
        "left_hip": (11, 23, 25),
        "right_hip": (12, 24, 26),
    }

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 history_size: int = 30,
                 num_poses: int = 1):

        if isinstance(num_poses, bool) or not isinstance(num_poses, int):
            raise ValueError("num_poses must be an integer")
        if not 1 <= num_poses <= self.MAX_CONFIGURED_POSES:
            raise ValueError(
                f"num_poses must be between 1 and {self.MAX_CONFIGURED_POSES}"
            )
        self.num_poses = num_poses

        # 查找模型文件
        model_path = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. "
                "Download it with: curl -sL -o models/pose_landmarker_lite.task "
                "\"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task\""
            )

        # 配置 PoseLandmarker
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            # VIDEO mode lets MediaPipe preserve temporal tracking state.  It
            # is materially more stable than treating every match frame as an
            # unrelated image, while still running entirely offline.
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            num_poses=self.num_poses,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        self.history_size = history_size
        # 关键点历史记录（用于速度/加速度计算）
        self.keypoint_history: deque = deque(maxlen=history_size)
        # Image-space pelvis trajectory. A monocular image cannot establish a
        # physical centre of mass without calibration and anthropometrics.
        self.pelvis_center_history: deque = deque(maxlen=history_size * 2)
        self._smoothed_keypoints: dict[int, tuple[float, float]] = {}
        self.smoothing_alpha = 0.6
        self.frame_count = 0
        self._last_timestamp_ms = -1
        self._last_media_pipe_pose_count: Optional[int] = None
        self._last_usable_candidate_count: Optional[int] = None

    def process_frame(self, frame_rgb: np.ndarray, timestamp_ms: Optional[int] = None) -> Optional[dict]:
        """
        处理单帧，返回分析结果

        Returns:
            {
                "keypoints": [{x, y, z, visibility, name}, ...],
                "skeleton": [[p1_idx, p2_idx], ...],
                "joint_angles": {name: angle_degrees, ...},
                "biomechanics": {image-space proxy metrics},
                "pelvis_center": {x, y},
                "mean_keypoint_visibility": float,
                "confidence": float,  # legacy alias for mean_keypoint_visibility
            }
        """
        h, w = frame_rgb.shape[:2]
        landmark_sets = self._detect_landmark_sets(frame_rgb, timestamp_ms)
        if not landmark_sets:
            return None

        # This is the legacy single-person API.  With ``num_poses > 1`` it
        # intentionally preserves its historical first-MediaPipe-result
        # behaviour; it does not guess which athlete is primary.  Use
        # MultiPersonPoseAnalyzer for explicit, stable track selection.
        landmarks = landmark_sets[0]

        # 1. 提取关键点（像素坐标）
        keypoints = []
        for idx in self.LANDMARK_NAMES:
            if idx < len(landmarks):
                lm = landmarks[idx]
                keypoints.append({
                    "id": idx,
                    "name": self.LANDMARK_NAMES[idx],
                    "x": lm.x * w,
                    "y": lm.y * h,
                    "z": lm.z,
                    "visibility": lm.visibility if lm.visibility is not None else 0.5,
                })

        if not keypoints:
            return None

        # 过滤低置信度
        avg_visibility = np.mean([kp["visibility"] for kp in keypoints])
        if avg_visibility < 0.3:
            return None
        keypoints = self._smooth_keypoint_positions(keypoints)

        # 2. 计算关节角度
        joint_angles = {}
        for name, (a, b, c) in self.JOINT_ANGLES.items():
            angle = self._calculate_angle_from_landmarks(landmarks, a, b, c)
            if angle is not None:
                joint_angles[name] = round(angle, 1)

        # 3. Pelvis midpoint, explicitly kept in image coordinates.
        point_map = {point["id"]: point for point in keypoints}
        if 23 in point_map and 24 in point_map:
            com_x = (point_map[23]["x"] + point_map[24]["x"]) / 2
            com_y = (point_map[23]["y"] + point_map[24]["y"]) / 2
        else:
            com_x, com_y = w / 2, h / 2
        pelvis_center = {"x": round(com_x, 1), "y": round(com_y, 1)}
        self.pelvis_center_history.append(pelvis_center)

        # 4. 记录关键点历史
        kp_dict = {kp["id"]: (kp["x"], kp["y"]) for kp in keypoints}
        self.keypoint_history.append(kp_dict)
        self.frame_count += 1

        # 5. 生物力学分析
        biomechanics = self._analyze_biomechanics(landmarks, w, h)

        return {
            "keypoints": keypoints,
            "skeleton": self.SKELETON_CONNECTIONS,
            "joint_angles": joint_angles,
            "biomechanics": biomechanics,
            "pelvis_center": pelvis_center,
            # Retain confidence for older consumers, but make the semantic
            # source explicit: this is mean landmark visibility, not an
            # independently calibrated detector probability.
            "mean_keypoint_visibility": round(avg_visibility, 2),
            "confidence": round(avg_visibility, 2),
        }

    def detect_pose_candidates(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return per-frame usable person candidates without choosing one.

        Unlike :meth:`process_frame`, this method does not apply the
        single-athlete EMA/history state.  It exposes every usable MediaPipe
        pose from the current frame, up to the configured ``num_poses`` limit,
        as a detector candidate suitable for :class:`backend.tracking.TrackManager`.

        Each candidate contains pixel-space ``center`` (pelvis midpoint when
        available), a positive pixel ``scale`` (visible landmark bounding-box
        height), average landmark ``confidence``, and named ``keypoints``.
        Returned candidates are *not* athlete identities; callers must track
        them across frames before selecting a primary athlete.
        """

        h, w = frame_rgb.shape[:2]
        landmark_sets = self._detect_landmark_sets(frame_rgb, timestamp_ms)
        candidates: list[dict[str, Any]] = []
        for candidate_index, landmarks in enumerate(landmark_sets):
            candidate = self.candidate_from_landmarks(
                landmarks,
                width=w,
                height=h,
                candidate_index=candidate_index,
                min_confidence=self.MIN_CANDIDATE_CONFIDENCE,
            )
            if candidate is not None:
                candidates.append(candidate)
        self._last_usable_candidate_count = len(candidates)
        return candidates

    def get_multi_pose_capabilities(self) -> dict[str, Any]:
        """Describe configured and observed multi-person detector limits.

        ``configured_candidate_limit`` is an input setting, not a promise
        that a particular frame contains that many valid people.  The last
        count is ``None`` until inference runs and otherwise distinguishes
        MediaPipe's raw observations from the candidates passing this
        application's visibility-quality gate.
        """

        return {
            "configured_candidate_limit": self.num_poses,
            "multi_person_configured": self.num_poses > 1,
            "last_media_pipe_pose_count": self._last_media_pipe_pose_count,
            "last_usable_candidate_count": self._last_usable_candidate_count,
            "candidate_confidence_threshold": self.MIN_CANDIDATE_CONFIDENCE,
            "limitations": [
                "MediaPipe can return fewer poses than the configured limit when people are occluded, small, or outside frame.",
                "Per-frame candidates are not persistent identities; use TrackManager before selecting an athlete.",
                "The application limit is a resource guard and not a calibrated crowd-counting guarantee.",
            ],
        }

    def _detect_landmark_sets(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: Optional[int],
    ) -> list[Sequence[Any]]:
        """Run one video-mode detector invocation and retain raw count."""

        if not isinstance(frame_rgb, np.ndarray) or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("frame_rgb must be an HxWx3 RGB numpy array")

        # 创建 MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # The Tasks video API requires strictly increasing timestamps.  Some
        # callers are live streams without a source timestamp, so supply a
        # deterministic 30 FPS clock and guard reused analyzers as well.
        requested_timestamp = timestamp_ms if timestamp_ms is not None else self._last_timestamp_ms + 34
        safe_timestamp = max(int(requested_timestamp), self._last_timestamp_ms + 1)
        result = self.landmarker.detect_for_video(mp_image, safe_timestamp)
        self._last_timestamp_ms = safe_timestamp
        landmark_sets = list(result.pose_landmarks or [])
        self._last_media_pipe_pose_count = len(landmark_sets)
        return landmark_sets

    @classmethod
    def candidate_from_landmarks(
        cls,
        landmarks: Sequence[Any],
        *,
        width: int,
        height: int,
        candidate_index: int,
        min_confidence: float = MIN_CANDIDATE_CONFIDENCE,
    ) -> Optional[dict[str, Any]]:
        """Build a JSON-safe detector candidate from one MediaPipe pose.

        This pure conversion helper deliberately accepts landmark-shaped test
        doubles, allowing its geometry and quality gates to be tested without
        loading a model or running inference.
        """

        if width <= 0 or height <= 0 or candidate_index < 0:
            raise ValueError("width, height and candidate_index must be positive/valid")
        if not landmarks:
            return None

        keypoints: list[dict[str, Any]] = []
        all_points: list[tuple[float, float, float]] = []
        for index, landmark in enumerate(landmarks):
            x_normalized = cls._finite_landmark_value(landmark, "x")
            y_normalized = cls._finite_landmark_value(landmark, "y")
            if x_normalized is None or y_normalized is None:
                continue
            visibility = cls._finite_landmark_value(landmark, "visibility")
            visibility = 0.5 if visibility is None else min(1.0, max(0.0, visibility))
            x = x_normalized * width
            y = y_normalized * height
            all_points.append((x, y, visibility))
            if index not in cls.LANDMARK_NAMES:
                continue
            z = cls._finite_landmark_value(landmark, "z")
            keypoints.append({
                "id": index,
                "name": cls.LANDMARK_NAMES[index],
                "x": x,
                "y": y,
                "z": 0.0 if z is None else z,
                "visibility": visibility,
            })

        if not keypoints:
            return None
        confidence = float(np.mean([keypoint["visibility"] for keypoint in keypoints]))
        if confidence < min_confidence:
            return None

        # Prefer spatially credible landmarks for the size proxy.  If all
        # points are low visibility, retain all finite points rather than
        # inventing a body box.
        bbox_points = [(x, y) for x, y, visibility in all_points if visibility >= min_confidence]
        if len(bbox_points) < 2:
            bbox_points = [(x, y) for x, y, _ in all_points]
        if len(bbox_points) < 2:
            return None
        left = min(point[0] for point in bbox_points)
        right = max(point[0] for point in bbox_points)
        top = min(point[1] for point in bbox_points)
        bottom = max(point[1] for point in bbox_points)
        bbox_width = max(0.0, right - left)
        bbox_height = max(0.0, bottom - top)
        # A strictly positive scale is required by TrackManager.  For an
        # unnaturally flat landmark set, width provides a conservative fallback
        # instead of sending a zero/NaN scale into association.
        scale = max(bbox_height, bbox_width, 1.0)

        point_map = {point["id"]: point for point in keypoints}
        if 23 in point_map and 24 in point_map:
            center_x = (point_map[23]["x"] + point_map[24]["x"]) / 2.0
            center_y = (point_map[23]["y"] + point_map[24]["y"]) / 2.0
            center_kind = "pelvis_midpoint_px"
        else:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            center_kind = "visible_landmark_bbox_center_px"

        return {
            "candidate_index": candidate_index,
            "center": {"x": center_x, "y": center_y},
            "center_kind": center_kind,
            "scale": scale,
            "scale_kind": "visible_landmark_bbox_max_extent_px",
            "confidence": confidence,
            "mean_keypoint_visibility": confidence,
            "keypoints": keypoints,
            "bbox": {
                "left": left,
                "top": top,
                "width": bbox_width,
                "height": bbox_height,
            },
        }

    @staticmethod
    def _finite_landmark_value(landmark: Any, name: str) -> Optional[float]:
        """Read finite numeric landmark fields from MediaPipe or test doubles."""

        try:
            value = float(getattr(landmark, name))
        except (AttributeError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _calculate_angle_from_landmarks(self, landmarks, a_idx, b_idx, c_idx) -> Optional[float]:
        """计算三个关键点形成的角度（以 b 为顶点）"""
        try:
            if a_idx >= len(landmarks) or b_idx >= len(landmarks) or c_idx >= len(landmarks):
                return None
            a = landmarks[a_idx]
            b = landmarks[b_idx]
            c = landmarks[c_idx]
            ba = np.array([a.x - b.x, a.y - b.y])
            bc = np.array([c.x - b.x, c.y - b.y])
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            return math.degrees(angle)
        except Exception:
            return None

    def _smooth_keypoint_positions(self, keypoints: list[dict]) -> list[dict]:
        """EMA-smooth image positions while retaining the raw landmark score."""

        smoothed = []
        for point in keypoints:
            previous = self._smoothed_keypoints.get(point["id"])
            x, y = point["x"], point["y"]
            if previous is not None:
                x = self.smoothing_alpha * x + (1 - self.smoothing_alpha) * previous[0]
                y = self.smoothing_alpha * y + (1 - self.smoothing_alpha) * previous[1]
            self._smoothed_keypoints[point["id"]] = (x, y)
            smoothed.append({**point, "x": x, "y": y})
        return smoothed

    def _analyze_biomechanics(self, landmarks, w: int, h: int) -> dict:
        """Image-space kinematic proxies, not calibrated clinical measurements."""
        result = {
            "wrist_speed_px_per_frame": 0.0,
            "image_spine_lean_deg": 0.0,
            "knee_flexion_proxy_deg": 0.0,
            "arm_extension_proxy_deg": 0.0,
            "image_levelness_proxy_pct": 0.0,
        }

        if len(self.keypoint_history) < 2:
            return result

        prev = self.keypoint_history[-2]
        curr = self.keypoint_history[-1]

        # 手腕速度（取左右手中速度更大的）
        wrist_speeds = []
        for wrist_id in [15, 16]:
            if wrist_id in prev and wrist_id in curr:
                dx = curr[wrist_id][0] - prev[wrist_id][0]
                dy = curr[wrist_id][1] - prev[wrist_id][1]
                speed = math.sqrt(dx * dx + dy * dy)
                wrist_speeds.append(speed)
        result["wrist_speed_px_per_frame"] = round(max(wrist_speeds) if wrist_speeds else 0, 1)

        # 身体倾斜角（脊柱与垂直线的夹角）
        try:
            if 11 < len(landmarks) and 12 < len(landmarks) and 23 < len(landmarks) and 24 < len(landmarks):
                mid_shoulder = np.array([
                    (landmarks[11].x + landmarks[12].x) / 2,
                    (landmarks[11].y + landmarks[12].y) / 2
                ])
                mid_hip = np.array([
                    (landmarks[23].x + landmarks[24].x) / 2,
                    (landmarks[23].y + landmarks[24].y) / 2
                ])
                spine = mid_shoulder - mid_hip
                vertical = np.array([0, -1])
                cos_angle = np.dot(spine, vertical) / (np.linalg.norm(spine) + 1e-8)
                lean_angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
                result["image_spine_lean_deg"] = round(lean_angle, 1)
        except Exception:
            pass

        # 膝盖弯曲度（取双膝平均）
        knee_angles = []
        for name in ["left_knee", "right_knee"]:
            a_id, b_id, c_id = self.JOINT_ANGLES[name]
            angle = self._calculate_angle_from_landmarks(landmarks, a_id, b_id, c_id)
            if angle is not None:
                knee_angles.append(angle)
        result["knee_flexion_proxy_deg"] = round(180 - np.mean(knee_angles) if knee_angles else 0, 1)

        # 手臂伸展度（肘部角度，越接近 180 越伸展）
        elbow_angles = []
        for name in ["left_elbow", "right_elbow"]:
            a_id, b_id, c_id = self.JOINT_ANGLES[name]
            angle = self._calculate_angle_from_landmarks(landmarks, a_id, b_id, c_id)
            if angle is not None:
                elbow_angles.append(angle)
        result["arm_extension_proxy_deg"] = round(np.mean(elbow_angles) if elbow_angles else 0, 1)

        # Image levelness proxy (0-100). It is deliberately not called a
        # symmetry score because camera roll and side-on views affect it.
        try:
            if 11 < len(landmarks) and 12 < len(landmarks) and 23 < len(landmarks) and 24 < len(landmarks):
                left_shoulder_y = landmarks[11].y
                right_shoulder_y = landmarks[12].y
                left_hip_y = landmarks[23].y
                right_hip_y = landmarks[24].y
                shoulder_diff = abs(left_shoulder_y - right_shoulder_y)
                hip_diff = abs(left_hip_y - right_hip_y)
                asymmetry = (shoulder_diff + hip_diff) / 2
                levelness = max(0, 100 - asymmetry * 500)
                result["image_levelness_proxy_pct"] = round(levelness, 1)
        except Exception:
            result["image_levelness_proxy_pct"] = 0.0

        return result

    def get_trajectory(self) -> list:
        """Return the smoothed image-space pelvis trajectory."""
        return list(self.pelvis_center_history)

    def reset(self):
        """重置状态"""
        self.keypoint_history.clear()
        self.pelvis_center_history.clear()
        self._smoothed_keypoints.clear()
        self.frame_count = 0

    def close(self):
        """释放资源"""
        self.landmarker.close()
