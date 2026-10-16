"""
Sport Vision — 动作识别模块
基于关键点时序分析的通用动作候选与拍类动作规则引擎
"""

import math
import numpy as np
from collections import deque
from typing import Optional


class ActionRecognizer:
    """
    规则引擎动作识别器
    默认识别通用的移动、抬臂与摆臂候选；仅在明确选择羽毛球或
    网球模式时启用拍类动作规则。所有输出都是可审计的规则候选。
    """

    # 支持识别的动作类型
    ACTIONS = {
        "serve": {"name": "发球 Serve", "icon": "🎯", "color": "#00f0ff"},
        "smash": {"name": "扣杀 Smash", "icon": "💥", "color": "#ff3366"},
        "forehand": {"name": "正手 Forehand", "icon": "➡️", "color": "#33ff88"},
        "backhand": {"name": "反手 Backhand", "icon": "⬅️", "color": "#ffaa33"},
        "lob": {"name": "挑球 Lob", "icon": "🌈", "color": "#aa66ff"},
        "drop": {"name": "吊球 Drop", "icon": "🪶", "color": "#66ddff"},
        "ready": {"name": "准备 Ready", "icon": "🧍", "color": "#888888"},
        "moving": {"name": "移动 Moving", "icon": "🏃", "color": "#ffdd44"},
    }
    GENERIC_ACTIONS = {
        "ready": {"name": "准备 Ready", "icon": "🧍", "color": "#888888"},
        "moving": {"name": "移动 Moving", "icon": "🏃", "color": "#ffdd44"},
        "arm_raise": {"name": "抬臂 Arm raise", "icon": "🙋", "color": "#00f0ff"},
        "arm_swing": {"name": "摆臂 Arm swing", "icon": "💫", "color": "#33ff88"},
    }

    def __init__(self, window_size: int = 15, debounce_frames: int = 10, mode: str = "generic"):
        """
        Args:
            window_size: 滑动窗口大小（帧数）
            debounce_frames: 动作去抖动间隔（防止同一动作重复触发）
        """
        self.mode = mode if mode in {"badminton", "tennis"} else "generic"
        self.actions = self.ACTIONS if self.mode != "generic" else self.GENERIC_ACTIONS
        self.window_size = window_size
        self.debounce_frames = debounce_frames
        # 关键点历史缓冲
        self.keypoint_buffer: deque = deque(maxlen=window_size)
        # 动作历史
        self.action_history: list = []
        self.last_action = "ready"
        self.last_action_frame = -debounce_frames
        self.frame_count = 0
        # 统计
        self.action_counts: dict = {k: 0 for k in self.actions}

    def update(self, keypoints: list, joint_angles: dict) -> dict:
        """
        输入当前帧的关键点和关节角度，输出识别结果

        Returns:
            {
                "action": str,
                "action_info": {name, icon, color},
                "confidence": float,  # legacy alias; see rule_match_score
                "rule_match_score": float,
                "score_kind": "rule_match_not_probability",
                "is_new_action": bool,
                "action_counts": dict,
                "action_history": list (recent 20),
            }
        """
        self.frame_count += 1

        # 构建关键点字典
        kp_map = {}
        for kp in keypoints:
            kp_map[kp["name"]] = kp
        self.keypoint_buffer.append(kp_map)

        if len(self.keypoint_buffer) < 5:
            return self._make_result("ready", 0.5, False)

        # 识别动作
        action, rule_match_score = self._recognize(kp_map, joint_angles)

        # 去抖动
        is_new = False
        if action != self.last_action and action not in ("ready", "moving"):
            if self.frame_count - self.last_action_frame >= self.debounce_frames:
                is_new = True
                self.last_action = action
                self.last_action_frame = self.frame_count
                self.action_counts[action] = self.action_counts.get(action, 0) + 1
                self.action_history.append({
                    "action": action,
                    "frame": self.frame_count,
                    "rule_match_score": rule_match_score,
                })
        elif action in ("ready", "moving"):
            self.last_action = action

        return self._make_result(action, rule_match_score, is_new)

    def _recognize(self, kp: dict, angles: dict) -> tuple:
        """Classify racket-sport motion in body-relative, not pixel, units.

        Fixed pixel thresholds make a detector change behaviour whenever the
        camera zooms.  These rules use shoulder width as a local scale and pick
        the arm with the higher recent velocity, making the baseline usable for
        either handedness.  Labels remain *rule candidates*, not ground truth.
        """

        nose = kp.get("nose", {})
        left_shoulder = kp.get("left_shoulder", {})
        right_shoulder = kp.get("right_shoulder", {})
        if not all((nose, left_shoulder, right_shoulder)):
            return "ready", 0.3

        body_scale = max(math.dist(
            (left_shoulder.get("x", 0), left_shoulder.get("y", 0)),
            (right_shoulder.get("x", 0), right_shoulder.get("y", 0)),
        ), 1.0)
        side = self._active_side()
        wrist = kp.get(f"{side}_wrist", {})
        shoulder = kp.get(f"{side}_shoulder", {})
        hip = kp.get(f"{side}_hip", {})
        if not all((wrist, shoulder, hip)):
            return "ready", 0.3

        wrist_speed = self._get_landmark_speed(f"{side}_wrist") / body_scale
        wrist_vertical_speed = self._get_landmark_velocity(f"{side}_wrist", "y") / body_scale
        lateral_speed = self._get_landmark_velocity(f"{side}_wrist", "x") / body_scale
        body_speed = self._get_body_speed() / body_scale
        elbow_angle = angles.get(f"{side}_elbow", 90)
        shoulder_angle = angles.get(f"{side}_shoulder", 90)
        wrist_y = wrist.get("y", 0)
        wrist_x = wrist.get("x", 0)
        shoulder_y = shoulder.get("y", 0)
        shoulder_x = shoulder.get("x", 0)
        hip_y = hip.get("y", 0)
        nose_y = nose.get("y", 0)

        if self.mode == "generic":
            return self._recognize_generic(
                wrist_speed=wrist_speed,
                wrist_vertical_speed=wrist_vertical_speed,
                body_speed=body_speed,
                wrist_y=wrist_y,
                shoulder_y=shoulder_y,
            )

        # Serve: raised, extended arm followed by a clearly downward movement.
        if wrist_y < nose_y and elbow_angle > 140 and shoulder_angle > 105 and wrist_vertical_speed > 0.10:
            return "serve", 0.85
        # Smash: fast, high downward stroke.
        if wrist_y < shoulder_y - 0.22 * body_scale and wrist_speed > 0.42 and wrist_vertical_speed > 0.14:
            return "smash", 0.80
        # Lob: a low wrist travelling upward.
        if wrist_y > hip_y and wrist_vertical_speed < -0.10 and wrist_speed > 0.25:
            return "lob", 0.70
        # Drop: elevated arm moving down, but not with smash-level speed.
        if wrist_y < shoulder_y and 0.02 < wrist_vertical_speed < 0.14 and wrist_speed < 0.35 and elbow_angle > 100:
            return "drop", 0.65

        same_side = wrist_x > shoulder_x if side == "right" else wrist_x < shoulder_x
        lateral_outward = lateral_speed > 0.06 if side == "right" else lateral_speed < -0.06
        if wrist_speed > 0.22:
            if same_side and lateral_outward:
                return "forehand", 0.75
            if not same_side and not lateral_outward:
                return "backhand", 0.70
        if body_speed > 0.12:
            return "moving", 0.60
        return "ready", 0.50

    @staticmethod
    def _recognize_generic(
        *,
        wrist_speed: float,
        wrist_vertical_speed: float,
        body_speed: float,
        wrist_y: float,
        shoulder_y: float,
    ) -> tuple[str, float]:
        """Return sport-neutral motion candidates for unspecified activities."""

        if wrist_y < shoulder_y and abs(wrist_vertical_speed) > 0.08:
            return "arm_raise", 0.70
        if wrist_speed > 0.25:
            return "arm_swing", 0.70
        if body_speed > 0.12:
            return "moving", 0.60
        return "ready", 0.50

    def _active_side(self) -> str:
        """Choose the currently active arm; this avoids a right-handed bias."""

        right_speed = self._get_landmark_speed("right_wrist")
        left_speed = self._get_landmark_speed("left_wrist")
        return "left" if left_speed > right_speed else "right"

    def _get_landmark_speed(self, name: str) -> float:
        if len(self.keypoint_buffer) < 2:
            return 0.0
        current = self.keypoint_buffer[-1].get(name, {})
        previous = self.keypoint_buffer[-2].get(name, {})
        if not current or not previous:
            return 0.0
        return math.dist((current.get("x", 0), current.get("y", 0)), (previous.get("x", 0), previous.get("y", 0)))

    def _get_landmark_velocity(self, name: str, axis: str) -> float:
        if len(self.keypoint_buffer) < 3:
            return 0.0
        current = self.keypoint_buffer[-1].get(name, {})
        previous = self.keypoint_buffer[-3].get(name, {})
        if not current or not previous:
            return 0.0
        return (current.get(axis, 0) - previous.get(axis, 0)) / 2

    def _get_wrist_speed(self) -> float:
        """计算手腕速度（像素/帧）"""
        return self._get_landmark_speed("right_wrist")

    def _get_wrist_vertical_speed(self) -> float:
        """手腕垂直速度（正值=向下，负值=向上）"""
        return self._get_landmark_velocity("right_wrist", "y")

    def _get_wrist_lateral_speed(self) -> float:
        """手腕横向速度（正值=向右，负值=向左）"""
        return self._get_landmark_velocity("right_wrist", "x")

    def _get_body_speed(self) -> float:
        """身体整体移动速度（基于髋部）"""
        if len(self.keypoint_buffer) < 2:
            return 0
        curr_lh = self.keypoint_buffer[-1].get("left_hip", {})
        curr_rh = self.keypoint_buffer[-1].get("right_hip", {})
        prev_lh = self.keypoint_buffer[-2].get("left_hip", {})
        prev_rh = self.keypoint_buffer[-2].get("right_hip", {})
        if not all([curr_lh, curr_rh, prev_lh, prev_rh]):
            return 0
        cx = (curr_lh.get("x", 0) + curr_rh.get("x", 0)) / 2
        cy = (curr_lh.get("y", 0) + curr_rh.get("y", 0)) / 2
        px = (prev_lh.get("x", 0) + prev_rh.get("x", 0)) / 2
        py = (prev_lh.get("y", 0) + prev_rh.get("y", 0)) / 2
        return math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

    def _make_result(self, action: str, rule_match_score: float, is_new: bool) -> dict:
        action_info = self.actions.get(action, self.actions["ready"])
        rounded_score = round(rule_match_score, 2)
        return {
            "action": action,
            "analysis_mode": self.mode,
            "action_info": action_info,
            # Retained for existing frontend/API consumers.  It is explicitly
            # not a calibrated model probability.
            "confidence": rounded_score,
            "rule_match_score": rounded_score,
            "score_kind": "rule_match_not_probability",
            "is_new_action": is_new,
            "action_counts": dict(self.action_counts),
            "action_history": self.action_history[-20:],
        }

    def reset(self):
        """重置状态"""
        self.keypoint_buffer.clear()
        self.action_history.clear()
        self.last_action = "ready"
        self.last_action_frame = -self.debounce_frames
        self.frame_count = 0
        self.action_counts = {k: 0 for k in self.ACTIONS}
