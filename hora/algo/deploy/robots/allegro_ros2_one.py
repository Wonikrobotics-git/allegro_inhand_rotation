#!/usr/bin/env python3
"""
Allegro Hand I/O (simple)

"""

import threading
import time
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


# =========================================================
# Allegro ROS2 I/O (백그라운드 스피너 + position_gap 퍼블리셔)
# =========================================================

DEFAULT_ORDER = {
    "right": [
        "ah_joint00", "ah_joint01", "ah_joint02", "ah_joint03",
        "ah_joint10", "ah_joint11", "ah_joint12", "ah_joint13",
        "ah_joint20", "ah_joint21", "ah_joint22", "ah_joint23",
        "ah_joint30", "ah_joint31", "ah_joint32", "ah_joint33",
    ],
}


class AllegroHandIO(Node):
    """ROS2 Allegro I/O 노드: 명령 퍼블리시 + joint_states 구독 + /position_gap 퍼블리시"""
    def __init__(
        self,
        side: str = "right",
        controller_name: Optional[str] = None,
        joint_states_topic: str = "/joint_states",
        command_topic: Optional[str] = None,
    ):
        super().__init__("allegro_hand_io")

        side = (side or "right").lower()
        if side not in ("right", "left"):
            self.get_logger().warn(f"Unknown side '{side}', defaulting to 'right'.")
            side = "right"
        self.side = side

        controller_name = controller_name or "allegro_hand_position_controller"
        if command_topic is None:
            command_topic = f"/{controller_name}/commands"

        # pubs/subs
        self._cmd_pub = self.create_publisher(Float64MultiArray, command_topic, 10)
        self._gap_pub = self.create_publisher(JointState, "/position_gap", 10)
        self.create_subscription(JointState, joint_states_topic, self._on_js, 50)

        # state
        self._last_js: Optional[JointState] = None
        self._index_map: Optional[List[int]] = None
        self._last_cmd: Optional[np.ndarray] = None

        self._desired_names = DEFAULT_ORDER["right"]

        # 안전 포즈
        self.safe_pose = np.array([
            0.5, 0.2, 0.0, 0.0,   # Thumb
            0.0, 0.0, 0.0, 0.0,   # Index
            0.0, 0.0, 0.0, 0.0,   # Middle
            0.0, 0.0, 0.0, 0.0,   # Ring
        ], dtype=float)

        self.get_logger().info(f"[AllegroHandIO] side={self.side}")
        self.get_logger().info(f"[AllegroHandIO] cmd topic={command_topic}")
        self.get_logger().info(f"[AllegroHandIO] joint_states topic={joint_states_topic}")
        self.get_logger().info(f"[AllegroHandIO] gap topic=/position_gap")

    # ---------- public ----------
    def command_joint_position(self, positions: List[float]) -> bool:
        """16D 목표 자세 퍼블리시 + 직후 1회 gap 퍼블리시"""
        try:
            data = [float(x) for x in positions]
        except Exception:
            self.get_logger().warn("command_joint_position: positions must be a sequence of numbers.")
            return False
        if len(data) != 16:
            self.get_logger().warn(f"command_joint_position: expected 16 elements, got {len(data)}.")
            return False

        msg = Float64MultiArray()
        msg.data = data
        self._cmd_pub.publish(msg)
        self._last_cmd = np.asarray(data, dtype=float)

        # 현재 최신 상태로 gap 한번 퍼블리시(논블로킹)
        self._publish_position_gap()
        return True

    def poll_joint_position(self, wait: bool = False, timeout: float = 0.0) -> Optional[np.ndarray]:
        """현재 조인트를 Allegro 순서(16D)로 반환"""
        if self._last_js is None and wait:
            # timer 콜백에서 블로킹은 권장하지 않으나, 초기화 시에는 사용 가능
            end_t = time.perf_counter() + max(0.0, float(timeout))
            while self._last_js is None and time.perf_counter() < end_t:
                rclpy.spin_once(self, timeout_sec=0.02)

        js = self._last_js
        if js is None or not js.position:
            return None

        if self._index_map is None and js.name:
            self._index_map = self._build_index_map(js.name)

        if self._index_map:
            try:
                vec = np.array([js.position[i] for i in self._index_map], dtype=float)
                if vec.size == 16:
                    return vec
            except Exception:
                self.get_logger().warn("poll_joint_position: index mapping failed, fallback to raw order.")

        if len(js.position) >= 16:
            return np.array(js.position[:16], dtype=float)
        return None

    def go_safe(self):
        self.command_joint_position(self.safe_pose)

    # ---------- internals ----------
    def _on_js(self, msg: JointState):
        self._last_js = msg

    def _build_index_map(self, joint_names: List[str]) -> Optional[List[int]]:
        mp = {n.lower(): i for i, n in enumerate(joint_names)}
        out = []
        for desired in self._desired_names:
            idx = mp.get(desired.lower())
            if idx is None:
                self.get_logger().warn(f"Missing joint name in /joint_states: '{desired}'")
                return None
            out.append(idx)
        return out if len(out) == 16 else None

    def _publish_position_gap(self):
        if self._last_cmd is None or self._last_js is None:
            return
        cur = self.poll_joint_position(wait=False)
        if cur is None or cur.size != 16:
            return
        gap = (self._last_cmd - cur).astype(float)
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.header.frame_id = ""
        js.name = list(self._desired_names)
        js.position = gap.tolist()
        self._gap_pub.publish(js)


# ========== FIX: Runner를 표준 threading 기반으로 ==========

import threading
from rclpy.executors import SingleThreadedExecutor

class _Runner:
    def __init__(self, node: 'AllegroHandIO'):
        self.node = node
        self.exec = SingleThreadedExecutor()
        self.exec.add_node(node)
        self.thread = threading.Thread(target=self.exec.spin, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        try:
            self.exec.shutdown()          # executor 중지
        finally:
            try:
                self.node.destroy_node()  # 노드 파괴
            finally:
                # 백그라운드 스레드 종료 대기 (너무 오래 기다리지 않도록 타임아웃)
                self.thread.join(timeout=2.0)



def start_allegro_io(side: str = "right") -> 'AllegroHandIO':
    if not rclpy.ok():
        rclpy.init()
    io = AllegroHandIO(side=side)
    io._runner = _Runner(io)
    io._runner.start()
    return io

def stop_allegro_io(io: 'AllegroHandIO'):
    if hasattr(io, "_runner") and io._runner:
        io._runner.stop()
    if rclpy.ok():
        try:
            rclpy.shutdown()
        except Exception:
            pass
