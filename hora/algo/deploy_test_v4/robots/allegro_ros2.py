#!/usr/bin/env python3
"""
Allegro Hand I/O (simple)

Main functions:
  - command_joint_position(positions)      # publish 16-D command
  - poll_joint_position(wait=False, ...)   # read 16-D current vector

Notes:
  - Side-aware (right/left) index mapping from /joint_states name list.
  - If mapping fails, falls back to first 16 positions.
  - start_allegro_io/stop_allegro_io manage rclpy + spinning in a background thread.
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


DEFAULT_ORDER = {
    "right": [
        "ahr_joint.p.00", "ahr_joint.00.01", "ahr_joint.01.02", "ahr_joint.02.03",
        "ahr_joint.p.10", "ahr_joint.10.11", "ahr_joint.11.12", "ahr_joint.12.13",
        "ahr_joint.p.20", "ahr_joint.20.21", "ahr_joint.21.22", "ahr_joint.22.23",
        "ahr_joint.p.30", "ahr_joint.30.31", "ahr_joint.31.32", "ahr_joint.32.33",
    ],
    "left": [
        "ahl_joint.p.00", "ahl_joint.00.01", "ahl_joint.01.02", "ahl_joint.02.03",
        "ahl_joint.p.10", "ahl_joint.10.11", "ahl_joint.11.12", "ahl_joint.12.13",
        "ahl_joint.p.20", "ahl_joint.20.21", "ahl_joint.21.22", "ahl_joint.22.23",
        "ahl_joint.p.30", "ahl_joint.30.31", "ahl_joint.31.32", "ahl_joint.32.33",
    ],
}


class AllegroHandIO(Node):
    """
    Minimal ROS 2 node for Allegro hand.
    - Publishes commands to /<controller_name>/commands
    - Subscribes to /joint_states and returns a 16-D vector in Allegro order
    """

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

        if controller_name is None:
            controller_name = (
                "allegro_hand_position_controller_r"
                if self.side == "right"
                else "allegro_hand_position_controller_l"
            )
        if command_topic is None:
            command_topic = f"/{controller_name}/commands"

        self._cmd_pub = self.create_publisher(Float64MultiArray, command_topic, 10)
        self._last_js: Optional[JointState] = None
        self._index_map: Optional[List[int]] = None
        self._desired_names = DEFAULT_ORDER[self.side]

        self.create_subscription(JointState, joint_states_topic, self._on_js, 10)

        self.safe_pose = np.array([
            0.5, 0.2, 0.0, 0.0,   # Thumb
            0.0, 0.0, 0.0, 0.0,   # Index
            0.0, 0.0, 0.0, 0.0,   # Middle
            0.0, 0.0, 0.0, 0.0,   # Ring
        ], dtype=float)

        self.get_logger().info(f"[AllegroHandIO] side={self.side}")
        self.get_logger().info(f"[AllegroHandIO] cmd topic={command_topic}")
        self.get_logger().info(f"[AllegroHandIO] joint_states topic={joint_states_topic}")

    def command_joint_position(self, positions: List[float]) -> bool:
        try:
            data = [float(x) for x in list(positions)]
        except Exception:
            self.get_logger().warn("command_joint_position: positions must be a sequence of numbers.")
            return False

        if len(data) != 16:
            self.get_logger().warn(f"command_joint_position: expected 16 elements, got {len(data)}.")
            return False

        msg = Float64MultiArray()
        msg.data = data
        self._cmd_pub.publish(msg)
        return True

    def poll_joint_position(self, wait: bool = False, timeout: float = 3.0) -> Optional[np.ndarray]:
        if self._last_js is None and wait:
            t0 = time.time()
            while self._last_js is None and (time.time() - t0) < timeout:
                rclpy.spin_once(self, timeout_sec=0.05)

        js = self._last_js
        if js is None or not js.position:
            return None

        if self._index_map is None and js.name:
            lower_to_idx = {n.lower(): i for i, n in enumerate(js.name)}
            idx_map = []
            ok = True
            for want in self._desired_names:
                i = lower_to_idx.get(want.lower(), None)
                if i is None:
                    ok = False
                    break
                idx_map.append(i)
            if ok and len(idx_map) == 16:
                self._index_map = idx_map

        try:
            if self._index_map is not None:
                vec = np.array([js.position[i] for i in self._index_map], dtype=float)
                if vec.size == 16:
                    return vec
        except Exception:
            pass

        if len(js.position) >= 16:
            return np.array(js.position[:16], dtype=float)
        return None

    def go_safe(self):
        # fix: use correct method name
        self.command_joint_position(self.safe_pose)

    def _on_js(self, msg: JointState):
        self._last_js = msg


class _Runner:
    def __init__(self, node: AllegroHandIO):
        self.node = node
        self.exec = SingleThreadedExecutor()
        self.exec.add_node(node)
        self.thread = threading.Thread(target=self.exec.spin, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        try:
            self.exec.shutdown()
        finally:
            self.node.destroy_node()


def start_allegro_io(side: str = "right") -> AllegroHandIO:
    if not rclpy.ok():
        rclpy.init()
    io = AllegroHandIO(side=side)
    io._runner = _Runner(io)   # keep reference
    io._runner.start()
    return io


def stop_allegro_io(io: AllegroHandIO):
    if hasattr(io, "_runner") and io._runner:
        io._runner.stop()
    if rclpy.ok():
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    io = start_allegro_io(side="right")
    try:
        print("[demo] waiting for /joint_states ...")
        cur = io.poll_joint_position(wait=True, timeout=5.0)
        print("[demo] current 16-D:", None if cur is None else np.round(cur, 6))
        if cur is not None:
            print("[demo] echoing current back as command")
            io.command_joint_position(cur)
        time.sleep(2.0)
    finally:
        print("[demo] going to safe pose and stopping...")
        io.go_safe()
        stop_allegro_io(io)
