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


DEFAULT_ORDER = {
    "right": [
        "ahr_joint00", "ahr_joint01", "ahr_joint02", "ahr_joint03",
        "ahr_joint10", "ahr_joint11", "ahr_joint12", "ahr_joint13",
        "ahr_joint20", "ahr_joint21", "ahr_joint22", "ahr_joint23",
        "ahr_joint30", "ahr_joint31", "ahr_joint32", "ahr_joint33",
    ],
    "left": [
        "ahl_joint00", "ahl_joint01", "ahl_joint02", "ahl_joint03",
        "ahl_joint10", "ahl_joint11", "ahl_joint12", "ahl_joint13",
        "ahl_joint20", "ahl_joint21", "ahl_joint22", "ahl_joint23",
        "ahl_joint30", "ahl_joint31", "ahl_joint32", "ahl_joint33",
    ],
}

###################################################################################################
# TBD #
# 양손 코드는 한손 코드 기반으로 직접 작성 예정 25.10.31


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

    # def poll_joint_position(self, wait: bool = False, timeout: float = 3.0) -> Optional[np.ndarray]:
    #     if self._last_js is None and wait:
    #         t0 = time.time()
    #         while self._last_js is None and (time.time() - t0) < timeout:
    #             rclpy.spin_once(self, timeout_sec=0.05)

    #     js = self._last_js
    #     if js is None or not js.position:
    #         return None

    #     if self._index_map is None and js.name:
    #         lower_to_idx = {n.lower(): i for i, n in enumerate(js.name)}
    #         idx_map = []
    #         ok = True
    #         for want in self._desired_names:
    #             i = lower_to_idx.get(want.lower(), None)
    #             if i is None:
    #                 ok = False
    #                 break
    #             idx_map.append(i)
    #         if ok and len(idx_map) == 16:
    #             self._index_map = idx_map

    #     try:
    #         if self._index_map is not None:
    #             vec = np.array([js.position[i] for i in self._index_map], dtype=float)
    #             if vec.size == 16:
    #                 return vec
    #     except Exception:
    #         pass

    #     if len(js.position) >= 16:
    #         return np.array(js.position[:16], dtype=float)
    #     return None

    def poll_joint_position(
        self, wait: bool = False, timeout: float = 3.0
    ) -> Optional[np.ndarray]:
        """현재 조인트 위치를 Allegro 순서(16-D)로 반환합니다.

        Args:
            wait (bool): 데이터를 기다릴지 여부 (True이면 timeout까지 spin)
            timeout (float): 최대 대기 시간(초)

        Returns:
            np.ndarray | None: 16-D 조인트 벡터 또는 None
        """

        # 1️⃣ JointState 수신 대기
        if self._last_js is None and wait:
            start_time = time.time()
            while self._last_js is None and (time.time() - start_time) < timeout:
                rclpy.spin_once(self, timeout_sec=0.05)

        js = self._last_js
        if js is None or not js.position:
            return None

        # 2️⃣ 이름 기반 인덱스 매핑 초기화 (첫 수신 시 1회만)
        if self._index_map is None and js.name:
            self._index_map = self._build_index_map(js.name)

        # 3️⃣ 인덱스 매핑 성공 시 그대로 정렬하여 반환
        if self._index_map:
            try:
                vec = np.array([js.position[i] for i in self._index_map], dtype=float)
                if vec.size == 16:
                    return vec
            except Exception:
                self.get_logger().warn("poll_joint_position: index mapping failed, fallback to raw order.")

        # 4️⃣ 매핑 실패 시 fallback — 앞 16개 사용
        if len(js.position) >= 16:
            return np.array(js.position[:16], dtype=float)

        return None


    def _build_index_map(self, joint_names: List[str]) -> Optional[List[int]]:
        """joint_states의 이름 리스트로부터 Allegro 16D 인덱스 매핑 생성."""
        name_to_index = {n.lower(): i for i, n in enumerate(joint_names)}
        index_map = []

        for desired in self._desired_names:
            idx = name_to_index.get(desired.lower())
            if idx is None:
                # 하나라도 매칭 실패하면 전체 매핑 무효화
                self.get_logger().warn(f"Missing joint name in /joint_states: '{desired}'")
                return None
            index_map.append(idx)

        return index_map if len(index_map) == 16 else None


    def go_safe(self):
        # fix: use correct method name
        self.command_joint_position(self.safe_pose)

    def _on_js(self, msg: JointState):
        self._last_js = msg


################### one hand io runner ####################

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

################ two hands io demo ####################

# --- 추가: 여러 노드를 한 executor에서 돌리기 ---
class _RunnerMany:
    def __init__(self, nodes):
        from rclpy.executors import SingleThreadedExecutor
        self.nodes = nodes
        self.exec = SingleThreadedExecutor()
        for n in nodes:
            self.exec.add_node(n)
        import threading
        self.thread = threading.Thread(target=self.exec.spin, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        try:
            self.exec.shutdown()
        finally:
            for n in self.nodes:
                n.destroy_node()

def start_allegro_ios(sides=("right", "left")):
    if not rclpy.ok():
        rclpy.init()
    nodes = [AllegroHandIO(side=s) for s in sides]
    runner = _RunnerMany(nodes)
    runner.start()
    # 러너를 참조로 보존(정리 시 필요)
    for n in nodes:
        n._runner_many = runner
    return {n.side: n for n in nodes}

def stop_allegro_ios(nodes_dict):
    # 동일 runner를 공유하므로 하나만 꺼주면 됨
    any_node = next(iter(nodes_dict.values()))
    if hasattr(any_node, "_runner_many"):
        any_node._runner_many.stop()
    if rclpy.ok():
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":

    ##################### simple demo for one hand #####################
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


    #####################  demo for two hands #####################
    # ios = start_allegro_ios(("right", "left"))
    # io_r = ios["right"]
    # io_l = ios["left"]

    # # /joint_states 도착 대기
    # cur_r = io_r.poll_joint_position(wait=True, timeout=5.0)
    # cur_l = io_l.poll_joint_position(wait=True, timeout=5.0)

    # # 에코 커맨드
    # if cur_r is not None: io_r.command_joint_position(cur_r)
    # if cur_l is not None: io_l.command_joint_position(cur_l)

    # time.sleep(2.0)

    # # 안전자세 후 종료
    # io_r.go_safe()
    # io_l.go_safe()
    # stop_allegro_ios(ios)
