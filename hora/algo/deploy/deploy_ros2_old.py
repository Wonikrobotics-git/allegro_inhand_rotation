#!/usr/bin/env python3
import os
import time
import numpy as np
import torch

from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.utils.misc import tprint

# Allegro I/O (백그라운드 스피너 사용)
# from hora.algo.deploy.robots.allegro_ros2 import start_allegro_io, stop_allegro_io
# from hora.algo.deploy.robots.allegro_ros2_one import start_allegro_io, stop_allegro_io
# from std_msgs.msg import Float64MultiArray


import threading
import time
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Header


DEFAULT_ORDER = {
    "right": [
        "ah_joint00", "ah_joint01", "ah_joint02", "ah_joint03",
        "ah_joint10", "ah_joint11", "ah_joint12", "ah_joint13",
        "ah_joint20", "ah_joint21", "ah_joint22", "ah_joint23",
        "ah_joint30", "ah_joint31", "ah_joint32", "ah_joint33",
    ],
    # name:
    # - ah_joint12 ====== 0  : index  2
    # - ah_joint13 ====== 1  : index  3
    # - ah_joint11 ====== 2  : index  1
    # - ah_joint03 ====== 3  : thumb  3
    # - ah_joint32 ====== 4  : ring   2
    # - ah_joint10 ====== 5  : index  0
    # - ah_joint01 ====== 6  : thumb  1
    # - ah_joint20 ====== 7  : middle 0
    # - ah_joint00 ====== 8  : thumb  0
    # - ah_joint33 ====== 9  : ring   3
    # - ah_joint02 ====== 10 : thumb  2
    # - ah_joint30 ====== 11 : ring   0
    # - ah_joint23 ====== 12 : middle 3
    # - ah_joint31 ====== 13 : ring   1
    # - ah_joint22 ====== 14 : middle 2
    # - ah_joint21 ====== 15 : middle 1
}


class AllegroHandIO(Node):
    """
    Minimal ROS 2 node for Allegro hand.
    - Publishes commands to /<controller_name>/commands
    - Subscribes to /joint_states and returns a 16-D vector in Allegro order
    - Publishes position gap (command - current) to /position_gap (JointState)
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

        # 컨트롤러명 고정 (필요 시 외부 인자로 바꿔도 됨)
        controller_name = controller_name or "allegro_hand_position_controller"

        if command_topic is None:
            command_topic = f"/{controller_name}/commands"

        # Publishers / Subscribers
        self._cmd_pub = self.create_publisher(Float64MultiArray, command_topic, 10)
        self._gap_pub = self.create_publisher(JointState, "/position_gap", 10)
        self.create_subscription(JointState, joint_states_topic, self._on_js, 10)

        # 최근 상태 / 매핑 / 명령 저장
        self._last_js: Optional[JointState] = None
        self._index_map: Optional[List[int]] = None
        self._last_cmd: Optional[np.ndarray] = None  # 명령 16D (Allegro 순서)

        # Allegro 조인트 이름(원하는 정렬 순서)
        self._desired_names = DEFAULT_ORDER["right"] if self.side == "right" else DEFAULT_ORDER["left"]

        # 명령 직후 gap 측정 전에 짧은 대기 (joint_states 반영 유도)
        self._gap_after_cmd_delay = 0.02  # seconds

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

    # -------------------- Public APIs --------------------

    def command_joint_position(self, positions: List[float]) -> bool:
        """16D 목표 자세 명령 퍼블리시 + 직후 position gap 1회 퍼블리시."""
        try:
            data = [float(x) for x in list(positions)]
        except Exception:
            self.get_logger().warn("command_joint_position: positions must be a sequence of numbers.")
            return False

        if len(data) != 16:
            self.get_logger().warn(f"command_joint_position: expected 16 elements, got {len(data)}.")
            return False

        # 명령 퍼블리시
        msg = Float64MultiArray()
        msg.data = data
        self._cmd_pub.publish(msg)

        # 마지막 명령 기억
        self._last_cmd = np.asarray(data, dtype=float)

        # 아주 짧게 대기 → 최신 /joint_states 반영 유도 (백그라운드 스피너가 콜백 실행)
        if self._gap_after_cmd_delay > 0:
            time.sleep(self._gap_after_cmd_delay)

        # 명령 직후 gap 1회 퍼블리시
        self._publish_position_gap()

        return True

    def poll_joint_position(
        self, wait: bool = False, timeout: float = 3.0
    ) -> Optional[np.ndarray]:
        """현재 조인트 위치를 Allegro 순서(16-D)로 반환.

        Args:
            wait (bool): 데이터를 기다릴지 여부 (True이면 timeout까지 spin)
            timeout (float): 최대 대기 시간(초)

        Returns:
            np.ndarray | None: 16-D 조인트 벡터 또는 None
        """

        # 1) JointState 수신 대기
        if self._last_js is None and wait:
            start_time = time.time()
            while self._last_js is None and (time.time() - start_time) < timeout:
                rclpy.spin_once(self, timeout_sec=0.05)

        js = self._last_js
        if js is None or not js.position:
            return None

        # 2) 이름 기반 인덱스 매핑 초기화 (첫 수신 시 1회만)
        if self._index_map is None and js.name:
            self._index_map = self._build_index_map(js.name)

        # 3) 인덱스 매핑 성공 시 그대로 정렬하여 반환
        if self._index_map:
            try:
                vec = np.array([js.position[i] for i in self._index_map], dtype=float)
                if vec.size == 16:
                    return vec
            except Exception:
                self.get_logger().warn("poll_joint_position: index mapping failed, fallback to raw order.")

        # 4) 매핑 실패 시 fallback — 앞 16개 사용
        if len(js.position) >= 16:
            return np.array(js.position[:16], dtype=float)

        return None

    def go_safe(self):
        """안전 포즈로 이동."""
        self.command_joint_position(self.safe_pose)

    # -------------------- Internals --------------------

    def _publish_position_gap(self):
        """마지막 명령(self._last_cmd)과 현재 관측을 비교해 gap을 /position_gap으로 퍼블리시."""
        if self._last_cmd is None or self._last_js is None:
            return

        # 현재 위치(16D, Allegro 순서) 읽기
        cur = self.poll_joint_position(wait=False)
        if cur is None or cur.size != 16:
            return

        cmd = np.asarray(self._last_cmd, dtype=float).reshape(16)
        gap = (cmd - cur).astype(float)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        # frame_id는 빈 문자열로 두거나 필요 시 "allegro" 등으로 지정
        msg.header.frame_id = ""
        msg.name = list(self._desired_names)  # 16개 조인트 이름
        msg.position = gap.tolist()           # gap(16D)을 position 필드에 담음
        # velocity/effort는 미사용 (빈 리스트)
        self._gap_pub.publish(msg)

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



# ---------- Reorder helpers ----------
def _action_hora2allegro(actions):
    if isinstance(actions, torch.Tensor):
        if actions.dim() > 1:
            actions = actions.view(-1)
        cmd_act = actions.clone()
        temp = actions[[4, 5, 6, 7]].clone()
        cmd_act[[4, 5, 6, 7]] = actions[[8, 9, 10, 11]]
        cmd_act[[12, 13, 14, 15]] = temp
        cmd_act[[8, 9, 10, 11]] = actions[[12, 13, 14, 15]]
        return cmd_act
    else:
        a = np.asarray(actions).flatten()
        cmd = a.copy()
        temp = a[[4, 5, 6, 7]].copy()
        cmd[[4, 5, 6, 7]] = a[[8, 9, 10, 11]]
        cmd[[12, 13, 14, 15]] = temp
        cmd[[8, 9, 10, 11]] = a[[12, 13, 14, 15]]
        return cmd


def _obs_allegro2hora(o):
    # allegro: index - middle - ring - thumb
    # hora  : index, thumb, middle, ring
    return np.concatenate([o[0:4], o[12:16], o[4:8], o[8:12]]).astype(np.float64)


def _reorder_imrt2timr(imrt):
    # [ROS1] index - middle - ring - thumb  ->  [ROS2] thumb - index - middle - ring
    return np.concatenate([imrt[12:16], imrt[0:12]]).astype(np.float64)


def _reorder_timr2imrt(timr):
    # [ROS2] thumb - index - middle - ring  ->  [ROS1] index - middle - ring - thumb
    return np.concatenate([timr[4:16], timr[0:4]]).astype(np.float64)


class HardwarePlayer(object):
    def __init__(self):
        self.action_scale = 1 / 24
        self.actions_num = 16
        self.device = "cuda"  # ← 여기서 디바이스 통일

        # ===== Model / RMS =====
        obs_shape = (96,)
        net_config = {
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            "actor_units": [512, 256, 128],
            "priv_mlp_units": [256, 128, 8],
            "priv_info": True,
            "proprio_adapt": True,
            "priv_info_dim": 9,
        }

        self.model = ActorCritic(net_config).to(self.device).eval()
        self.running_mean_std = RunningMeanStd(obs_shape).to(self.device).eval()
        self.sa_mean_std = RunningMeanStd((30, 32)).to(self.device).eval()

        # ===== Runtime buffers (device 통일) =====
        self.obs_buf = torch.zeros((1, 16 * 3 * 2), dtype=torch.float32, device=self.device)           # 96
        self.proprio_hist_buf = torch.zeros((1, 30, 16 * 2), dtype=torch.float32, device=self.device)  # (1,30,32)

        # ===== Allegro joint limits =====
        self.allegro_dof_lower = torch.tensor([
            -0.4700, -0.1960, -0.1740, -0.2270,   # Index
             0.2630, -0.1050, -0.1890, -0.1620,   # Thumb
            -0.4700, -0.1960, -0.1740, -0.2270,   # Middle
            -0.4700, -0.1960, -0.1740, -0.2270,   # Ring
        ], dtype=torch.float32, device=self.device)

        self.allegro_dof_upper = torch.tensor([
             0.4700, 1.6100, 1.7090, 1.6180,      # Index
             1.3960, 1.1630, 1.6440, 1.7190,      # Thumb
             0.4700, 1.6100, 1.7090, 1.6180,      # Middle
             0.4700, 1.6100, 1.7090, 1.6180,      # Ring
        ], dtype=torch.float32, device=self.device)

        # ===== Poses =====
        self.init_pose = [
            0.0627, 1.2923, 0.3383, 0.1088,
            0.0724, 1.1983, 0.1551, 0.1499,
            0.1343, 1.1736, 0.5355, 0.2164,
            1.1202, 1.1374, 0.8535, -0.0852,
        ]

        # ===== Targets (rad) =====
        self.prev_target = torch.zeros((1, 16), dtype=torch.float32, device=self.device)
        self.cur_target  = torch.zeros((1, 16), dtype=torch.float32, device=self.device)

    # ---------- Steps ----------
    def pre_physics_step(self, action):
        self.action = action.clone()
        target = self.prev_target + self.action_scale * self.action
        self.cur_target = torch.clip(target, self.allegro_dof_lower, self.allegro_dof_upper)
        self.prev_target = self.cur_target.clone()

    def post_physics_step(self, obses):
        # normalize current obs (obses: real q) -> (1,16)
        self.cur_obs_buf = self.unscale(
            obses, self.allegro_dof_lower, self.allegro_dof_upper
        )[None]

        # obs_buf roll
        self.prev_obs_buf = self.obs_buf[:, 32:].clone()
        self.obs_buf[:, :64]   = self.prev_obs_buf
        self.obs_buf[:, 64:80] = self.cur_obs_buf.clone()
        self.obs_buf[:, 80:96] = self.cur_target.clone()  # real target

        # proprio history (30)
        cur_norm_t = self.cur_obs_buf.unsqueeze(1)          # (1,1,16)
        cur_tgt_t  = self.cur_target.unsqueeze(1)           # (1,1,16)
        cur = torch.cat([cur_norm_t, cur_tgt_t], dim=-1)    # (1,1,32)
        prev = self.proprio_hist_buf[:, 1:30, :].clone()    # (1,29,32)
        self.proprio_hist_buf[:] = torch.cat([prev, cur], dim=1)

    # ---------- Utils ----------
    def unscale(self, x, lower, upper):
        return (2.0 * x - upper - lower) / (upper - lower)

    # ---------- Deploy ----------
    def deploy(self):
        # 1) Allegro I/O 노드 시작
        self.allegro = start_allegro_io(side='right')

        hz = 20

        # 2) 초기 셋업 루프
        warmup = hz * 4
        for t in range(warmup):
            tprint(f"setup {t} / {warmup}")
            pose = _reorder_imrt2timr(np.array(self.init_pose, dtype=np.float64))
            self.allegro.command_joint_position(pose)
            time.sleep(1/hz)

        # 3) 첫 관측
        q_pos = self.allegro.poll_joint_position(wait=True, timeout=5.0)
        if q_pos is None:
            print("❌ failed to read joint state.")
            stop_allegro_io(self.allegro)
            return

        ros1_q = _reorder_timr2imrt(q_pos)
        hora_q = _obs_allegro2hora(ros1_q)
        obs_q = torch.from_numpy(hora_q.astype(np.float32)).to(self.device)  # ← device 통일

        # q값은 normalize(=unscale) 해서 버퍼에 저장 // isaacgym 순서
        self.cur_obs_buf = self.unscale(obs_q, self.allegro_dof_lower, self.allegro_dof_upper)[None]
        # prev_target은 실측(rad)
        self.prev_target = obs_q[None].clone()

        # obs_buf 초기화
        for i in range(3):
            print(f"init fill: {i*32}:{i*32+16}, {i*32+16}:{i*32+32}")
            self.obs_buf[:, i*32:i*32+16] = self.cur_obs_buf      # q_t
            self.obs_buf[:, i*32+16:i*32+32] = self.prev_target   # a_{t-1}

        # proprio_hist_buf 초기화
        self.proprio_hist_buf[:, :, :16] = self.cur_obs_buf
        self.proprio_hist_buf[:, :, 16:32] = self.prev_target

        timestep = 0
        print("Deployment started. Press Ctrl+C to stop.")
        try:
            while True:
                loop_start = time.perf_counter()

                # Normalize the observation buffer (device 동일)
                self.obs_buf = self.running_mean_std(self.obs_buf.clone())

                input_dict = {
                    "obs": self.obs_buf,
                    "proprio_hist": self.sa_mean_std(self.proprio_hist_buf.clone()),
                }
                action = torch.clamp(self.model.act_inference(input_dict), -1.0, 1.0)

                # control
                self.pre_physics_step(action)

                # 명령 전송: 하드웨어 I/O 직전에만 CPU로 내림
                cmd = self.cur_target.detach().to('cpu').numpy()[0]
                ros1 = _action_hora2allegro(cmd)
                ros2 = _reorder_imrt2timr(ros1)
                self.allegro.command_joint_position(ros2)

                # 관측 업데이트 / o_{t+1}
                q_pos = self.allegro.poll_joint_position(wait=True, timeout=0.2)
                ros1_q = _reorder_timr2imrt(q_pos)
                hora_q = _obs_allegro2hora(ros1_q)
                obs_q = torch.from_numpy(hora_q.astype(np.float32)).to(self.device)  # ← device 통일

                self.post_physics_step(obs_q)  # buffer 업데이트

                time.sleep(0.03)  # ~20 Hz
                timestep += 1

                freq = 1.0 / (time.perf_counter() - loop_start)
                print(f"Hz={freq:.2f}")

        except KeyboardInterrupt:
            print("KeyboardInterrupt, stopping...")
        finally:
            # 안전 정지
            try:
                self.allegro.go_safe()
            except Exception:
                pass
            stop_allegro_io(self.allegro)
            print("🧠 Deployment stopped cleanly.")

    # ---------- checkpoint restore ----------
    def restore(self, fn):
        # 체크포인트 로딩 시에도 device 매핑 통일
        checkpoint = torch.load(fn, map_location=self.device)
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        self.model.load_state_dict(checkpoint["model"])
        self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])
