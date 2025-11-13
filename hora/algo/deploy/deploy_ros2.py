#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import os
import time
import signal
# from typing import List, Optional

import numpy as np
import torch

# one hand (right)
from hora.algo.deploy.robots.allegro_ros2_one import start_allegro_io, stop_allegro_io
# TBD: two hands
# from hora.algo.deploy.robots.allegro_ros2 import start_allegro_io, stop_allegro_io


from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.utils.misc import tprint


# =========================================================
# 변환/재배열 유틸
# =========================================================

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
    # hora   : index, thumb, middle, ring
    return np.concatenate([o[0:4], o[12:16], o[4:8], o[8:12]]).astype(np.float64)


def _reorder_imrt2timr(imrt):
    # [ROS1] index-middle-ring-thumb → [ROS2] thumb-index-middle-ring
    return np.concatenate([imrt[12:16], imrt[0:12]]).astype(np.float64)


def _reorder_timr2imrt(timr):
    # [ROS2] thumb-index-middle-ring → [ROS1] index-middle-ring-thumb
    return np.concatenate([timr[4:16], timr[0:4]]).astype(np.float64)


# =========================================================
# 제어 에이전트 (Timer 기반)
# =========================================================

class HardwarePlayer:
    def __init__(self, hz: float = 20.0, device: str = "cuda"):
        torch.set_grad_enabled(False)
        self.hz = float(hz)
        self.device = device

        # model / rms
        obs_shape = (96,)
        net_config = {
            "actions_num": 16,
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

        # buffers
        self.obs_buf = torch.zeros((1, 96), dtype=torch.float32, device=self.device)
        self.proprio_hist_buf = torch.zeros((1, 30, 32), dtype=torch.float32, device=self.device)

        # limits
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

        # poses
        self.init_pose = [
            0.0627, 1.2923, 0.3383, 0.1088,
            0.0724, 1.1983, 0.1551, 0.1499,
            0.1343, 1.1736, 0.5355, 0.2164,
            1.1202, 1.1374, 0.8535, -0.0852,
        ]

        # state
        self.action_scale = 1.0 / 24.0
        self.prev_target = torch.zeros((1, 16), dtype=torch.float32, device=self.device)
        self.cur_target  = torch.zeros((1, 16), dtype=torch.float32, device=self.device)
        self._last_obs_q = None
        self._skipped = 0
        self._last_step_t = None

        # ros
        self.timer = None
        self.allegro = None

    # ---------- utils ----------
    @staticmethod
    def _unscale(x, lower, upper):
        return (2.0 * x - upper - lower) / (upper - lower)

    def _pre_physics_step(self, action):
        target = self.prev_target + self.action_scale * action
        self.cur_target = torch.clamp(target, min=self.allegro_dof_lower, max=self.allegro_dof_upper)
        self.prev_target = self.cur_target

    def _post_physics_step(self, obses):
        # 1) 현재 관측 정규화 (obses: (16,) 또는 (1,16) 텐서 on self.device)
        #    결과를 (1,16)로 보장
        cur_obs = self._unscale(
            obses.view(-1), self.allegro_dof_lower, self.allegro_dof_upper
        ).view(1, 16)

        # 2) obs_buf 롤링 (96 = 32*3)
        #    [0:64] <- [32:96],  [64:80] <- cur_obs,  [80:96] <- cur_target
        #    ⚠️ 겹침 방지: 소스 부분을 먼저 clone() 해서 임시 보관
        src64 = self.obs_buf[:, 32:96].clone()     # (1,64)
        self.obs_buf[:, 0:64] = src64              # 앞 64칸으로 당김
        self.obs_buf[:, 64:80] = cur_obs           # 현재 관측(정규화)
        self.obs_buf[:, 80:96] = self.cur_target   # 최신 타깃(rad)

        # 3) proprio_hist_buf 롤링 (T=30)
        #    [:, 0:-1, :] <- [:, 1:, :];  마지막 step에 [cur_obs | cur_target]
        #    ⚠️ 동일 텐서 내 겹침 방지: 소스 clone()
        src_hist = self.proprio_hist_buf[:, 1:, :].clone()  # (1,29,32)
        self.proprio_hist_buf[:, 0:-1, :] = src_hist
        self.proprio_hist_buf[:, -1, :16] = cur_obs
        self.proprio_hist_buf[:, -1, 16:32] = self.cur_target

    # ---------- timer callback ----------
    @torch.inference_mode()
    def _control_step(self):
        t0 = time.perf_counter()

        # 1) norm
        obs_norm = self.running_mean_std(self.obs_buf)

        # 2) inference
        input_dict = {
            "obs": obs_norm,
            "proprio_hist": self.sa_mean_std(self.proprio_hist_buf),
        }
        action = torch.clamp(self.model.act_inference(input_dict), -1.0, 1.0)

        # 3) update target
        self._pre_physics_step(action)

        # 4) publish command (CPU로만 내릴 때 변환)
        cmd = self.cur_target.detach().to("cpu").numpy()[0]
        ros1 = _action_hora2allegro(cmd)
        ros2 = _reorder_imrt2timr(ros1)
        self.allegro.command_joint_position(ros2)

        # 5) non-blocking obs update (드랍 시 마지막 유효 관측 사용)
        q_pos = self.allegro.poll_joint_position(wait=False, timeout=0.0)
        if q_pos is not None:
            ros1_q = _reorder_timr2imrt(q_pos)
            hora_q = _obs_allegro2hora(ros1_q)
            obs_q = torch.from_numpy(hora_q.astype(np.float32)).to(self.device)
            self._last_obs_q = obs_q
        else:
            obs_q = self._last_obs_q
            self._skipped += 1

        if obs_q is not None:
            self._post_physics_step(obs_q)

        # 6) light jitter log
        if self._last_step_t is None:
            self._last_step_t = t0
        else:
            dt = t0 - self._last_step_t
            self._last_step_t = t0
            # 5초마다 한 번만 출력
            if int(time.time()) % 5 == 0:
                hz_est = 1.0 / max(dt, 1e-6)
                print(f"[timer] {hz_est:.2f} Hz, skipped={self._skipped}")

    # ---------- deploy ----------
    def deploy(self):

        run_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"🧠 Starting HardwarePlayer deployment at {run_start_time}...")

        # ROS2 I/O 시작(백그라운드 실행기)
        self.allegro = start_allegro_io(side='right')

        # 워밍업(블로킹) — 하드웨어 settle
        warmup = int(self.hz * 4)
        for t in range(warmup):
            tprint(f"setup {t} / {warmup}")
            pose = _reorder_imrt2timr(np.array(self.init_pose, dtype=np.float64))
            self.allegro.command_joint_position(pose)
            time.sleep(1.0 / self.hz)

        # 첫 관측(블로킹 1회 — 초기화 안정)
        q_pos = self.allegro.poll_joint_position(wait=True, timeout=5.0)
        if q_pos is None:
            print("❌ failed to read joint state.")
            stop_allegro_io(self.allegro)
            return

        ros1_q = _reorder_timr2imrt(q_pos)
        hora_q = _obs_allegro2hora(ros1_q)
        obs_q = torch.from_numpy(hora_q.astype(np.float32)).to(self.device)
        self._last_obs_q = obs_q

        # buffers 초기화
        cur_obs_buf = self._unscale(obs_q, self.allegro_dof_lower, self.allegro_dof_upper)[None]
        self.prev_target = obs_q[None]
        for i in range(3):
            self.obs_buf[:, i*32:i*32+16] = cur_obs_buf
            self.obs_buf[:, i*32+16:i*32+32] = self.prev_target
        self.proprio_hist_buf[:, :, :16] = cur_obs_buf
        self.proprio_hist_buf[:, :, 16:32] = self.prev_target

        # Timer 등록 (정확 주기)
        period = 1.0 / self.hz
        self.timer = self.allegro.create_timer(period, self._control_step)
        print(f"Deployment started (timer-based {self.hz:.1f} Hz). Ctrl+C to stop.")

        # 메인 스레드: 시그널 처리 + 유지
        interrupted = False

        def _sigint(_sig, _frm):
            nonlocal interrupted
            interrupted = True
        signal.signal(signal.SIGINT, _sigint)

        try:
            while not interrupted:
                time.sleep(0.2)
        finally:
            try:
                if self.timer is not None:
                    self.timer.cancel()
            except Exception:
                pass
            try:
                self.allegro.go_safe()
            except Exception:
                pass
            stop_allegro_io(self.allegro)
            print("🧠 Deployment stopped cleanly.")

            run_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # Calculate elapsed time more clearly (HH:MM:SS) using the start/end timestamps
            fmt = "%Y-%m-%d %H:%M:%S"
            try:
                t0 = time.mktime(time.strptime(run_start_time, fmt))
                t1 = time.mktime(time.strptime(run_end_time, fmt))
                elapsed = int(round(t1 - t0))
                hrs, rem = divmod(elapsed, 3600)
                mins, secs = divmod(rem, 60)
                print(f"🧠 Total Running Time: {hrs:02d}:{mins:02d}:{secs:02d}")
            except Exception:
                # Fallback: print raw start/end strings if parsing fails
                print(f"🔥 Run started at {run_start_time}, ended at {run_end_time}")

    # ---------- checkpoint ----------
    def restore(self, fn):
        ckpt = torch.load(fn, map_location=self.device)
        self.running_mean_std.load_state_dict(ckpt["running_mean_std"])
        self.model.load_state_dict(ckpt["model"])
        self.sa_mean_std.load_state_dict(ckpt["sa_mean_std"])


# =========================================================
# 실행 예시
# =========================================================
if __name__ == "__main__":
    # 예: CUDA가 없으면 device="cpu"로 변경
    agent = HardwarePlayer(hz=20.0, device="cuda" if torch.cuda.is_available() else "cpu")
    # 필요한 경우 체크포인트 로드
    # agent.restore("/path/to/checkpoint.pth")
    agent.deploy()
