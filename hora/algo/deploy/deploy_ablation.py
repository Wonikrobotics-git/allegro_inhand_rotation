# 전역 플래그 설정
PLOT_ACT = True
PLOT_OBS = False
PLOT_ERR = True

# --------------------------------------------------------
# Ablation for check action process (using PyQtGraph)
# --------------------------------------------------------

import numpy as np
from collections import deque
import time

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
import torch
from hora.utils.misc import tprint

import rospy
from hora.algo.deploy.robots.allegro import Allegro


class HardwarePlayer(object):
    def __init__(self, config):
        self.action_scale = 1 / 24
        self.actions_num = 16
        self.device = "cuda"

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
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()
        self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd((30, 32)).to(self.device)
        self.sa_mean_std.eval()

        self.init_pose = [
            0.0627, 1.2923, 0.3383, 0.1088,
            0.0724, 1.1983, 0.1551, 0.1499,
            0.1343, 1.1736, 0.5355, 0.2164,
            1.1202, 1.1374, 0.8535, -0.0852,
        ]

        self.allegro_dof_lower = torch.from_numpy(
            np.array([
                -0.4700, -0.1960, -0.1740, -0.2270,
                 0.2630, -0.1050, -0.1890, -0.1620,
                -0.4700, -0.1960, -0.1740, -0.2270,
                -0.4700, -0.1960, -0.1740, -0.2270,
            ])
        ).to(self.device)
        self.allegro_dof_upper = torch.from_numpy(
            np.array([
                 0.4700, 1.6100, 1.7090, 1.6180,
                 1.3960, 1.1630, 1.6440, 1.7190,
                 0.4700, 1.6100, 1.7090, 1.6180,
                 0.4700, 1.6100, 1.7090, 1.6180,
            ])
        ).to(self.device)

        # 제어와 플로팅 변수
        self.current_joint_position = None
        self.current_joint_target = None
        self.current_action_raw = None
        self.current_action_clipped = None
        self.current_error = None  # 각 채널별 에러 (절댓값 차이)

        # Plotting buffers (최근 500 스텝)
        self.plot_act_raw_deque = deque(maxlen=500)
        self.plot_act_clipped_deque = deque(maxlen=500)
        self.plot_act_joint_target_deque = deque(maxlen=500)
        self.plot_obs_joint_position_deque = deque(maxlen=500)
        self.plot_obs_joint_target_deque = deque(maxlen=500)
        self.plot_error_deque = deque(maxlen=500)  # 에러 데이터를 저장
        self.time_steps = deque(maxlen=500)

        self.allegro = None

        # PyQtGraph 윈도우와 curve 객체들 (초기화는 deploy()에서)
        self.win_act = None      # "Action and Command" 윈도우
        self.act_curves = []     # 각 채널별로 [curve_raw, curve_clipped, curve_target]
        self.win_obs = None      # "Observation (Joint Degrees)" 윈도우
        self.obs_curves = []     # 각 채널별로 [curve_position, curve_target]
        self.win_err = None      # "Error (Absolute Difference)" 윈도우
        self.err_curves = []     # 16채널 각각의 에러 curve

    def preprocess_data(self, data):
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        else:
            return np.array(data)

    def init_plots(self):
        # Action and Command 플롯 윈도우 생성 (PLOT_ACT 플래그에 따라)
        if PLOT_ACT:
            self.win_act = pg.GraphicsLayoutWidget(title="Action and Command")
            self.win_act.resize(1200, 900)
            self.win_act.show()
            self.act_curves = []
            for channel in range(16):
                p = self.win_act.addPlot(row=channel, col=0)
                p.showGrid(x=True, y=True)
                p.setLabel('left', f"{channel}")
                if channel == 15:
                    p.setLabel('bottom', "Time step")
                if channel == 0:
                    p.setTitle("Action and Command")
                    p.addLegend(offset=(10, 10))
                raw_curve = p.plot(pen='r', name="raw")
                clipped_curve = p.plot(pen='g', name="clipped")
                target_curve = p.plot(pen='b', name="target")
                self.act_curves.append([raw_curve, clipped_curve, target_curve])

        # Observation 플롯 윈도우 생성 (PLOT_OBS 플래그에 따라)
        if PLOT_OBS:
            self.win_obs = pg.GraphicsLayoutWidget(title="Observation (Joint Degrees)")
            self.win_obs.resize(1200, 900)
            self.win_obs.show()
            self.obs_curves = []
            for channel in range(16):
                p = self.win_obs.addPlot(row=channel, col=0)
                p.showGrid(x=True, y=True)
                p.setLabel('left', f"{channel}")
                if channel == 15:
                    p.setLabel('bottom', "Time step")
                if channel == 0:
                    p.setTitle("Observation (Joint Degrees)")
                    p.addLegend(offset=(10, 10))
                pos_curve = p.plot(pen='r', name="position")
                target_curve = p.plot(pen='b', name="target")
                self.obs_curves.append([pos_curve, target_curve])

        # Error 플롯 윈도우 생성 (PLOT_ERR 플래그에 따라)
        if PLOT_ERR:
            self.win_err = pg.GraphicsLayoutWidget(title="Error (Absolute Difference per Channel)")
            self.win_err.resize(1200, 400)  # 한 줄로 표시
            self.win_err.show()
            p_err = self.win_err.addPlot(row=0, col=0)
            p_err.showGrid(x=True, y=True)
            p_err.setLabel('left', "Absolute Error")
            p_err.setLabel('bottom', "Time step")
            p_err.setTitle("Error (Absolute Difference per Channel)")
            p_err.addLegend(offset=(-10, 10))
            self.err_curves = []
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w', 'k',
                      'orange', 'purple', 'brown', 'pink', 'lime', 'teal', 'navy', 'gray']
            for channel in range(16):
                color = colors[channel % len(colors)]
                curve = p_err.plot(pen=color, name=f"Ch {channel}")
                self.err_curves.append(curve)

    def update_plots(self):
        times = np.array(self.time_steps)
        if times.size == 0:
            return

        if PLOT_ACT:
            act_raw = np.array(self.plot_act_raw_deque)
            act_clipped = np.array(self.plot_act_clipped_deque)
            act_target = np.array(self.plot_act_joint_target_deque)
            for channel in range(16):
                self.act_curves[channel][0].setData(times, act_raw[:, channel])
                self.act_curves[channel][1].setData(times, act_clipped[:, channel])
                self.act_curves[channel][2].setData(times, act_target[:, channel])

        if PLOT_OBS:
            obs_pos = np.array(self.plot_obs_joint_position_deque)
            obs_target = np.array(self.plot_obs_joint_target_deque)
            for channel in range(16):
                self.obs_curves[channel][0].setData(times, obs_pos[:, channel])
                self.obs_curves[channel][1].setData(times, obs_target[:, channel])

        if PLOT_ERR:
            obs_pos = np.array(self.plot_obs_joint_position_deque)
            obs_target = np.array(self.plot_obs_joint_target_deque)
            err = np.abs(obs_pos - obs_target)  # 각 채널별 절댓값 차이
            for channel in range(16):
                self.err_curves[channel].setData(times, err[:, channel])

        QtWidgets.QApplication.processEvents()

    def _action_hora2allegro(self, actions):
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
            actions = np.asarray(actions).flatten()
            cmd_act = actions.copy()
            temp = actions[[4, 5, 6, 7]].copy()
            cmd_act[[4, 5, 6, 7]] = actions[[8, 9, 10, 11]]
            cmd_act[[12, 13, 14, 15]] = temp
            cmd_act[[8, 9, 10, 11]] = actions[[12, 13, 14, 15]]
            return cmd_act

    def _obs_allegro2hora(self, obses):
        obs_index = obses[0:4]
        obs_middle = obses[4:8]
        obs_ring = obses[8:12]
        obs_thumb = obses[12:16]
        obses = np.concatenate([obs_index, obs_thumb, obs_middle, obs_ring]).astype(np.float32)
        return obses

    def print_4x4_table(self, title, arr):
        print(f"{title}:")
        print("---------------------------------------------")
        for row in range(4):
            for col in range(4):
                idx = row * 4 + col
                print(f"J{idx:02d}: {arr[idx]:8.4f}  ", end="")
            print()
        print("---------------------------------------------\n")

    def deploy(self):
        rospy.init_node("hora")
        self.allegro = Allegro(hand_topic_prefix="allegroHand_0")
        rospy.sleep(0.5)

        # 플롯 객체 초기화 (플래그에 따라)
        self.init_plots()

        target_period = 1.0 / 20  # 20Hz, 0.05초
        ros_rate = rospy.Rate(20)
        for t in range(20 * 4):
            tprint(f"setup {t} / {20 * 4}")
            self.allegro.command_joint_position(self.init_pose)
            obses, _ = self.allegro.poll_joint_position(wait=True)
            ros_rate.sleep()

        obses, _ = self.allegro.poll_joint_position(wait=True)
        self.current_joint_position = np.array(obses).copy()
        obses = self._obs_allegro2hora(obses)

        obs_buf = torch.from_numpy(np.zeros((1, 16 * 3 * 2), dtype=np.float32)).cuda()
        proprio_hist_buf = torch.from_numpy(np.zeros((1, 30, 16 * 2), dtype=np.float32)).cuda()

        def unscale(x, lower, upper):
            return (2.0 * x - upper - lower) / (upper - lower)

        obses = torch.from_numpy(obses.astype(np.float32)).cuda()
        prev_target = obses[None].clone()
        cur_obs_buf = unscale(obses, self.allegro_dof_lower, self.allegro_dof_upper)[None]

        for i in range(3):
            obs_buf[:, i * 16: i * 16 + 16] = cur_obs_buf.clone()
            obs_buf[:, i * 16 + 16: i * 16 + 32] = prev_target.clone()

        proprio_hist_buf[:, :, :16] = cur_obs_buf.clone()
        proprio_hist_buf[:, :, 16:32] = prev_target.clone()

        timestep = 0
        print("Deployment started. Press Ctrl+C to stop.")
        try:
            while not rospy.is_shutdown():
                loop_start = time.perf_counter()

                # 제어 처리
                obs = self.running_mean_std(obs_buf.clone())
                input_dict = {
                    "obs": obs,
                    "proprio_hist": self.sa_mean_std(proprio_hist_buf.clone()),
                }
                action = self.model.act_inference(input_dict)
                self.current_action_raw = action.clone()

                action = torch.clamp(action, -1.0, 1.0)
                self.current_action_clipped = torch.clamp(self.current_action_raw.clone(), min=-1.0, max=1.0)

                target = prev_target + self.action_scale * action
                target = torch.clip(target, self.allegro_dof_lower, self.allegro_dof_upper)
                prev_target = target.clone()

                commands = target.cpu().numpy()[0]
                commands = self._action_hora2allegro(commands)
                self.current_joint_target = commands.copy()
                self.allegro.command_joint_position(commands)

                # 관측값 업데이트
                obses, _ = self.allegro.poll_joint_position(wait=True)
                self.current_joint_position = np.array(obses).copy()
                obses = self._obs_allegro2hora(obses)
                obses = torch.from_numpy(obses.astype(np.float32)).cuda()

                cur_obs_buf = unscale(obses, self.allegro_dof_lower, self.allegro_dof_upper)[None]
                prev_obs_buf = obs_buf[:, 32:].clone()
                obs_buf[:, :64] = prev_obs_buf
                obs_buf[:, 64:80] = cur_obs_buf.clone()
                obs_buf[:, 80:96] = target.clone()

                priv_proprio_buf = proprio_hist_buf[:, 1:30, :].clone()
                cur_proprio_buf = torch.cat([cur_obs_buf, target.clone()], dim=-1)[:, None]
                proprio_hist_buf[:] = torch.cat([priv_proprio_buf, cur_proprio_buf], dim=1)

                # Plotting 버퍼 업데이트
                raw = np.squeeze(self.preprocess_data(self.current_action_raw))
                clipped = np.squeeze(self.preprocess_data(self.current_action_clipped))
                target_np = np.squeeze(self.preprocess_data(self.current_joint_target))
                joint_position = np.squeeze(self.preprocess_data(self.current_joint_position))

                self.plot_act_raw_deque.append(raw)
                self.plot_act_clipped_deque.append(clipped)
                self.plot_act_joint_target_deque.append(target_np)
                self.plot_obs_joint_position_deque.append(joint_position)
                self.plot_obs_joint_target_deque.append(target_np)
                # 에러 계산: 각 채널별 절댓값 차이
                self.current_error = np.abs(joint_position - target_np)  # shape (16,)
                self.plot_error_deque.append(self.current_error)
                self.time_steps.append(timestep)

                timestep += 1

                # 플롯 업데이트는 200회마다 실행 (플래그에 따라)
                if timestep % 200 == 0:
                    self.update_plots()

                ros_rate.sleep()
                print(f"Control loop iteration took {1 / (time.perf_counter() - loop_start):.2f} Hz")
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught. Stopping deployment.")

        print("Hardware deployment stopped.")

    def restore(self, fn):
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        self.model.load_state_dict(checkpoint["model"])
        self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])
