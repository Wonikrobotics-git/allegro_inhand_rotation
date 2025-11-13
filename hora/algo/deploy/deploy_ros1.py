# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
import torch
from hora.utils.misc import tprint


def _obs_allegro2hora(obses):
    obs_index = obses[0:4]
    obs_middle = obses[4:8]
    obs_ring = obses[8:12]
    obs_thumb = obses[12:16]
    obses = np.concatenate([obs_index, obs_thumb, obs_middle, obs_ring]).astype(np.float32)
    return obses


def _action_hora2allegro(actions):
    cmd_act = actions.copy()
    cmd_act[[4, 5, 6, 7]] = actions[[8, 9, 10, 11]]
    cmd_act[[12, 13, 14, 15]] = actions[[4, 5, 6, 7]]
    cmd_act[[8, 9, 10, 11]] = actions[[12, 13, 14, 15]]
    return cmd_act


class HardwarePlayer(object):
    def __init__(self, config):
        self.action_scale = 1 / 24
        self.actions_num = 16
        self.device = 'cuda'

        obs_shape = (96,)
        net_config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'actor_units': [512, 256, 128],
            'priv_mlp_units': [256, 128, 8],
            'priv_info': True,
            'proprio_adapt': True,
            'priv_info_dim': 9,
        }

        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()
        self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd((30, 32)).to(self.device)
        self.sa_mean_std.eval()

        # hand setting
        self.init_pose = [
            0.0627, 1.2923, 0.3383, 0.1088,  # index
            0.0724, 1.1983, 0.1551, 0.1499, # middle
            0.1343, 1.1736, 0.5355, 0.2164, # ring
            1.1202, 1.1374, 0.8535, -0.0852, # thumb
        ]
        self.allegro_dof_lower = torch.from_numpy(np.array([
            -0.4700, -0.1960, -0.1740, -0.2270,
            0.2630, -0.1050, -0.1890, -0.1620,
            -0.4700, -0.1960, -0.1740, -0.2270,
            -0.4700, -0.1960, -0.1740, -0.2270
        ])).to(self.device)
        self.allegro_dof_upper = torch.from_numpy(np.array([
            0.4700, 1.6100, 1.7090, 1.6180,
            1.3960, 1.1630, 1.6440, 1.7190,
            0.4700, 1.6100, 1.7090, 1.6180,
            0.4700, 1.6100, 1.7090, 1.6180
        ])).to(self.device)

    def deploy(self):
        import rospy
        from sensor_msgs.msg import JointState  # <<< NEW: for gap topic
        from hora.algo.deploy.robots.allegro import Allegro

        # try to set up rospy
        rospy.init_node('example')
        allegro = Allegro(hand_topic_prefix='allegroHand_0')
        # Wait for connections.
        rospy.sleep(0.5)

        # <<< NEW: publisher for the command-vs-current gap
        # Publishes a JointState where `position` holds (command - actual)
        gap_pub = rospy.Publisher('allegroHand_0/position_gap', JointState, queue_size=10)

        hz = 20
        ros_rate = rospy.Rate(hz)

        # command to the initial position
        for t in range(hz * 4):
            tprint(f'setup {t} / {hz * 4}')
            allegro.command_joint_position(self.init_pose)
            _obses_allegro, _ = allegro.poll_joint_position(wait=True)
            ros_rate.sleep()

        # get first observation
        obses_allegro, _ = allegro.poll_joint_position(wait=True)  # keep allegro-order as well
        obses_hora = _obs_allegro2hora(obses_allegro)

        # hardware deployment buffer
        obs_buf = torch.from_numpy(np.zeros((1, 16 * 3 * 2)).astype(np.float32)).cuda()
        proprio_hist_buf = torch.from_numpy(np.zeros((1, 30, 16 * 2)).astype(np.float32)).cuda()

        def unscale(x, lower, upper):
            return (2.0 * x - upper - lower) / (upper - lower)

        obses = torch.from_numpy(obses_hora.astype(np.float32)).cuda()
        prev_target = obses[None].clone()
        cur_obs_buf = unscale(obses, self.allegro_dof_lower, self.allegro_dof_upper)[None]

        for i in range(3):
            print(f"init fill: {i*16+0}:{i*16+16}, {i*16+16}:{i*16+32}")
            obs_buf[:, i*16+0:i*16+16] = cur_obs_buf.clone()     # joint position (scaled)
            obs_buf[:, i*16+16:i*16+32] = prev_target.clone()     # current target

        proprio_hist_buf[:, :, :16] = cur_obs_buf.clone()
        proprio_hist_buf[:, :, 16:32] = prev_target.clone()

        while not rospy.is_shutdown():
            start_time = rospy.get_time()

            obs = self.running_mean_std(obs_buf.clone())
            input_dict = {
                'obs': obs,
                'proprio_hist': self.sa_mean_std(proprio_hist_buf.clone()),
            }
            action = self.model.act_inference(input_dict)
            action = torch.clamp(action, -1.0, 1.0)

            target = prev_target + self.action_scale * action
            target = torch.clip(target, self.allegro_dof_lower, self.allegro_dof_upper)
            prev_target = target.clone()

            # interact with the hardware
            commands_hora = target.cpu().numpy()[0]          # hora order
            commands_allegro = _action_hora2allegro(commands_hora)  # convert to allegro order
            allegro.command_joint_position(commands_allegro) # send command
            ros_rate.sleep()  # keep 20 Hz command

            # get o_{t+1}
            raw_obs_allegro, torques = allegro.poll_joint_position(wait=True)  # keep allegro order
            # <<< NEW: compute and publish gap in allegro order
            try:
                gap = np.asarray(commands_allegro, dtype=np.float32) - np.asarray(raw_obs_allegro, dtype=np.float32)
                js = JointState()
                js.header.stamp = rospy.Time.now()
                # (선택) joint 이름을 알고 있으면 채워주세요. 비워도 됨.
                # js.name = ["ah_joint00", "ah_joint01", ...]  # 필요 시
                js.position = gap.tolist()  # gap = command - current
                gap_pub.publish(js)
            except Exception as e:
                rospy.logwarn(f"[position_gap] publish failed: {e}")

            # 계속 기존 파이프라인 진행 (학습 입력용 버퍼 업데이트)
            obses_hora = _obs_allegro2hora(raw_obs_allegro)
            obses = torch.from_numpy(obses_hora.astype(np.float32)).cuda()

            cur_obs_buf = unscale(obses, self.allegro_dof_lower, self.allegro_dof_upper)[None]
            prev_obs_buf = obs_buf[:, 32:].clone()
            obs_buf[:, :64] = prev_obs_buf
            obs_buf[:, 64:80] = cur_obs_buf.clone()
            obs_buf[:, 80:96] = target.clone()

            priv_proprio_buf = proprio_hist_buf[:, 1:30, :].clone()
            cur_proprio_buf = torch.cat([cur_obs_buf, target.clone()], dim=-1)[:, None]
            proprio_hist_buf[:] = torch.cat([priv_proprio_buf, cur_proprio_buf], dim=1)

            end_time = rospy.get_time()
            tprint(f'loop: {1/(end_time - start_time):.2f} Hz')

    def restore(self, fn):
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.model.load_state_dict(checkpoint['model'])
        self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])


# """
# rosrun rqt_plot rqt_plot \
# /allegroHand_0/joint_states/effort[0] \
# /allegroHand_0/joint_states/effort[1] \
# /allegroHand_0/joint_states/effort[2] \
# /allegroHand_0/joint_states/effort[3] \
# /allegroHand_0/joint_states/effort[4] \
# /allegroHand_0/joint_states/effort[5] \
# /allegroHand_0/joint_states/effort[6] \
# /allegroHand_0/joint_states/effort[7] \
# /allegroHand_0/joint_states/effort[8] \
# /allegroHand_0/joint_states/effort[9] \
# /allegroHand_0/joint_states/effort[10] \
# /allegroHand_0/joint_states/effort[11] \
# /allegroHand_0/joint_states/effort[12] \
# /allegroHand_0/joint_states/effort[13] \
# /allegroHand_0/joint_states/effort[14] \
# /allegroHand_0/joint_states/effort[15]
# """
