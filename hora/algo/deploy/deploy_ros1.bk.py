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
        from hora.algo.deploy.robots.allegro import Allegro
        # try to set up rospy
        rospy.init_node('example')
        allegro = Allegro(hand_topic_prefix='allegroHand_0')
        # Wait for connections.
        rospy.sleep(0.5)

        hz = 20
        ros_rate = rospy.Rate(hz)

        # command to the initial position
        for t in range(hz * 4):
            tprint(f'setup {t} / {hz * 4}')
            allegro.command_joint_position(self.init_pose)
            obses, _ = allegro.poll_joint_position(wait=True)
            ros_rate.sleep()

        obses, _ = allegro.poll_joint_position(wait=True)
        obses = _obs_allegro2hora(obses)
        # hardware deployment buffer
        obs_buf = torch.from_numpy(np.zeros((1, 16 * 3 * 2)).astype(np.float32)).cuda()
        proprio_hist_buf = torch.from_numpy(np.zeros((1, 30, 16 * 2)).astype(np.float32)).cuda()

        def unscale(x, lower, upper):
            return (2.0 * x - upper - lower) / (upper - lower)

        obses = torch.from_numpy(obses.astype(np.float32)).cuda()
        prev_target = obses[None].clone()
        cur_obs_buf = unscale(obses, self.allegro_dof_lower, self.allegro_dof_upper)[None]

        for i in range(3):
            # 0:16, 16:32
            # init fill: 16:32, 32:48
            # init fill: 32:48, 48:64

            print(f"init fill: {i*16+0}:{i*16+16}, {i*16+16}:{i*16+32}")
            obs_buf[:, i*16+0:i*16+16] = cur_obs_buf.clone()  # joint position
            obs_buf[:, i*16+16:i*16+32] = prev_target.clone()  # current target (obs_t-1 + s * act_t-1)

            # obs_buf[:, i*32:i*32+16] = cur_obs_buf.clone()  # joint position
            # obs_buf[:, i*32+16:i*32+32] = prev_target.clone()  # current target (obs_t-1 + s * act_t-1)

        proprio_hist_buf[:, :, :16] = cur_obs_buf.clone()
        proprio_hist_buf[:, :, 16:32] = prev_target.clone()

        while True:
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
            commands = target.cpu().numpy()[0]
            commands = _action_hora2allegro(commands)
            allegro.command_joint_position(commands)
            ros_rate.sleep()  # keep 20 Hz command
            # get o_{t+1}
            obses, torques = allegro.poll_joint_position(wait=True)
            obses = _obs_allegro2hora(obses)
            obses = torch.from_numpy(obses.astype(np.float32)).cuda()

            # --- Data collection for plotting ---
            # initialize storage on first iteration
            if 'plot_steps' not in locals():
                plot_steps = 300
                targets_list = []
                actions_list = []
                actions_raw_list = []
                obses_list = []
                gap_list = []
                plotted = False

            # store raw numpy copies (shape: 16,)
            targets_list.append(target.clone().cpu().numpy()[0])
            # store only the first `plot_steps` steps
            if len(targets_list) < plot_steps:
                # store action: raw (in [-1,1]) and scaled delta (radians)
                actions_raw_list.append(action.clone().cpu().numpy()[0])
                actions_list.append((self.action_scale * action).clone().cpu().numpy()[0])
                obses_list.append(obses.clone().cpu().numpy())
                # gap between target and observation
                gap = (target - obses).abs().clone().cpu().numpy()[0]
                gap_list.append(gap)
                targets_list.append(target.clone().cpu().numpy()[0])

            cur_obs_buf = unscale(obses, self.allegro_dof_lower, self.allegro_dof_upper)[None]
            prev_obs_buf = obs_buf[:, 32:].clone()
            obs_buf[:, :64] = prev_obs_buf
            obs_buf[:, 64:80] = cur_obs_buf.clone()
            obs_buf[:, 80:96] = target.clone()

            priv_proprio_buf = proprio_hist_buf[:, 1:30, :].clone()
            cur_proprio_buf = torch.cat([
                cur_obs_buf, target.clone()
            ], dim=-1)[:, None]
            proprio_hist_buf[:] = torch.cat([priv_proprio_buf, cur_proprio_buf], dim=1)

            end_time = rospy.get_time()
            tprint(f'loop: {1/(end_time - start_time):.2f} Hz')

            # if we've collected enough steps, plot once but keep running
            if len(targets_list) >= plot_steps and not plotted:
                tprint(f'Collected {plot_steps} steps, plotting (will continue running)...')
                try:
                    # convert to arrays: (steps, dofs)
                    targets_arr = np.array(targets_list)
                    actions_arr = np.array(actions_list)
                    actions_raw_arr = np.array(actions_raw_list)
                    obses_arr = np.array(obses_list)
                    gap_arr = np.array(gap_list)

                    steps = np.arange(targets_arr.shape[0])
                    n_dofs = targets_arr.shape[1]

                    # --- Plot 1: action (left, -1..1) and gap (right) per-DOF ---
                    fig1, axes1 = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
                    axes1 = axes1.flatten()
                    for d in range(n_dofs):
                        ax = axes1[d]
                        ax.plot(steps, actions_raw_arr[:, d], color='C0', label='action (raw)')
                        ax.set_ylim(-1.0, 1.0)
                        ax.axhline(0.0, color='k', linewidth=0.6, linestyle=':')
                        ax.set_title(f'DOF {d}')
                        ax.set_ylabel('action [-1,1]')
                        ax.grid(True, linewidth=0.3)

                        # twin axis for gap
                        ax_r = ax.twinx()
                        ax_r.plot(steps, gap_arr[:, d], color='C1', linestyle='--', label='|target-obses|')
                        ax_r.set_ylabel('gap (radians)')

                        if d == 0:
                            ax.legend(loc='upper left', fontsize='small')
                            ax_r.legend(loc='upper right', fontsize='small')

                    for dd in range(n_dofs, len(axes1)):
                        axes1[dd].axis('off')

                    fig1.suptitle('Action (raw) and gap per-DOF', fontsize=16)
                    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
                    out_fn1 = os.path.join(os.getcwd(), 'deploy_plot_action_gap_4x4.png')
                    plt.savefig(out_fn1)
                    plt.close(fig1)

                    # --- Plot 2: target and obses per-DOF ---
                    fig2, axes2 = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
                    axes2 = axes2.flatten()
                    for d in range(n_dofs):
                        ax = axes2[d]
                        ax.plot(steps, targets_arr[:, d], color='C2', label='target')
                        ax.plot(steps, obses_arr[:, d], color='C3', label='obses')
                        ax.set_title(f'DOF {d}')
                        if d % 4 == 0:
                            ax.set_ylabel('radians')
                        ax.grid(True, linewidth=0.3)
                        if d == 0:
                            ax.legend(fontsize='small')

                    for dd in range(n_dofs, len(axes2)):
                        axes2[dd].axis('off')

                    fig2.suptitle('Target and Obses per-DOF', fontsize=16)
                    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
                    out_fn2 = os.path.join(os.getcwd(), 'deploy_plot_target_obses_4x4.png')
                    plt.savefig(out_fn2)
                    plt.close(fig2)
                    tprint(f'Plots saved to: {out_fn1}, {out_fn2}')
                except Exception as e:
                    tprint(f'Failed to create plot: {e}')
                plotted = True

    def restore(self, fn):
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.model.load_state_dict(checkpoint['model'])
        self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])
