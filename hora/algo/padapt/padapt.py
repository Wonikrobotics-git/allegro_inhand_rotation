# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import time
import torch
import numpy as np
from termcolor import cprint

from hora.utils.misc import AverageScalarMeter, tprint
from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
from tensorboardX import SummaryWriter


class ProprioAdapt(object):
    def __init__(self, env, output_dir, full_config):
        self.device = full_config["rl_device"]
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config["num_actors"]
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        self.action_space = self.env.action_space
        self.actions_num = self.action_space.shape[0]
        # ---- Priv Info ----
        self.priv_info = self.ppo_config["priv_info"]
        self.priv_info_dim = self.ppo_config["priv_info_dim"]
        self.proprio_adapt = self.ppo_config["proprio_adapt"]
        self.proprio_hist_dim = self.env.prop_hist_len
        # ---- Model ----
        net_config = {
            "actor_units": self.network_config.mlp.units,
            "priv_mlp_units": self.network_config.priv_mlp.units,
            "actions_num": self.actions_num,
            "input_shape": self.obs_shape,
            "priv_info": self.priv_info,
            "proprio_adapt": self.proprio_adapt,
            "priv_info_dim": self.priv_info_dim,
        }
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd((self.proprio_hist_dim, 32)).to(self.device)
        self.sa_mean_std.train()
        # ---- Output Dir ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, "stage2_nn")
        self.tb_dir = os.path.join(self.output_dir, "stage2_tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        writer = SummaryWriter(self.tb_dir)
        self.writer = writer
        self.direct_info = {}
        # ---- Misc ----
        self.batch_size = self.num_actors
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        self.best_rewards = -10000
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config["max_agent_steps"] # get rid of hardcoded 1e9
        # ---- Optim ----
        # --- Stage 2의 핵심: 적응 모듈만 학습 ---
        # Stage 1에서 학습된 모델의 대부분의 파라미터는 동결(학습되지 않도록 설정)합니다.
        adapt_params = []
        for name, p in self.model.named_parameters():
            # 이름에 'adapt_tconv'가 포함된, 즉 '적응 모듈'에 해당하는 파라미터만 학습 대상으로 선택합니다.
            if "adapt_tconv" in name:
                adapt_params.append(p)
            else:
                p.requires_grad = False
        # 선택된 적응 모듈 파라미터만 최적화(Adam)하도록 설정합니다.
        self.optim = torch.optim.Adam(adapt_params, lr=3e-4)
        # ---- Training Misc
        self.internal_counter = 0
        self.latent_loss_stat = 0
        self.loss_stat_cnt = 0
        batch_size = self.num_actors
        self.step_reward = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        self.step_length = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )

    def set_eval(self):
        self.model.eval()
        self.running_mean_std.eval()
        self.sa_mean_std.eval()

    def test(self):
        # --- 배포(테스트) 단계 ---
        # 이 단계에서는 더 이상 특권 정보(priv_info)를 사용하지 않습니다.
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            # 모델에 현재 관측(obs)과 고유수용성 감각 이력(proprio_hist)만을 입력합니다.
            input_dict = {
                "obs": self.running_mean_std(obs_dict["obs"]),
                "proprio_hist": self.sa_mean_std(obs_dict["proprio_hist"].detach()),
            }
            # 학습된 적응 모듈이 proprio_hist를 바탕으로 환경 잠재 벡터를 *추론*하고,
            # 정책(Actor)은 이 추론된 정보를 활용하여 최종 행동(mu)을 결정합니다.
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)

    def train(self):
        _t = time.time()
        _last_t = time.time()

        obs_dict = self.env.reset()
        self.agent_steps += self.batch_size
        while self.agent_steps <= self.max_agent_steps: # 1e9
            # --- 적응 모듈 학습 ---
            # 모델에 현재 관측, 특권 정보, 고유수용성 감각 이력을 모두 입력합니다.
            input_dict = {
                "obs": self.running_mean_std(obs_dict["obs"]).detach(),
                "priv_info": obs_dict["priv_info"],
                "proprio_hist": self.sa_mean_std(obs_dict["proprio_hist"].detach()),
            }
            # 모델은 행동(mu)과 함께 두 개의 잠재 벡터를 출력합니다:
            # e: '적응 모듈'이 고유수용성 감각 이력(proprio_hist)을 기반으로 *예측*한 환경 잠재 벡터
            # e_gt: 'Privilege Encoder'가 실제 특권 정보(priv_info)를 기반으로 *생성*한 정답 환경 잠재 벡터
            mu, _, _, e, e_gt = self.model._actor_critic(input_dict)

            # 손실 계산: 예측된 잠재 벡터(e)가 정답 잠재 벡터(e_gt)를 모방하도록 평균 제곱 오차(MSE) 손실을 계산합니다.
            # e_gt.detach()를 사용하여 이 부분으로는 그래디언트가 흐르지 않도록 합니다. 즉, 오직 'e'를 생성하는 적응 모듈만 학습됩니다.
            loss = ((e - e_gt.detach()) ** 2).mean()

            # 계산된 손실을 바탕으로 적응 모듈의 파라미터를 업데이트합니다.
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # 환경과 상호작용하기 위해 계산된 행동(mu)을 사용합니다.
            mu = mu.detach()
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            self.agent_steps += self.batch_size

            # ---- statistics
            self.step_reward += r
            self.step_length += 1
            done_indices = done.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])

            not_dones = 1.0 - done.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            self.log_tensorboard()

            if self.agent_steps % 1e8 == 0:
                self.save(os.path.join(self.nn_dir, f"{self.agent_steps // 1e8}00m"))
                self.save(os.path.join(self.nn_dir, f"last"))

            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_rewards:
                self.save(os.path.join(self.nn_dir, f"best"))
                self.best_rewards = mean_rewards

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = (
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                f"Last FPS: {last_fps:.1f} | "
                f"Current Best: {self.best_rewards:.2f}"
            )
            tprint(info_string)

    def log_tensorboard(self):
        self.writer.add_scalar(
            "episode_rewards/step", self.mean_eps_reward.get_mean(), self.agent_steps
        )
        self.writer.add_scalar(
            "episode_lengths/step", self.mean_eps_length.get_mean(), self.agent_steps
        )
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f"{k}/frame", v, self.agent_steps)

    def restore_train(self, fn):
        checkpoint = torch.load(fn)
        cprint("careful, using non-strict matching", "red", attrs=["bold"])
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    def restore_test(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        self.model.load_state_dict(checkpoint["model"])
        self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])

    def save(self, name):
        weights = {
            "model": self.model.state_dict(),
        }
        if self.running_mean_std:
            weights["running_mean_std"] = self.running_mean_std.state_dict()
        if self.sa_mean_std:
            weights["sa_mean_std"] = self.sa_mean_std.state_dict()
        torch.save(weights, f"{name}.pth")
