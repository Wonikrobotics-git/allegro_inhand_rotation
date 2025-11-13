# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ProprioAdaptTConv(nn.Module):
    """
    고유수용성 감각 이력 인코더 (Proprioceptive History Encoder).
    Stage 2에서 사용되며, 로봇의 과거 움직임 데이터(proprioceptive history)를 입력받아
    1D 컨볼루션(Temporal Aggregation)을 통해 시간적 특징을 추출하고,
    이를 통해 환경의 숨겨진 특성(extrinsic)을 추론하는 잠재 벡터를 생성합니다.
    """
    def __init__(self):
        super(ProprioAdaptTConv, self).__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(16 + 16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(32 * 3, 8)

    def forward(self, x):
        x = self.channel_transform(x)  # (N, 50, 32)
        x = x.permute((0, 2, 1))  # (N, 32, 50)
        x = self.temporal_aggregation(x)  # (N, 32, 3)
        x = self.low_dim_proj(x.flatten(1))
        return x


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop("actions_num")
        input_shape = kwargs.pop("input_shape")
        self.units = kwargs.pop("actor_units")
        self.priv_mlp = kwargs.pop("priv_mlp_units")
        mlp_input_shape = input_shape[0]

        out_size = self.units[-1]
        self.priv_info = kwargs["priv_info"]
        self.priv_info_stage2 = kwargs["proprio_adapt"]

        # --- 인코더 정의 부분 ---
        if self.priv_info:
            mlp_input_shape += self.priv_mlp[-1]
            # 1. 특권 정보 인코더 (Privileged Information Encoder)
            #    - 역할: 환경의 실제 물리 정보(priv_info)를 잠재 벡터로 변환합니다.
            #    - Stage 1에서는 정책 학습에 직접 사용되고, Stage 2에서는 '정답'을 제공하는 선생님 역할을 합니다.
            self.env_mlp = MLP(units=self.priv_mlp, input_size=kwargs["priv_info_dim"])

            if self.priv_info_stage2:
                # 2. 고유수용성 감각 이력 인코더 (Proprioceptive History Encoder)
                #    - 역할: 로봇의 과거 움직임(proprio_hist)만으로 환경 정보를 '추론'하는 잠재 벡터를 생성합니다.
                #    - Stage 2에서만 학습 및 사용됩니다.
                self.adapt_tconv = ProprioAdaptTConv()

        # 3. 정책 상태 인코더 (Policy State Encoder)
        #    - 역할: 일반 관측 정보(obs)와 위에서 처리된 환경 잠재 벡터를 결합하여,
        #            최종 행동과 가치를 결정하는 데 사용될 '상태(state)'를 생성합니다.
        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape) # actor network
        self.value = torch.nn.Linear(out_size, 1)                          # critic network
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(
            torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
            requires_grad=True,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            "neglogpacs": -distr.log_prob(selected_action).sum(
                1
            ),  # self.neglogp(selected_action, mu, sigma, logstd),
            "values": value,
            "actions": selected_action,
            "mus": mu,
            "sigmas": sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        obs = obs_dict["obs"]
        extrin, extrin_gt = None, None

        # --- 인코더 사용 및 정보 결합 부분 ---
        if self.priv_info:
            # Stage 2: 적응 단계 (고유수용성 감각 이력 인코더 사용)
            if self.priv_info_stage2:
                # 2. 고유수용성 감각 이력 인코더가 환경 정보를 '추론' (extrin)
                extrin = self.adapt_tconv(obs_dict["proprio_hist"])

                # 1. 특권 정보 인코더가 '정답' 환경 정보 생성 (extrin_gt)
                #    (학습 시에만 priv_info가 제공됨)
                extrin_gt = (
                    self.env_mlp(obs_dict["priv_info"])
                    if "priv_info" in obs_dict
                    else extrin
                )
                extrin_gt = torch.tanh(extrin_gt)
                extrin = torch.tanh(extrin)

                # 추론된 환경 정보(extrin)를 일반 관측(obs)과 결합
                obs = torch.cat([obs, extrin], dim=-1)
            # Stage 1: PPO 학습 단계 (특권 정보 인코더만 사용)
            else:
                # 1. 특권 정보 인코더가 환경 정보를 '직접' 생성 (extrin)
                extrin = self.env_mlp(obs_dict["priv_info"])
                extrin = torch.tanh(extrin)

                # 생성된 환경 정보를 일반 관측(obs)과 결합
                obs = torch.cat([obs, extrin], dim=-1)

        # 3. 정책 상태 인코더가 결합된 정보를 최종 '상태' 벡터로 변환
        x = self.actor_mlp(obs)

        # 최종 상태 벡터를 사용하여 가치(value)와 행동(mu)을 출력
        value = self.value(x)
        mu = self.mu(x)
        sigma = self.sigma
        return mu, mu * 0 + sigma, value, extrin, extrin_gt

    def forward(self, input_dict):
        prev_actions = input_dict.get("prev_actions", None)
        rst = self._actor_critic(input_dict)
        mu, logstd, value, extrin, extrin_gt = rst
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            "prev_neglogp": torch.squeeze(prev_neglogp),
            "values": value,
            "entropy": entropy,
            "mus": mu,
            "sigmas": sigma,
            "extrin": extrin,
            "extrin_gt": extrin_gt,
        }
        return result
