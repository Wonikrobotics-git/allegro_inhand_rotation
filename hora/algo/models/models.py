# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Modified by Wonik Robotics (2025)
# Adaptations for Allegro Hand V4 deployment
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
    Proprioceptive History Encoder.
    Used in Stage 2, it takes the robot's past movement data (proprioceptive history) as input,
    extracts temporal features through 1D convolution (Temporal Aggregation),
    and generates a latent vector that infers the hidden characteristics (extrinsic) of the environment.
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

        # --- Encoder Definition Part ---
        if self.priv_info:
            mlp_input_shape += self.priv_mlp[-1]
            # 1. Privileged Information Encoder
            #    - Role: Converts the actual physical information of the environment (priv_info) into a latent vector.
            #    - In Stage 1, it is used directly for policy learning, and in Stage 2, it acts as a teacher providing the 'ground truth'.
            self.env_mlp = MLP(units=self.priv_mlp, input_size=kwargs["priv_info_dim"])

            if self.priv_info_stage2:
                # 2. Proprioceptive History Encoder
                #    - Role: Generates a latent vector that 'infers' environment information solely from the robot's past movements (proprio_hist).
                #    - Trained and used only in Stage 2.
                self.adapt_tconv = ProprioAdaptTConv()

        # 3. Policy State Encoder
        #    - Role: Combines the general observation information (obs) with the environment latent vector processed above,
        #            to generate the 'state' to be used for determining the final action and value.
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

        # --- Encoder Usage and Information Combination Part ---
        if self.priv_info:
            # Stage 2: Adaptation phase (using proprioceptive history encoder)
            if self.priv_info_stage2:
                # 2. The proprioceptive history encoder 'infers' environment information (extrin)
                extrin = self.adapt_tconv(obs_dict["proprio_hist"])

                # 1. The privileged information encoder generates 'ground truth' environment information (extrin_gt)
                #    (priv_info is provided only during training)
                extrin_gt = (
                    self.env_mlp(obs_dict["priv_info"])
                    if "priv_info" in obs_dict
                    else extrin
                )
                extrin_gt = torch.tanh(extrin_gt)
                extrin = torch.tanh(extrin)

                # Combine the inferred environment information (extrin) with the general observation (obs)
                obs = torch.cat([obs, extrin], dim=-1)
            # Stage 1: PPO learning phase (using only the privileged information encoder)
            else:
                # 1. The privileged information encoder 'directly' generates environment information (extrin)
                extrin = self.env_mlp(obs_dict["priv_info"])
                extrin = torch.tanh(extrin)

                # Combine the generated environment information with the general observation (obs)
                obs = torch.cat([obs, extrin], dim=-1)

        # 3. The policy state encoder converts the combined information into a final 'state' vector
        x = self.actor_mlp(obs)

        # Use the final state vector to output the value and action (mu)
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
