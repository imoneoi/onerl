from tkinter.messagebox import NO
from typing import Optional
from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn
import numpy as np

from onerl.algorithms.algorithm import Algorithm
from onerl.utils.batch.cuda import BatchCuda


class TD3Algorithm(Algorithm):
    def __init__(self,
                 network: dict,
                 env_params: dict,
                 # hyper-parameters
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 update_actor_freq: int = 2,
                 start_steps: int = 25000,
                 # exploration
                 noise_std: float = 0.1,
                 # unused
                 batch_size: Optional[int] = None,
                 replay_buffer_size: Optional[int] = None
                ):
        super().__init__(network, env_params)
        # action
        assert "act_max" in self.env_params, "TD3Algorithm: Environment actions must be continuous."
        assert np.isclose(self.env_params["act_max"], 1).all(), "TD3Algorithm: Actions must be in [-1, 1]."
        # TODO: Action scaling
        # hyper-parameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau

        self.iter = 0
        self.update_actor_freq = update_actor_freq

        self.start_steps = start_steps
        # exploration
        self.noise_std = noise_std
        # target networks
        self.target_network = nn.ModuleDict({k: deepcopy(self.network[k])
                                             for k in ["feature_extractor", "critic1", "critic2"]})
        self.target_network.train()  # target network BN train mode, to stabilize Q learning
        # optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.network["actor"].parameters(),
            lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(
            sum([list(self.network[k].parameters()) for k in ["feature_extractor", "critic1", "critic2"]], []),
            lr=self.lr_critic)

    def train(self, mode: bool = True):
        self.training = mode
        self.network.train(mode)
        return self

    def forward(self,
                obs: torch.Tensor,
                ticks: int,
                feature: Optional[torch.Tensor] = None) -> torch.Tensor:
        # pure random start
        if (ticks is not None) and (ticks < self.start_steps):
            # (BS, ACT_SHAPE)
            return torch.rand((obs.shape[0], *self.env_params["act_shape"])) * 2 - 1

        # feature
        if feature is None:
            feature = self.network["feature_extractor"](obs)
        # act
        act = torch.tanh(self.network["actor"](feature))
        # exploration noise
        if ticks is not None:
            act = torch.clip(act + self.noise_std * torch.randn_like(act), -1, 1)
        return act

    def sync_weight(self):
        # Ref: stable baselines 3
        # https://github.com/DLR-RM/stable-baselines3/blob/914bc10a0dd7b522172e538771a69055853ecf94/stable_baselines3/common/utils.py#L393
        with torch.no_grad():
            for k, dst in self.target_network.items():
                src = self.network[k]
                for d, s in zip(dst.parameters(), src.parameters()):
                    d.data.mul_(1 - self.tau)
                    torch.add(d.data, s.data, alpha=self.tau, out=d.data)

    def policy_state_dict(self) -> OrderedDict:
        # state dict of networks
        result = OrderedDict()
        for net_name in ["feature_extractor", "actor"]:
            net = self.network[net_name]
            net_state_dict = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
            result.update(OrderedDict({"{}.{}".format(net_name, k): v
                                       for k, v in net_state_dict.items()}))

        return result

    def learn(self, batch: BatchCuda, ticks: int) -> dict:
        # start training at start_steps
        if ticks < self.start_steps:
            return {}

        # TODO: WARNING: DistributedDataParallel enabled here
        with torch.no_grad():
            next_obs = batch.data["obs"][:, 1:]
            
            # with target policy smoothing (ticks=ticks, exploration noise)
            next_act = self(obs=next_obs, ticks=ticks)

            next_obs_feature = self.target_network["feature_extractor"](next_obs)
            next_q = torch.min(
                self.target_network["critic1"](next_obs_feature, next_act).view(-1),
                self.target_network["critic2"](next_obs_feature, next_act).view(-1)
            )

            # update target
            update_target = batch.data["rew"][:, -2] + self.gamma * (1. - batch.data["done"][:, -2]) * next_q

        # critic q learning
        obs_feature = self.network["feature_extractor"](batch.data["obs"][:, :-1])
        act = batch.data["act"][:, -2]
        q1 = self.network["critic1"](obs_feature, act).view(-1)
        q2 = self.network["critic2"](obs_feature, act).view(-1)
        # back prop
        q1_loss = torch.mean((q1 - update_target) ** 2)
        q2_loss = torch.mean((q2 - update_target) ** 2)
        q_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # sync weight
        self.sync_weight()

        metrics = {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item()
        }

        self.iter += 1
        if self.iter % self.update_actor_freq == 0:
            # actor quasi-policy gradient
            # stop feature extractor gradient
            obs_feature_detached = obs_feature.detach()
            actor_act = self(obs=None, ticks=None, feature=obs_feature_detached)  # no exploration noise (ticks=None)
            current_qa = self.network["critic1"](obs_feature_detached, actor_act).view(-1)
            actor_loss = -torch.mean(current_qa)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            metrics["actor_loss"] = actor_loss.item()

        return metrics
