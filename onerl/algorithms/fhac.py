from typing import Optional
from collections import OrderedDict
import time

import torch
import numpy as np

from onerl.algorithms.algorithm import Algorithm
from onerl.utils.batch.cuda import BatchCuda


class FHACAlgorithm(Algorithm):
    def __init__(
        self, network: dict, env_params: dict,
        # hyper-parameters
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        h: int = 64,
        # lam: float,
        # exploration
        exploration_len: int = 10000,
        noise_scale: float = 0.2,
        # unused
        batch_size: int = None,
        replay_buffer_size: int = None
    ):
        super().__init__(network, env_params)
        # action
        assert "act_max" in self.env_params, "FHACAlgorithm: Environment actions must be continuous."
        assert np.isclose(self.env_params["act_max"], 1).all(), "FHACAlgorithm: Actions must be in [-1, 1]."
        # hyper-parameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.h = h
        # self.lam = lam
        # exploration
        self.noise_scale = noise_scale
        self.exploration_len = exploration_len
        # optimizer
        self.actor_optimizer = torch.optim.Adam(self.network["actor"].parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(
            sum([list(self.network[k].parameters()) for k in ["feature_extractor", "critic"]], []),
            lr=self.lr_critic)

    def forward(self,
                obs: torch.Tensor,
                ticks: int,
                feature: Optional[torch.Tensor] = None):
        # random exploration
        if (ticks is not None) and (ticks < self.exploration_len):
            return torch.rand(self.env_params["act_shape"]) * 2 - 1

        # feature
        if feature is None:
            feature = self.network["feature_extractor"](obs)
        # actor forward
        act = torch.tanh(self.network["actor"](feature))
        # exploration
        if ticks is not None:
            act = torch.clamp(act + self.noise_scale * torch.randn(act.shape, device=act.device), -1, 1)
        return act

    def policy_state_dict(self):
        # state dict of networks
        result = OrderedDict()
        for net_name in ["feature_extractor", "actor"]:
            net = self.network[net_name]
            net_state_dict = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
            result.update(OrderedDict({"{}.{}".format(net_name, k): v
                                       for k, v in net_state_dict.items()}))

        return result

    def learn(self, batch: BatchCuda):
        # Part I. critic FHQ learning
        with torch.no_grad():
            next_obs = batch.data["obs"][:, 1:]
            next_obs_feature = self.network["feature_extractor"](next_obs)

            next_a = self(None, int(1e9), feature=next_obs_feature)  # target policy smoothing
            next_q = self.network["critic"](next_obs_feature, next_a)

            # update target
            update_target = batch.data["rew"][:, -2].unsqueeze(-1) + \
                            (1. - batch.data["done"][:, -2]).unsqueeze(-1) * torch.cat([torch.zeros(next_q.shape[0], 1, device=next_q.device), next_q[:, :-1]], -1)

        obs_feature = self.network["feature_extractor"](batch.data["obs"][:, :-1])
        act = batch.data["act"][:, -2]
        q = self.network["critic"](obs_feature, act)
        # q mean (for visualization)
        with torch.no_grad():
            q_mean = torch.mean(q)
        # back prop
        q_weight = torch.arange(self.h, 0, -1, device=q.device).unsqueeze(0) / self.h
        q_loss = torch.mean((q_weight * (q - update_target)) ** 2)
        # q_reg_loss = torch.mean((q_weight[:, 1:] * (q[:, 1:] - q.detach()[:, :-1])) ** 2)
        # q_tot_loss = q_loss + self.lam * q_reg_loss

        self.critic_optimizer.zero_grad()
        # q_tot_loss.backward()
        q_loss.backward()
        self.critic_optimizer.step()

        # Part II. Actor learning
        obs_feature_detached = obs_feature.detach()
        act = self(None, None, feature=obs_feature_detached)
        q_a = self.network["critic"](obs_feature_detached, act)
        actor_loss = -torch.mean(q_weight * q_a)
        # actor_loss = -torch.mean(q_a)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "q_loss": q_loss.item(),
            # "q_reg_loss": q_reg_loss.item(),
            "q_mean": q_mean.item(),

            "actor_loss": actor_loss.item(),
        }
