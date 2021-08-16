from typing import Optional
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal

from onerl.algorithms.algorithm import Algorithm
from onerl.utils.batch.cuda import BatchCuda


class SACAlgorithm(Algorithm):
    def __init__(
            self, network: dict, env_params: dict,
            # hyper-parameters
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            lr_alpha: float = 3e-4,
            gamma: float = 0.99,
            tau: float = 0.005,
            # exploration
            target_entropy: Optional[float] = None,
            # unused
            batch_size: Optional[int] = None,
            replay_buffer_size: Optional[int] = None
    ):
        super().__init__(network, env_params)
        # action
        assert "act_max" in self.env_params, "SACAlgorithm: Environment actions must be continuous."
        assert np.isclose(self.env_params["act_max"], 1).all(), "SACAlgorithm: Actions must be in [-1, 1]."
        # hyper-parameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_alpha = lr_alpha
        self.gamma = gamma
        # alpha
        self.alpha = 1.0
        self.log_alpha = torch.nn.Parameter(torch.zeros(()))
        self.target_entropy = target_entropy if target_entropy else -np.prod(self.env_params["act_shape"])
        # target networks
        self.tau = tau
        self.target_network = nn.ModuleDict({k: deepcopy(self.network[k])
                                             for k in ["feature_extractor", "critic1", "critic2"]})
        self.target_network.train()  # target network BN train mode, to stabilize Q learning
        # optimizer
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)
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
                feature: Optional[torch.Tensor] = None,
                return_log_prob: bool = False):
        # network forward
        if feature is None:
            feature = self.network["feature_extractor"](obs)
        mu, sigma = self.network["actor"](feature)

        # calculate action distribution
        log_prob = None
        if ticks is None:
            # deterministic eval mode
            act = mu
        else:
            # sample from normal
            dist = Independent(Normal(mu, sigma), -1)
            act = dist.rsample()
            if return_log_prob:
                log_prob = dist.log_prob(act).unsqueeze(-1)

        # action squashing
        squashed_act = torch.tanh(act)
        if return_log_prob:
            # apply squashing correction to log prob
            # You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
            # in appendix C to get some understanding of this equation.
            log_prob = log_prob - torch.log((1 - squashed_act ** 2) + self.eps).sum(-1, keepdim=True)
            return squashed_act, log_prob

        return squashed_act

    def sync_weight(self):
        pass

    def learn(self, batch: BatchCuda):
        # TODO: WARNING: DistributedDataParallel enabled here
        with torch.no_grad():
            next_obs = batch.data["obs"][:, 1:]

            next_obs_feature = self.network["feature_extractor"](next_obs)
            next_act, next_log_prob = self(None, -1, feature=next_obs_feature, return_log_prob=True)
            next_q = torch.min(
                self.network["critic1"](next_obs_feature, next_act),
                self.network["critic2"](next_obs_feature, next_act)
            ) - self.alpha * next_log_prob

            # update target
            update_target = batch.data["rew"][:, -2] + self.gamma * (1. - batch.data["done"][:, -2]) * next_q

        # critic q learning
        obs_feature = self.network["feature_extractor"](batch.data["obs"][:, :-1])
        act = batch.data["act"][:, -2]
        q1 = self.network["critic1"](obs_feature, act)
        q2 = self.network["critic2"](obs_feature, act)
        # q mean (for visualization)
        q1_mean = torch.mean(q1)
        q2_mean = torch.mean(q2)
        # back prop
        q1_loss = torch.mean((q1 - update_target) ** 2)
        q2_loss = torch.mean((q2 - update_target) ** 2)
        q_loss = (q1_loss + q2_loss) / 2

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # actor policy gradient
        # stop feature extractor gradient
        obs_feature_detached = obs_feature.detach()
        actor_act, actor_log_prob = self(None, -1, feature=obs_feature_detached, return_log_prob=True)
        current_q1a = self.network["critic1"](obs_feature_detached, actor_act)
        current_q2a = self.network["critic2"](obs_feature_detached, actor_act)
        actor_loss = torch.mean(self.alpha * actor_log_prob - torch.min(current_q1a, current_q2a))
        # q_a mean (for visualization)
        q1_a_mean = torch.mean(current_q1a)
        q2_a_mean = torch.mean(current_q2a)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # alpha optimizer
        alpha_loss = -torch.mean(self.log_alpha * (actor_log_prob.detach() + self.target_entropy))
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.detach().exp()

        # sync weight
        self.sync_weight()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),

            "alpha": self.alpha,
            "q1_mean": q1_mean.item(),
            "q2_mean": q2_mean.item(),
            "q1_a_mean": q1_a_mean.item(),
            "q2_a_mean": q2_a_mean.item()
        }
