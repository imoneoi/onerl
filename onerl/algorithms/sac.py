from typing import Optional
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from torch import nn

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
        self.act_max = self.env_params["act_max"]
        # hyper-parameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_alpha = lr_alpha
        self.gamma = gamma
        # alpha
        actor_device = list(self.network["actor"].parameters())[0].device
        self.log_alpha = torch.nn.Parameter(torch.zeros((), device=actor_device))
        self.target_entropy = target_entropy if target_entropy else -np.prod(self.env_params["act_shape"])
        # target networks
        self.tau = tau
        self.target_network = nn.ModuleDict({k: deepcopy(self.network[k])
                                             for k in ["feature_extractor", "critic1", "critic2"]})
        self.target_network.eval()
        # optimizer
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)
        self.actor_optimizer = torch.optim.Adam(
            self.network["actor"].parameters(),
            lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(
            sum([list(self.network[k].parameters()) for k in ["feature_extractor", "critic1", "critic2"]], []),
            lr=self.lr_critic)
