from typing import Optional

import torch
from torch import nn
import numpy as np

from onerl.algorithms.algorithm import Algorithm
from onerl.utils.batch.cuda import BatchCuda


class WMZeroAlgorithm(Algorithm):
    def __init__(self,
                 network: dict,
                 env_params: dict,
                 # Exploration
                 exploration_len: int = 10000,
                 noise_scale: float = 0.2,
                 # WM
                 rollout_steps: int = 10,
                 gamma: float = 0.99,
                 # learning rates
                 lr_wm: float = 0.001,
                 lr_policy: float = 0.001,
                 # unused
                 batch_size: Optional[int] = None,
                 replay_buffer_size: Optional[int] = None
                 ):
        super().__init__(network, env_params)
        # action
        assert "act_max" in self.env_params, "WMZeroAlgorithm: Environment actions must be continuous."
        assert np.isclose(self.env_params["act_max"], 1).all(), "WMZeroAlgorithm: Actions must be in [-1, 1]."
        # exploration
        self.exploration_len = exploration_len
        self.noise_scale = noise_scale
        # WM
        self.rollout_steps = rollout_steps
        self.gamma = gamma
        # models
        self.max_h = sum([k.startswith("T") for k in self.network.keys()])
        self.wm = [{k: self.network[k + str(h)] for k in ["T", "E", "V"]} for h in range(self.max_h)]
        # optimizers
        self.wm_optimizer = [torch.optim.Adam(sum([net.parameters() for net in self.wm[h].values()], []), lr=lr_wm)
                             for h in range(self.max_h)]
        self.policy_optimizer = torch.optim.Adam(self.network["P"].parameters(), lr=lr_policy)

    def policy_state_dict(self):
        policy = self.network["policy"]
        return policy.module.state_dict() if hasattr(policy, "module") else policy.state_dict()

    def learn(self, batch: BatchCuda):
        # Abstract MDP
        for h in range(self.max_h):
            with torch.no_grad():
                if h == 0:
                    e = self.wm[h]["E"](batch.data["obs"][:, 0])
                    target_a = torch.transpose(batch.data["act"], 0, 1)
                    target_r = torch.transpose(batch.data["rew"], 0, 1)
                else:
                    # rollout using last model
                    for t in range(self.rollout_steps):
                        pass
