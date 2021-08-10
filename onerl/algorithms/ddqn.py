from copy import deepcopy

import torch
from torch import nn

from onerl.algorithms.algorithm import Algorithm
from onerl.utils.batch.cuda import BatchCuda


class DDQN(Algorithm):
    def __init__(
        self, network: dict, env_params: dict,
        # hyper-parameters
        lr: float,
        gamma: float,
        target_update_freq: int,
        # exploration
        eps_start: float,
        eps_final: float,
        eps_final_steps: float
    ):
        super().__init__(network, env_params)
        # action
        assert "act_n" in self.env_params, "DDQN: Environment actions must be discrete."
        self.act_n = self.env_params["act_n"]
        # hyper-parameters
        self.lr = lr
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        # exploration
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_final_steps = eps_final_steps
        # target network
        self.target_iter = 0
        self.target_network = {k: deepcopy(v) for k, v in self.network.items()}
        for v in self.target_network.values():
            v.eval()

    def forward(self, obs: torch.Tensor):
        with torch.no_grad():
            feature = self.network["feature_extractor"](obs)
            q = self.network["critic"](feature)

        return act

    def learn(self, batch: BatchCuda):
        pass

    def serialize_policy(self):
        pass

    def deserialize_policy(self, data):
        pass
