from collections import OrderedDict

import torch

from onerl.algorithms.algorithm import Algorithm
from onerl.utils.batch.cuda import BatchCuda


class RandomAlgorithm(Algorithm):
    def __init__(self,
                 network: dict,
                 env_params: dict,
                 **kwargs):
        super().__init__(network, env_params)

    def forward(self, obs: torch.Tensor, ticks: int) -> torch.Tensor:
        if "act_n" in self.env_params:
            # discrete action space
            return torch.randint(0, self.env_params["act_n"], (obs.shape[0], ))
        else:
            # uniform -act_max ... act_max
            return self.env_params["act_max"] * (torch.rand(obs.shape[0], *self.env_params["act_shape"]) * 2 - 1)

    def learn(self, batch: BatchCuda, ticks: int) -> dict:
        return {}

    def policy_state_dict(self) -> OrderedDict:
        return OrderedDict()
