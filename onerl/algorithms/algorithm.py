from collections import OrderedDict

import torch
from torch import nn

from onerl.utils.batch.cuda import BatchCuda


class Algorithm(nn.Module):
    def __init__(self,
                 network: dict,
                 env_params: dict):
        super().__init__()
        self.network = nn.ModuleDict(network)
        self.env_params = env_params

    def forward(self, obs: torch.Tensor, ticks: int) -> torch.Tensor:
        raise NotImplementedError("Algorithm: forward not implemented.")

    def learn(self, batch: BatchCuda, ticks: int) -> dict:
        raise NotImplementedError("Algorithm: learn not implemented.")

    def policy_state_dict(self) -> OrderedDict:
        raise NotImplementedError("Algorithm: policy_state_dict not implemented.")
