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

    def forward(self, obs: torch.Tensor, ticks: int):
        pass

    def learn(self, batch: BatchCuda):
        pass

    def policy_state_dict(self):
        pass
