import torch
from torch import nn

from onerl.utils.batch.cuda import BatchCuda


class Algorithm(nn.Module):
    def __init__(self,
                 network: dict,
                 env_params: dict):
        super().__init__()
        self.network = network
        self.env_params = env_params

    def forward(self, obs: torch.Tensor):
        pass

    def learn(self, batch: BatchCuda):
        pass

    def serialize_policy(self):
        pass

    def deserialize_policy(self, data):
        pass
