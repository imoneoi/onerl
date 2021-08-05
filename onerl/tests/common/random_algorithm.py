import torch
from torch import nn

from onerl.utils.batch.cuda import BatchCuda


class RandomAlgorithm(nn.Module):
    def __init__(self,
                 network: dict,
                 env_params: dict,
                 **kwargs):
        super().__init__()
        self.env_params = env_params

    def forward(self, obs: torch.Tensor):
        if "act_n" in self.env_params:
            return torch.randint(0, self.env_params["act_n"], (obs.shape[0], ))
        else:
            # uniform -act_max ... act_max
            return self.env_params["act_max"] * (torch.rand(obs.shape[0], *self.env_params["act_shape"]) * 2 - 1)

    def learn(self, batch: BatchCuda):
        pass
