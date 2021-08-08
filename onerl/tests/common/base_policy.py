import torch
from torch import nn

from onerl.utils.batch.cuda import BatchCuda


class BasePolicy(nn.Module):
    def __init__(self,
                 network: dict,
                 env_params: dict,
                 **kwargs):
        super().__init__()
        self.env_params = env_params

    def forward(self, obs: torch.Tensor):
        """
        Compute the action based on the observation

        :param obs:
        :return:
        """
        pass

    def learn(self, batch: BatchCuda):
        """
        Update the policy

        :param batch:
        :return:
        """
        pass

    def preprocess(self, batch: BatchCuda, buffer):
        """
        Calculate the return of each episode in the batch

        :param batch:
        :param buffer:
        :return: Modified batch
        """
        pass
