import torch
from torch import nn
import numpy as np


class GRURecurrent(nn.Module):
    def __init__(self,
                 input_dims: int,

                 hidden_dims: int = 256,
                 num_layers: int = 1):
        # TODO: Dropout ?
        super().__init__()
        self.hidden_dims = hidden_dims

        self.gru = nn.GRU(input_size=input_dims,

                          hidden_size=hidden_dims,
                          num_layers=num_layers,

                          batch_first=True)

    def recurrent_state_shape(self):
        return (self.hidden_dims,), np.float32

    def forward(self, x: torch.Tensor, rstate: torch.Tensor):
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

        # x: (BS, T, C)
        # rstate: (BS, C)
        output, h_n = self.gru(x, rstate.unsqueeze(0))

        # output, h_n
        # output: (BS, T, C) of hidden states
        # h_n: new rstate
        return output, h_n.squeeze(0)
