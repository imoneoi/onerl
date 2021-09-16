from typing import Optional

import torch
from torch import nn


class ResMLPBlock(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        # self.layer_norm = nn.LayerNorm(dims)
        self.linear = nn.Linear(dims, dims)
        self.activation = nn.ReLU()

    def forward(self, x):
        shortcut = x

        # x = self.layer_norm(x)
        x = self.linear(x)
        x = self.activation(x)
        return x + shortcut


class ResMLP(nn.Module):
    """Pre-LN MLP

    On Layer Normalization in the Transformer Architecture
    https://openreview.net/forum?id=B1x8anVFPr
    """
    def __init__(self,
                 input_dims: int,
                 output_dims: Optional[int] = None,
                 hidden_layers: int = 2,
                 hidden_dims: int = 256):
        super().__init__()
        layers = []
        if input_dims != hidden_dims:
            # projection layer
            layers.append(nn.Linear(input_dims, hidden_dims))
        for _ in range(hidden_layers):
            layers.append(ResMLPBlock(hidden_dims))
        if output_dims is not None:
            # output dims
            layers.append(nn.Linear(hidden_dims, output_dims))
        self.layers = nn.Sequential(*layers)

    def forward(self, *inputs):
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=-1)
        else:
            x = inputs[0]

        if len(x.shape) == 3:
            # N FS C --> N FS*C
            x = x.view(x.shape[0], -1)

        return self.layers(x)
