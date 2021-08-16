from typing import Optional

import torch
from torch import nn


def ortho_linear_layer(*args, **kwargs):
    linear = nn.Linear(*args, **kwargs)

    torch.nn.init.orthogonal_(linear.weight)
    torch.nn.init.zeros_(linear.bias)
    return linear


class MLP(nn.Module):
    """
    MLP with orthogonal init, GELU and batch norm
    """
    def __init__(self,
                 input_dims: int,
                 output_dims: Optional[int] = None,
                 num_hidden: Optional[list] = None,
                 use_bn: bool = True):
        super().__init__()

        layers = []
        # hidden
        last_dims = input_dims
        if num_hidden is not None:
            for layer_size in num_hidden:
                layers.append(ortho_linear_layer(last_dims, layer_size))
                layers.append(nn.GELU())
                if use_bn:
                    layers.append(nn.BatchNorm1d(layer_size))

                last_dims = layer_size

        # output
        if output_dims is not None:
            layers.append(ortho_linear_layer(last_dims, output_dims))
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
