import torch
from torch import nn


def normalization_layer(num_channels: int, norm_type: str, groups: int, is_2d: bool = True):
    if norm_type == "batch_norm":
        return nn.BatchNorm2d(num_channels) if is_2d else nn.BatchNorm1d(num_channels)
    elif norm_type == "group_norm":
        return nn.GroupNorm(groups, num_channels)
    elif norm_type == "layer_norm":
        assert is_2d is False, "LayerNorm can only be used on MLP layers."
        return nn.LayerNorm(num_channels)
    elif norm_type == "none":
        return nn.Identity()
    else:
        assert False, "Unknown normalization type: {}".format(norm_type)
