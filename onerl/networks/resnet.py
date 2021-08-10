import torch
from torch import nn
import torch.nn.functional as F


class PreactResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_bn: bool = True):
        super(PreactResBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_bn)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        shortcut = self.downsample(x) if hasattr(self, "downsample") else x

        y = x
        y = self.conv1(F.relu(self.bn1(y)))
        y = self.conv2(F.relu(self.bn2(y)))
        return y + shortcut


class ResnetEncoder(nn.Module):
    def __init__(self,
                 in_shape: tuple,
                 num_layers: int = 3,
                 start_channels: int = 16,
                 use_bn: bool = True):
        super().__init__()
        # network architecture
        # initial conv
        layers = [nn.Conv2d(in_shape[0], start_channels, kernel_size=3, stride=1, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(start_channels))
        # res blocks
        last_channels = num_channels = start_channels
        for idx in range(num_layers):
            layers.append(PreactResBlock(last_channels, num_channels, 2, use_bn))
            layers.append(PreactResBlock(num_channels, num_channels, 1, use_bn))
            last_channels = num_channels
            num_channels *= 2

        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.layers(x)
        x = self.avgpool(x).view(x.shape[0], -1)
        return x
