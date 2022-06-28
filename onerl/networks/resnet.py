import torch
from torch import nn
import torch.nn.functional as F

from onerl.networks.norm_layer import normalization_layer


class PreactResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 norm_type: str,
                 groups: int):
        super(PreactResBlock, self).__init__()

        use_bias = norm_type == "none"
        self.bn1 = normalization_layer(in_channels, norm_type, groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=use_bias)
        self.bn2 = normalization_layer(out_channels, norm_type, groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias)

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
                 in_channels: int,
                 frame_stack: bool = True,
                 # Network architecture
                 # [2 * resblock(num_channels[0]), 2 * resblock(num_channels[1]), , ...]
                 num_channels: int = [16, 32, 64],
                 norm_type: str = "batch_norm",
                 # Group norm groups, only used in group_norm
                 groups: int = 8):
        super().__init__()
        self.frame_stack = frame_stack

        # network architecture
        # initial conv
        start_channels = num_channels[0]
        use_bias = norm_type == "none"
        layers = [
            nn.Conv2d(in_channels, start_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            normalization_layer(start_channels, norm_type, groups)
        ]
        # res blocks
        last_channels = start_channels
        for channels in num_channels:
            layers.append(PreactResBlock(last_channels, channels, 2, norm_type, groups))
            layers.append(PreactResBlock(channels, channels, 1, norm_type, groups))
            last_channels = channels

        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if self.frame_stack:
            # frame stack, reshape N FS C H W --> N C*FS H W
            x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        # uint8 --> float
        if x.dtype is torch.uint8:
            x = x.to(torch.float) / 255

        x = self.layers(x)
        x = self.avgpool(x).view(x.shape[0], -1)
        return x
