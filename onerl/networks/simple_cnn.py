import torch
from torch import nn


class SimpleCNNEncoder(nn.Module):
    def __init__(self,
                 in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten())

    def forward(self, x):
        # reshape N FS C H W --> N C*FS H W
        if len(x.shape) == 5:
            x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        # uint8 --> float
        if x.dtype is torch.uint8:
            x = x.to(torch.float) / 255

        return self.net(x)
