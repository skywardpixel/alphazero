import torch
import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, num_filters) -> None:
        super().__init__()
        self.conv = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        residual = x
        x = self.bn(self.conv(x))
        x = x + residual
        x = F.relu(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels: int, num_filters: int, num_blocks: int) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_filters)
        self.res_tower = nn.Sequential(*([ResBlock(num_filters)] * num_blocks))

    def forward(self, encoded_state: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        x = encoded_state
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = F.relu(x)
        x = self.res_tower(x)
        return x
