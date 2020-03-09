import torch
import torch.nn.functional as F
from torch import nn


class ConvLinearPolicyHead(nn.Module):
    def __init__(self,
                 game_size: int,
                 in_channels: int,
                 output_dim: int,
                 num_conv_filters: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_conv_filters, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_conv_filters)
        self.fc = nn.Linear(num_conv_filters * game_size * game_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(-1, self.fc.in_features)
        x = self.fc(x)
        return F.softmax(x, dim=-1)
