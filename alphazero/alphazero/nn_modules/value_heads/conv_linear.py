import torch
import torch.nn.functional as F
from torch import nn


class ConvLinearValueHead(nn.Module):
    def __init__(self,
                 game_size: int,
                 in_channels: int,
                 hidden_dim: int,
                 num_conv_filters: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_conv_filters, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_conv_filters)
        self.fc1 = nn.Linear(num_conv_filters * game_size * game_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(-1, self.fc1.in_features)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.tanh(x)
