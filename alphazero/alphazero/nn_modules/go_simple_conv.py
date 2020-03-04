import torch
import torch.nn.functional as F
from torch import nn


class GoSimpleConvNetEncoder(nn.Module):
    def __init__(self, board_size: int, in_channels: int, output_dim: int):
        super().__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(in_channels, 5, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(5)
        self.fc_out = nn.Linear(5 * board_size * board_size, output_dim)

    def forward(self, game_state: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        x = self.conv1(game_state)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.view(-1, 5 * self.board_size * self.board_size)
        x = self.fc_out(x)
        x = F.relu(x)
        return x
