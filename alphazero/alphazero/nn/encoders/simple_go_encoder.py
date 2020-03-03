import torch
from torch import nn

from .base import GameStateEncoder


class GoSimpleConvNetEncoder(GameStateEncoder):
    def __init__(self, board_size: int):
        super().__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(3, 5, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(5)
        self.linear1 = nn.Linear(81, 16)

    def forward(self, game_state: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        x = self.conv1(game_state)
        x = self.bn1(x)
        x = x.reshape(x.size())
        x = self.linear1(x)
        return x
