from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn


class SimpleConvNetEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.board_size = config['game_size']
        self.conv1 = nn.Conv2d(config['num_history'], 5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(0.1)
        self.fc_out = nn.Linear(5 * self.board_size * self.board_size,
                                config['encoding_dim'])

    def forward(self, game_state: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        x = self.conv1(game_state)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.view(-1, 5 * self.board_size * self.board_size)
        x = self.fc_out(x)
        x = F.relu(x)
        return x
