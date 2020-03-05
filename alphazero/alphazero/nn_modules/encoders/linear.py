from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn


class LinearEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.board_size = config['game_size']
        self.in_channels = config['num_history']
        self.fc_out = nn.Linear(self.in_channels * self.board_size * self.board_size,
                                config['encoding_dim'])

    def forward(self, game_state: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        x = game_state
        x = x.view(-1, self.in_channels * self.board_size * self.board_size)
        x = self.fc_out(x)
        x = F.relu(x)
        return x
