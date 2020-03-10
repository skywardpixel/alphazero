from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn


class SimpleFullyConnectedPolicyHead(nn.Module):
    def __init__(self, action_space_size: int, config: Dict[str, Any]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config['encoding_dim'], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, action_space_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc_out(x)
        x = F.log_softmax(x, dim=-1)
        return x
