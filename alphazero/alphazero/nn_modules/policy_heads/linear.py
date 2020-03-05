from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch import nn


class LinearPolicyHead(nn.Module):
    def __init__(self, action_space_size: int, config: Dict[str, Any]) -> None:
        super().__init__()
        self.fc_out = nn.Linear(config['encoding_dim'], action_space_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        x = self.fc_out(x)
        x = F.softmax(x, dim=-1)
        return x
