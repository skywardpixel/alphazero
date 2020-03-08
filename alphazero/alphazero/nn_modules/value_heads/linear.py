import torch
import torch.nn.functional as F
from torch import nn


class LinearValueHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_out = nn.Linear(config['encoding_dim'], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        x = self.fc_out(x)
        return F.tanh(x)
