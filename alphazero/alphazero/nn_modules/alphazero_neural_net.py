from typing import Tuple

import torch
from torch import nn

from .loss_function import AlphaZeroLoss


class AlphaZeroNeuralNet(nn.Module):
    """
    A neural network for AlphaZero algorithms.
    Takes a GameState as input, and outputs a state value and
    a estimated policy for state s. The loss function is defined
    in `loss_function.py`.
    """

    def __init__(self,
                 encoder: nn.Module,
                 policy_head: nn.Module,
                 value_head: nn.Module) -> None:
        """
        Constructs an AlphaZeroNN instance.
        :param encoder: module to encode a state into an IR
        :param policy_head: network to project the IR into a policy vector
        :param value_head: network to project the IR into a value estimate
        """
        super().__init__()
        self._loss_fn = AlphaZeroLoss()
        self._encoder = encoder
        self._policy_head = policy_head
        self._value_head = value_head

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=arguments-differ
        encoded = self._encoder(state)
        policy = self._policy_head(encoded)
        value = self._value_head(encoded)
        return policy, value
